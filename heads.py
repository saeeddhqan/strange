import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import time

def set_seed(seed=1234):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed()

block_size = 64
num_heads = 4
num_layers = 12
embeds_size = 128

# class CausalSelfAttention(nn.Module):
# 	def __init__(self, idx):
# 		super().__init__()
# 		assert embeds_size % num_heads == 0, 'embeds size is not divisible to the num_heads'
# 		self.dim = embeds_size
# 		head_size = self.dim // num_heads
# 		self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=False)
# 		self.c_proj = nn.Linear(self.dim, self.dim, bias=False)
# 		self.attn_dropout = nn.Dropout(0.1)
# 		self.resid_dropout = nn.Dropout(0.1)
# 		self.n_head = num_heads
# 		self.dropout = 0.1
# 		self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size))
# 									.view(1, 1, block_size, block_size))

# 	def forward(self, x):
# 		B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

# 		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
# 		q, k, v  = self.c_attn(x).split(self.dim, dim=2)
# 		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
# 		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
# 		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

# 		att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
# 		att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
# 		att = F.softmax(att, dim=-1)
# 		att = self.attn_dropout(att)
# 		y = att @ v # (B, nh, T, T) x (B, nho, T, hs) -> (B, nh, T, hs)
# 		y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

# 		# output projection
# 		y = self.resid_dropout(self.c_proj(y))
# 		return y


class CausalSelfAttention2(nn.Module):
	def __init__(self, idx):
		super().__init__()
		assert embeds_size % num_heads == 0, 'embeds size is not divisible to the num_heads'
		self.dim = embeds_size
		self.idx = idx
		self.head_size = self.dim // num_heads
		self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=False)
		self.c_proj = nn.Linear(self.dim, self.dim, bias=False)
		# self.attn_dropout = nn.Dropout(0.1)
		self.resid_dropout = nn.Dropout(0.1)
		self.n_groups = block_size // 8
		self.per_group = (block_size // self.n_groups) 
		self.n_head = num_heads
		self.n_layers = num_layers
		self.odd_even = self.n_layers % 2
		self.dropout = 0.1
		# self.register_buffer('bias', torch.tril(torch.ones(self.per_group + 1, self.per_group + 1))
		# 							.view(1, 1, 1, self.per_group + 1, self.per_group + 1))
		# self.register_buffer('bias2', torch.tril(torch.ones(self.n_groups, self.n_groups))
		# 							.view(1, 1, self.n_groups, self.n_groups))
		# self.register_buffer('bias3', torch.tril(torch.ones(self.per_group, self.per_group))
		# 							.view(1, 1, self.per_group, self.per_group))


	def do_att(self, q, k, v):
		# att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
		# att = att.masked_fill(bias == 0, float('-inf'))
		# att = F.softmax(att, dim=-1)
		# # att = self.attn_dropout(att) # Find a replacement for it.
		# y = att @ v # (B, nh, T, T) x (B, nho, T, hs) -> (B, nh, T, hs)
		# del att
		# return y
		if True:
			y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
				attn_mask=None,
				dropout_p=0,
				is_causal=True,
			)
		else:
			att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
			att = att.masked_fill(bias == 0, float('-inf'))
			att = F.softmax(att, dim=-1)
			# att = self.attn_dropout(att) # Find a replacement for it.
			y = att @ v # (B, nh, T, T) x (B, nho, T, hs) -> (B, nh, T, hs)
		return y

	def forward(self, x):
		B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
		q, k, v  = self.c_attn(x).split(self.dim, dim=2)
		odd_head = self.idx % 2
		its_time = (self.odd_even ^ odd_head)
		n_groups = self.n_groups

		if self.idx == 0 and T % self.per_group != 0:
			remain = self.per_group - (T % self.per_group) 
			comp = remain * self.dim
			T = T + remain
			pad = torch.zeros(B, remain, embeds_size).to(x.device)
			q = torch.cat((q, pad), dim=1)
			k = torch.cat((k, pad), dim=1)
			v = torch.cat((v, pad), dim=1)
			del pad, comp, remain

		n_groups = min(T // self.per_group, self.n_groups)
		q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
		k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
		v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

		if n_groups > 1:
			n_groups = T // (self.per_group + 1) if (its_time and self.idx != 0) else n_groups
			q = q.view(B, self.n_head, n_groups, -1, self.head_size) # (B, nh, ng, gs, hs)
			k = k.view(B, self.n_head, n_groups, -1, self.head_size) # (B, nh, ng, gs, hs)
			v = v.view(B, self.n_head, n_groups, -1, self.head_size) # (B, nh, ng, gs, hs)
			if not its_time and q.size(2):
				qblock = q.mean(dim=3).unsqueeze(3)
				kblock = k.mean(dim=3).unsqueeze(3)
				vblock = v.mean(dim=3).unsqueeze(3)
				q = torch.cat((q, qblock), dim=3)
				k = torch.cat((k, kblock), dim=3)
				v = torch.cat((v, vblock), dim=3)
				T += n_groups
				n_groups = min(T // self.per_group, self.n_groups)
			# bias = self.bias[:,:,:,:q.size(3),:q.size(3)]
		# else:
			# bias = self.bias3[:,:,:self.per_group,:self.per_group]

		# So far so good. let's check the following line
		x = self.do_att(q, k, v)
		# So far so good. let's check the following block
		if x.dim() > 4 and not its_time:
			bsize = q.size(2)
			y = self.do_att(
				q[:,:,:,-1],
				k[:,:,:,-1],
				x[:,:,:,-1:].view(B, self.n_head, -1, self.head_size),
				# self.bias2[:,:,:bsize,:bsize],
			).unsqueeze(3)
			r = torch.cat((y[:,:,:-1,:], x[:,:,1:,:-1]), dim=3)
			x = torch.cat((x[:,:,:1], r), dim=2)
		else:
			if self.idx != 0 and x.dim() > 4:
				x = x[:,:,:,1:]
				T -= max(0, x.size(2))
		x = x.contiguous().view(B, self.n_head, -1, x.size(-1))
		x = x.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
		# output projection
		x = self.resid_dropout(self.c_proj(x))
		return x


class Block(nn.Module):
	def __init__(self, idx: int):
		super().__init__()
		self.dim = config.embeds_size
		self.heads = CausalSelfAttention2(idx)
		self.ffn = NonLinear()
		self.ln1 = RMSNorm(self.dim)
		self.ln2 = RMSNorm(self.dim)

	def forward(self, hidden_state):
		# hidden_state = hidden_state + self.heads(self.ln1(hidden_state))
		hidden_state = self.heads(self.ln1(hidden_state))
		hidden_state = self.ffn(self.ln2(hidden_state))
		return hidden_state


class layers(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = nn.ModuleList([CausalSelfAttention2(idx) for idx in range(num_layers)])
		self.ln = nn.LayerNorm(embeds_size)

	def forward(self, x):
		for k,i in enumerate(self.layers):
			# print(x[0][0].mean(dim=0).item(), x[0][0].abs().mean(dim=0).item())
			x = i(self.ln(x))

			# print(x[0][0])

# a = layers().cuda()
# b = torch.rand(3, block_size, embeds_size).cuda()
# a(b)
##################################################################
a = layers().cuda()
complete = torch.rand(64, 64, embeds_size).cuda()
# arbit = torch.rand(3, 123, embeds_size).cuda()
# start = torch.rand(3, 5, embeds_size).cuda()
# barely = torch.rand(2, 10, embeds_size).cuda()
# barely2 = torch.rand(3, 125, embeds_size).cuda()
# barely3 = torch.rand(3, 255, embeds_size).cuda()
# a(complete)
print('passed complete')
# a(arbit)
# print('passed arbit')
# a(start)
# print('passed start')
# a(barely)
# print('passed barely')
# a(barely2)
# print('passed barely2')
# a(barely3)
# print('passed barely3')
# print(hasattr(a, 'named_parameters'))
# for param in a.named_parameters():
	# print(param[0])
	# print(param[1].norm(2).item())
	# print(param[1].mean(dim=-1))
	# print(dir(param))
	# exit()
	# print(param.grad)
    # if param.grad is not None:

t0 = time.time()
for i in range(1000):
	a(complete)
t1 = time.time()
print(t1-t0)