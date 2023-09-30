
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import time
from torch import Tensor

def set_seed(seed=1234):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed()

block_size = 64
nheads = 4
nlayers = 4
dim = 128
bias = False


class RMSNorm(nn.Module):
	def __init__(self, dim: int, eps: float = 1e-5):
		super().__init__()
		self.eps = eps
		self.weight = nn.Parameter(torch.ones(dim))
		# self.bias = nn.Parameter(torch.zeros(dim))

	def _norm(self, x):
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

	def forward(self, x):
		output = self._norm(x.float()).type_as(x)
		return (output * self.weight)


class NonLinear(nn.Module):
	'''Based on llama2 code'''
	def __init__(self):
		super().__init__()
		self.dim = dim
		self.w1 = nn.Linear(self.dim, 4 * self.dim, bias=bias) # bias=False in llama
		self.w2 = nn.Linear(4 * self.dim, self.dim, bias=bias) # bias=False in llama
		self.w3 = nn.Linear(self.dim, 4 * self.dim, bias=bias) # bias=False in llama

	def forward(self, x: Tensor):
		'''
			Init forward method.
		'''
		return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
	def __init__(self, idx):
		super().__init__()
		assert dim % nheads == 0, 'embeds size is not divisible to the nheads'
		self.dim = dim
		head_size = self.dim // nheads
		self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=False)
		self.c_proj = nn.Linear(self.dim, self.dim, bias=False)
		self.attn_dropout = nn.Dropout(0.1)
		self.resid_dropout = nn.Dropout(0.1)
		self.n_head = nheads
		self.dropout = 0.1
		self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size))
									.view(1, 1, block_size, block_size))

	def forward(self, x):
		B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
		q, k, v  = self.c_attn(x).split(self.dim, dim=2)
		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

		y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
			attn_mask=None,
			dropout_p=self.dropout if self.training else 0,
			is_causal=True,
		)

		y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

		# output projection
		y = self.resid_dropout(self.c_proj(y))
		return y


class CausalSelfAttention2(nn.Module):
	def __init__(self, idx: int):
		super().__init__()
		assert dim % nheads == 0, 'embeds size is not divisible to the nheads'
		self.idx = idx
		self.dim = dim
		self.nheads = nheads
		self.hsize = self.dim // self.nheads
		# self.v_attn = nn.Linear(self.dim, self.dim, bias=bias)
		self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=bias)
		self.c_proj = nn.Linear(self.dim, self.dim, bias=bias)
		self.dropout = 0.0
		self.resid_dropout = nn.Dropout(self.dropout)
		self.n_groups = 4
		self.group_t = (block_size // self.n_groups) # tokens per group
		self.its_time = nlayers % 2 ^ ((self.idx + 1) % 2)
		self.odd_even = nlayers % 2

	def do_att(self, q, k, v, group=False):
		return torch.nn.functional.scaled_dot_product_attention(q, k, v, 
				attn_mask=None,
				dropout_p=self.dropout if (self.training and not group) else 0,
				is_causal=True,
			)

	def do_block_merge(self, xblock, x):
		other_blocks = torch.cat((xblock, x[:,:,1:,:]), dim=3)
		first_block = torch.cat((x[:,:,:1], xblock[:,:,:1,-1:]), dim=3)
		x = torch.cat((first_block, other_blocks), dim=2)
		return x

	def forward(self, x, y):
		'''
			# Add rotary to synthetic tokens
			# Use a decay on groups
			# it does not support arbitrary window size. Only fixed window size
		'''
		B, T, C = x.size()
		n_groups = min(T // self.group_t, self.n_groups)
		v = F.silu(self.v_attn(x))
		q, k  = self.c_attn(v).split(self.dim, dim=2)

		# change shape (B, T, C) to (B, nh, ng, pg, C)
		q = q.view(B, n_groups, self.group_t, self.nheads, self.hsize).permute(0, 3, 1, 2, 4)
		k = k.view(B, n_groups, self.group_t, self.nheads, self.hsize).permute(0, 3, 1, 2, 4)
		v = v.view(B, n_groups, self.group_t, self.nheads, self.hsize).permute(0, 3, 1, 2, 4)
		if self.its_time and n_groups > 0:
			q = torch.cat((q, q.mean(dim=3).unsqueeze(3)), dim=3)
			k = torch.cat((k, k.mean(dim=3).unsqueeze(3)), dim=3)
			v = torch.cat((v, v.mean(dim=3).unsqueeze(3)), dim=3)
		elif y is not None and n_groups > 0:
			q = self.do_block_merge(y[0], q)
			k = self.do_block_merge(y[1], k)
			v = self.do_block_merge(y[2], v)

		x = self.do_att(q, k, v)
		if self.its_time and n_groups > 0:
			# remove last block from q, k, v
			q, k = q[:,:,:-1,-1], k[:,:,:-1,-1]
			v = self.do_att(
				q,
				k,
				x[:,:,:-1,-1],
				group=True,
			).unsqueeze(3)
			y = (q.unsqueeze(3), k.unsqueeze(3), v)
			x = x[:,:,:,:-1] # crop footprints(blocks)
		else:
			if x.size(3) > self.group_t and n_groups > 0:
				x = torch.cat((x[:,:,:1,:-1], x[:,:,1:,1:]), dim=2)
			y = None
		x = x.contiguous().view(B, self.nheads, x.size(2) * x.size(3), self.hsize).transpose(2, 1).contiguous().view(B, T, C)
		x = self.resid_dropout(self.c_proj(x))
		return x, y


class CausalSelfAttention3(nn.Module):
	def __init__(self, idx: int):
		super().__init__()
		assert dim % nheads == 0, 'embeds size is not divisible to the nheads'
		self.idx = idx
		self.dim = dim
		self.nheads = nheads
		self.hsize = self.dim // self.nheads
		# self.v_attn = nn.Linear(self.dim, self.dim, bias=bias)
		self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=bias)
		self.c_proj = nn.Linear(self.dim, self.dim, bias=bias)
		self.dropout = 0.0
		self.resid_dropout = nn.Dropout(self.dropout)
		self.n_groups = 4
		self.group_t = (block_size // self.n_groups) # tokens per group
		self.its_time = nlayers % 2 ^ ((self.idx + 1) % 2)
		self.odd_even = nlayers % 2

	def do_att(self, q, k, v, group=False):
		return torch.nn.functional.scaled_dot_product_attention(q, k, v, 
				attn_mask=None,
				dropout_p=self.dropout if (self.training and not group) else 0,
				is_causal=True,
			)

	def do_block_merge(self, xblock, x):
		other_blocks = torch.cat((xblock, x[:,:,1:,:]), dim=3)
		first_block = torch.cat((x[:,:,:1], xblock[:,:,:1,-1:]), dim=3)
		x = torch.cat((first_block, other_blocks), dim=2)
		return x

	def forward(self, x, y):
		'''
			# Add rotary to synthetic tokens
			# Use a decay on groups
			# it does not support arbitrary window size. Only fixed window size
		'''
		B, T, C = x.size()
		n_groups = min(T // self.group_t, self.n_groups)
		q, k, v  = self.c_attn(x).split(self.dim, dim=2)

		if self.pos_method == 'rope':
			q = q.view(B, T, self.n_head, self.hsize)
			k = k.view(B, T, self.n_head, self.hsize)
			q, k = apply_rotary_emb(q, k, freqs_cis)
			q = q.view(B, T, C)
			k = k.view(B, T, C)
		# Change shape (B, T, C) to (B, nh, ng, gt, C)
		q = q.view(B, n_groups, self.group_t, self.nheads, self.hsize).permute(0, 3, 1, 2, 4)
		k = k.view(B, n_groups, self.group_t, self.nheads, self.hsize).permute(0, 3, 1, 2, 4)
		v = v.view(B, n_groups, self.group_t, self.nheads, self.hsize).permute(0, 3, 1, 2, 4)
		if self.its_time and n_groups > 0:
			# Create and add synthetic tokens
			if y is None:
				qm, km, vm = q.mean(dim=3).unsqueeze(3), k.mean(dim=3).unsqueeze(3), v.mean(dim=3).unsqueeze(3)
				q = torch.cat((q, qm), dim=3)
				k = torch.cat((k, km), dim=3)
				v = torch.cat((v, vm), dim=3)
				q = self.do_block_merge(qm[:,:,:-1,:], q)
				k = self.do_block_merge(km[:,:,:-1,:], k)
				v = self.do_block_merge(vm[:,:,:-1,:], v)
			else:
				q = torch.cat((q, y[0]), dim=3)
				k = torch.cat((k, y[1]), dim=3)
				v = torch.cat((v, y[2]), dim=3)
		elif y is not None and n_groups > 0:
			# Embed synthetic tokens at the beginning of blocks so that tokens can communicate with it
			q = self.do_block_merge(y[0][:,:,:-1,-1:], q)
			k = self.do_block_merge(y[1][:,:,:-1,-1:], k)
			v = self.do_block_merge(y[2][:,:,:-1,-1:], v)
		x = self.do_att(q, k, v)
		if self.its_time and n_groups > 0:
			# One communication between synthetic tokens to share information between groups
			# v = self.do_att(
			# 	q[:,:,:,-1],
			# 	k[:,:,:,-1],
			# 	x[:,:,:,-1],
			# 	group=True,
			# ).unsqueeze(3)
			v = torch.nn.functional.scaled_dot_product_attention(q[:,:,1:,0], k[:,:,:-1,-1], v[:,:,:-1,-1], 
				attn_mask=None,
				dropout_p=config.dropout if (self.training and not group) else 0,
				is_causal=True,
			)

			y = (q[:,:,:,-1:], k[:,:,:,-1:], v)
			# y = self.block_drop(y[0]), self.block_drop(y[1]), self.block_drop(y[2])
			x = x[:,:,:,:-1] # crop footprints(blocks)
		else:
			# If true, then remove synthetic tokens to clean the sequence.
			if x.size(3) > self.group_t and n_groups > 0:
				x = torch.cat((x[:,:,:1,:-1], x[:,:,1:,1:]), dim=2)
			# y = None
		x = x.contiguous().view(B, self.nheads, x.size(2) * x.size(3), self.hsize).transpose(2, 1).contiguous().view(B, T, C)
		x = self.resid_dropout(self.c_proj(x))
		return x, y


class Block(nn.Module):
	def __init__(self, idx: int):
		super().__init__()
		self.dim = dim
		self.heads = CausalSelfAttention2(idx)
		self.ffn = NonLinear()
		self.ln1 = RMSNorm(self.dim)
		self.ln2 = RMSNorm(self.dim)
		# self.rescon1 = nn.Parameter(torch.ones(1, 1, self.dim))
		# self.rescon2 = nn.Parameter(torch.ones(1, 1, self.dim))

	def forward(self, x, y):
		# hidden_state = hidden_state + self.heads(self.ln1(hidden_state))
		# hidden_state, y = self.heads(self.ln1(x), y)
		hidden_state, y1 = self.heads(self.ln1(x), y)
		hidden_state2, y2 = self.heads.forward2(self.ln1(x), y)
		# hidden_state = self.rescon1 * x + hidden_state 
		# hidden_state = self.rescon2 * hidden_state + self.ffn(self.ln2(hidden_state))
		print(hidden_state[0,0])
		print(y1[0][0,0,0])
		print(hidden_state2[0,0])
		print(y2[0][0,0,0])
		return hidden_state, y1


class layers(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = nn.ModuleList([Block(idx) for idx in range(nlayers)])
		self.ln = RMSNorm(dim)

	def forward(self, x):
		y = None
		for k, block in enumerate(self.layers):
			print(x[0][0].mean(dim=0).item(), x[0][0].abs().mean(dim=0).item())
			x, y = block(x, y)
		print(x[0][0])

# a = layers().cuda()
# b = torch.rand(3, block_size, dim).cuda()
# a(b)
##################################################################
a = layers().cuda()
complete = torch.rand(64, block_size, dim).cuda()
a(complete)
print('passed complete')
# arbit = torch.rand(3, 50, dim).cuda()
start = torch.rand(3, 5, dim).cuda()
# barely = torch.rand(2, 10, dim).cuda()
# barely2 = torch.rand(3, 40, dim).cuda()
# barely3 = torch.rand(3, 30, dim).cuda()

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
# for param in a.parameters():
	# print(param.norm(2).item())
	# print(param.mean(dim=-1))
	# if param.grad is not None:
		# print(param[1].grad)

# t0 = time.time()
# for i in range(1000):
# 	a(complete)
# t1 = time.time()
# print(t1-t0)