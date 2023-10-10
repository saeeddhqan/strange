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
num_layers = 25
dim = 128
bias = False
mode = 'train'

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


class CausalSelfAttention(nn.Module):
	def __init__(self, idx: int):
		super().__init__()
		assert dim % nheads == 0, 'embeds size is not divisible to the nheads'
		self.dim = dim
		self.nheads = nheads
		self.dropout = 0.1
		self.hsize = self.dim // self.nheads
		self.block_size = block_size

		self.c_attn = nn.Linear(self.dim, 3 * self.dim)
		self.c_proj = nn.Linear(self.dim, self.dim)

		self.attn_dropout = nn.Dropout(self.dropout)
		self.resid_dropout = nn.Dropout(self.dropout)
		self.k, self.v = None, None

	def forward(self, x, k, v):
		B, T, C = x.size()
		q, k, v  = self.c_attn(x).split(self.dim, dim=2)
		k = k.view(B, T, self.nheads, self.hsize).transpose(1, 2)
		q = q.view(B, T, self.nheads, self.hsize).transpose(1, 2)
		v = v.view(B, T, self.nheads, self.hsize).transpose(1, 2)
		y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
			attn_mask=None,
			dropout_p=self.dropout if self.training else 0,
			is_causal=True,
		)
		y = y.transpose(1, 2).contiguous().view(B, T, C)
		y = self.resid_dropout(self.c_proj(y))
		self.k, self.v = k, v
		return y

class SecondSelfAttention(nn.Module):
	def __init__(self, idx):
		super().__init__()
		assert dim % nheads == 0, 'embeds size is not divisible to the nheads'
		self.idx = idx
		self.dim = dim
		self.hsize = self.dim // nheads

		self.c_attn_q = nn.Linear(self.dim, self.dim, bias=False)
		self.c_proj = nn.Linear(self.dim, self.dim, bias=False)
		self.attn_dropout = nn.Dropout(0.1)
		self.resid_dropout = nn.Dropout(0.1)

		self.n_head = nheads
		self.dropout = 0.1
		self.k = None
		self.v = None
		self.scale = 1.0 / math.sqrt(self.hsize)


	def forward(self, x, k, v):
		B, T, C = x.size()
		train = mode == 'train'

		q = self.c_attn_q(x if train else x[:,-1:])
		q = q.view(B, T if train else 1, self.n_head, C // self.n_head).transpose(1, 2)

		if train:
			y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
				attn_mask=None,
				dropout_p=self.dropout if self.training else 0,
				is_causal=True,
			)
		else:
			att = (q * k).sum(-1).unsqueeze(2) * self.scale
			att = F.softmax(att, dim=-1)
			y = att @ v

		y = y.transpose(1, 2).contiguous().view(B, T if train else 1, C)
		y = self.resid_dropout(self.c_proj(y))
		return y


class NonLinear(nn.Module):
	def __init__(self):
		super().__init__()
		self.dim = dim
		self.w1 = nn.Linear(self.dim, 4 * self.dim, bias=bias)
		self.w2 = nn.Linear(4 * self.dim, self.dim, bias=bias)
		self.w3 = nn.Linear(self.dim, 4 * self.dim, bias=bias)

	def forward(self, x: Tensor):
		return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
	def __init__(self, idx: int):
		super().__init__()
		self.idx = idx
		self.dim = dim
		self.heads = CausalSelfAttention(idx) if idx == 0 else SecondSelfAttention(idx)
		self.ffn = NonLinear()
		self.ln1 = RMSNorm(self.dim)
		self.ln2 = RMSNorm(self.dim)

	def forward(self, x, k=None, v=None):
		hidden_state = x + self.heads(self.ln1(x), k, v)
		hidden_state = hidden_state + self.ffn(self.ln2(hidden_state))
		return hidden_state


class layers(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = nn.ModuleList([Block(idx) for idx in range(num_layers)])
		self.ln = RMSNorm(dim)

	def forward(self, x):
		k, v = None, None
		for idx, layer in enumerate(self.layers):
			# print(x[0][0].mean(dim=0).item(), x[0][0].abs().mean(dim=0).item())
			x = self.ln(layer(x, k, v))
			if idx == 0:
				k, v = layer.heads.k, layer.heads.v
			# print(sx.shape, x.norm(2).item())
		# print(x.shape)
		# print(x[0][0])

# a = layers().cuda()
# b = torch.rand(3, block_size, dim).cuda()
# a(b)
##################################################################
a = layers().cuda()
complete = torch.rand(64, block_size, dim).cuda()
# a(complete)
# print('passed complete')

# for param in a.parameters():
# 	# print(param)
# 	# print(param[1].norm(2).item())
# 	# print(param[1].mean(dim=-1))
# 	# print(dir(param))
# 	# exit()
# 	# print(param.grad)
# 	if param.grad is not None:
# 		print(param[1].grad)
t0 = time.time()
for i in range(50):
	a(complete)
	print(i, end='\r')
t1 = time.time()
print('train:', t1-t0)
mode = 'test'
t0 = time.time()
for i in range(50):
	a(complete)
	print(i, end='\r')
t1 = time.time()
print('test:', t1-t0)