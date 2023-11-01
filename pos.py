
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import NoReturn, ClassVar, Union, Optional, Tuple
from transformers import AutoTokenizer 
import random, numpy
import math

def set_seed(seed: int):
	random.seed(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed(1244)


dim = 8
seq_len = 18
max_seq_len = 9

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0) -> Tensor:
	freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
	t = torch.arange(seq_len, device=freqs.device)  # type: ignore
	freqs = torch.outer(t, freqs).float()  # type: ignore
	freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
	return freqs_cis

def precompute_freqs_cis_ntk(dim: int, seq_len: int, max_seq_len: int, theta: float = 10000.0) -> Tensor:
	if seq_len > max_seq_len:
		alpha = 8.0
		theta = theta * ((alpha * seq_len / max_seq_len) - (alpha - 1)) ** (dim / (dim - 2))
	freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
	t = torch.arange(seq_len, device=freqs.device)  # type: ignore
	freqs = torch.outer(t, freqs).float()  # type: ignore
	freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
	return freqs_cis

def apply_rotary_emb_mine(
	xq: Tensor,
	xk: Tensor,
	freqs_cis: Tensor,
) -> Tuple[Tensor, Tensor]:

	xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
	xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

	# Reshape for broadcast
	ndim = xq_.ndim
	assert 0 <= 1 < ndim
	assert freqs_cis.shape == (xq_.shape[2], xq_.shape[-1])
	freqs_cis = freqs_cis.view(1, 1, xq_.size(2), xq_.size(3))

	xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
	xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
	return xq_out.type_as(xq), xk_out.type_as(xk)

def apply_rotary_emb(
	xq: Tensor,
	xk: Tensor,
	freqs_cis: Tensor,
) -> Tuple[Tensor, Tensor]:

	xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
	xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

	# Reshape for broadcast
	ndim = xq_.ndim
	assert 0 <= 1 < ndim
	assert freqs_cis.shape == (xq_.shape[1], xq_.shape[-1])
	shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
	freqs_cis = freqs_cis.view(*shape)

	xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
	xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
	return xq_out.type_as(xq), xk_out.type_as(xk)

freqs = precompute_freqs_cis(dim, max_seq_len * 2)
# freqs = precompute_freqs_cis_ntk(dim, seq_len, max_seq_len)

print(freqs.shape)

q2 = torch.ones((1, 4, seq_len, dim))
k2 = torch.ones((1, 4, seq_len, dim))
q = torch.ones((1, seq_len, 4, dim))
k = torch.ones((1, seq_len, 4, dim))
print('-'*30)
q2, k2 = apply_rotary_emb_mine(q2, k2, freqs[:q2.size(2)])
print(q2.transpose(1, 2)[0,2])
q, k = apply_rotary_emb(q, k, freqs[:q.size(1)])
print(q[0,2])

# print(torch.view_as_complex(r.float().view(*r.shape[:-1], -1, 2))[0,:,0])
# print(q[0,0])
# print(q[:,0,0])
# print(q[:,0,1])
# print(q[:,0,2])
# print(q[:,0,3])
# print(q[:,0,4])
# print(q[:,0,5])
# print(q[:,0,6])
# print('-'*30)