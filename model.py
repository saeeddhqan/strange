
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import NoReturn, ClassVar, Union, Optional, Tuple
import math, json, sentencepiece

def init_(tensor):
	dim = tensor.shape[-1]
	std = 1 / math.sqrt(dim)
	tensor.uniform_(-std, std)
	return tensor

config = None

class Data:
	def __init__(self, config: ClassVar) -> NoReturn:
		with open(config.data_file) as fp:
			text = fp.read()

		self.chars = sorted(list(set(text)))
		self.vocab_size = len(self.chars)
		config.vocab_size = self.vocab_size
		self.stoi = {c:i for i,c in enumerate(self.chars)}
		self.itos = {i:c for c,i in self.stoi.items()}
		self.encode = lambda s: [self.stoi[x] for x in s]
		self.decode = lambda e: ''.join([self.itos[x] for x in e])
		data = torch.tensor(self.encode(text), dtype=torch.long)
		train_split = int(0.9 * len(data))
		self.train_data = data[:train_split]
		self.test_data = data[train_split:]
		self.block_size = config.block_size
		self.batch_size = config.batch_size


	def __len__(self) -> int:
		return self.vocab_size


	def get_batch(self, 
		idx: int, split: str = 'train',
		block_size = None,
		batch_size: int = -1,
	) -> tuple[Tensor, Tensor]:
		block_size = self.block_size if block_size is None else block_size
		batch_size = self.batch_size if batch_size == -1 else batch_size

		data = self.train_data if split == 'train' else self.test_data
		ix = torch.randint(len(data) - block_size, (batch_size,))
		x = torch.stack([data[i:i + block_size] for i in ix])
		y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
		return x.pin_memory().to(config.device, non_blocking=True), y.pin_memory().to(config.device, non_blocking=True)


class RMSNorm(nn.Module):
	def __init__(self,
		dim: int, eps: float = 1e-5,
	) -> NoReturn:
		super().__init__()
		self.eps = eps
		self.weight = nn.Parameter(torch.ones(dim))

	def _norm(self, x) -> Tensor:
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

	def forward(self, x) -> Tensor:
		output = self._norm(x.float()).type_as(x)
		return (output * self.weight)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
	freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
	t = torch.arange(end, device=freqs.device)  # type: ignore
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
	assert freqs_cis.shape == (xq_.shape[2], xq_.shape[-1])
	freqs_cis = freqs_cis.view(1, 1, xq_.size(2), xq_.size(3))

	xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
	xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
	return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
	def __init__(self, idx: int) -> NoReturn:
		super().__init__()
		assert config.dim % config.nheads == 0, 'embeds size is not divisible to the num_heads'
		self.dim = config.dim
		self.nheads = config.nheads
		self.dropout = config.dropout
		self.pos_method = config.pos
		self.hsize = self.dim // self.nheads
		self.block_size = config.block_size

		self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=config.bias)
		self.c_proj = nn.Linear(self.dim, self.dim, bias=config.bias)
		self.attn_dropout = nn.Dropout(self.dropout)
		self.resid_dropout = nn.Dropout(self.dropout)

		if self.pos_method == 'dynamic':
			self.pos_win = config.pos_win
			self.dim_snip = self.dim // self.pos_win
			self.pos_coef = nn.Parameter(torch.tensor(data=0.5))
			self.lnq = RMSNorm(self.dim)
			self.lnk = RMSNorm(self.dim)

		self.flash = config.flash_attention
		if not self.flash:
			self.register_buffer('bias', torch.tril(torch.ones(self.block_size, self.block_size))
										.view(1, 1, self.block_size, self.block_size))


	def create_dype_v2(self, x: Tensor) -> Tensor:
		snip = x[:,:,:self.dim_snip].flatten(1)
		snip = F.pad(snip, (self.dim - self.dim_snip, 0), value=1.0)
		pos_emb = snip.unfold(1, self.dim, self.dim_snip)
		# Blend
		heads = pos_emb.view(x.size(0), x.size(1), self.nheads, self.hsize)
		comb = heads.transpose(2, 3).contiguous().view(x.size(0), x.size(1), -1)
		return comb * self.pos_coef


	def create_mean_dype(self, x: Tensor) -> Tensor:
		B, T, C = x.size()
		x = F.pad(x, (0, 0, self.pos_win, 0), mode='constant', value=0)
		x = x.flatten(1).unfold(1, (self.pos_win * C), C)
		x = x[:,:-1].view(B, T, self.pos_win, C)
		x = x.mean(dim=2)
		return x * self.pos_coef


	def forward(self,
		x: Tensor,
		y: None,
		freqs_cis: Optional[Union[Tensor, None]] = None,
	) -> Tuple[Tensor, None]:
		B, T, C = x.size()
		q, k, v  = self.c_attn(x).split(self.dim, dim=2)

		if self.pos_method == 'dynamic':
			dype = self.create_dype_v2(v)
			q = q + self.lnq(dype)
			k = k + self.lnk(dype)

		q = q.view(B, T, self.nheads, self.hsize).transpose(1, 2)
		k = k.view(B, T, self.nheads, self.hsize).transpose(1, 2)
		v = v.view(B, T, self.nheads, self.hsize).transpose(1, 2)

		if self.pos_method == 'rope':
			q, k = apply_rotary_emb(q, k, freqs_cis)

		if self.flash:
			y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
				attn_mask=None,
				dropout_p=self.dropout if self.training else 0,
				is_causal=True,
			)
		else:
			att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
			att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
			att = F.softmax(att, dim=-1)
			att = self.attn_dropout(att)
			y = att @ v
		
		y = y.transpose(1, 2).contiguous().view(B, T, C)

		y = self.resid_dropout(self.c_proj(y))
		return y, None


class CausalSelfAttention2(nn.Module):
	def __init__(self, idx: int) -> NoReturn:
		super().__init__()
		assert config.dim % config.nheads == 0, 'embeds size is not divisible to the nheads'
		self.idx = idx
		self.dim = config.dim
		self.nheads = config.nheads
		self.pos_method = config.pos
		self.hsize = self.dim // self.nheads
		self.k_apx = 1
		self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=config.bias)
		self.c_proj = nn.Linear(self.dim, self.dim, bias=config.bias)
		self.dropout = config.dropout
		self.resid_dropout = nn.Dropout(self.dropout)
		self.block_drop = nn.Dropout(self.dropout)

		if self.pos_method == 'dynamic':
			self.pos_win = config.pos_win
			self.dim_snip = self.dim // self.pos_win
			self.pos_coef = nn.Parameter(torch.tensor(data=0.8))
			self.lnq = RMSNorm(self.dim)
			self.lnk = RMSNorm(self.dim)

		self.n_groups = config.ngroups
		self.group_t = (config.block_size // self.n_groups) # tokens per group
		self.its_time = config.nlayers % 2 ^ ((self.idx + 1) % 2)
		self.q_proj = nn.Parameter(init_(torch.zeros(self.group_t, self.k_apx)))
		self.k_proj = nn.Parameter(init_(torch.zeros(self.group_t, self.k_apx)))
		self.v_proj = nn.Parameter(init_(torch.zeros(self.group_t, self.k_apx)))
		self.proj_seq_len = lambda args: torch.einsum('bhgtc,tk->bhgkc', *args)


	def create_dype_v2(self, x: Tensor) -> Tensor:
		snip = x[:,:,:self.dim_snip].flatten(1)
		snip = F.pad(snip, (self.dim - self.dim_snip, 0), value=1.0)
		pos_emb = snip.unfold(1, self.dim, self.dim_snip)
		# Blend
		heads = pos_emb.view(x.size(0), x.size(1), self.nheads, self.hsize)
		comb = heads.transpose(2, 3).contiguous().view(x.size(0), x.size(1), -1)
		return comb * self.pos_coef


	def do_att(self, q: Tensor, k: Tensor, v: Tensor, group: bool = False) -> Tensor:
		return F.scaled_dot_product_attention(q, k, v, 
			attn_mask=None,
			dropout_p=config.dropout if (self.training and not group) else 0,
			is_causal=True,
		)


	def do_block_merge(self, xblock: Tensor, x: Tensor) -> Tensor:
		other_blocks = torch.cat((xblock, x[:,:,1:,:]), dim=3)
		first_block = torch.cat((x[:,:,:1], xblock[:,:,:1]), dim=3)
		x = torch.cat((first_block, other_blocks), dim=2)
		return x

	def forward(self,
		x: Tensor,
		y: Union[Tensor, None] = None,
		freqs_cis: Union[Tensor, None] = None,
	) -> Tuple[Tensor, Union[Tensor, None]]:

		B, T, C = x.size()
		n_groups = min(T // self.group_t, self.n_groups)
		q, k, v = self.c_attn(x).split(self.dim, dim=2)

		if self.pos_method == 'dynamic':
			dype = self.create_dype_v2(v)
			q = q + self.lnq(dype)
			k = k + self.lnk(dype)

		q = q.view(B, T, self.nheads, self.hsize).transpose(1, 2).contiguous()
		k = k.view(B, T, self.nheads, self.hsize).transpose(1, 2).contiguous()

		if self.pos_method == 'rope':
			q, k = apply_rotary_emb(q, k, freqs_cis)

		# Change shape (B, T, C) to (B, nh, ng, gt, C)
		q = q.view(B, n_groups, self.group_t, self.nheads, self.hsize).permute(0, 3, 1, 2, 4)
		k = k.view(B, n_groups, self.group_t, self.nheads, self.hsize).permute(0, 3, 1, 2, 4)
		v = v.view(B, n_groups, self.group_t, self.nheads, self.hsize).permute(0, 3, 1, 2, 4)
		if self.its_time and n_groups > 0:
			qkv_projs = (self.q_proj, self.k_proj, self.v_proj)
			qb, kb, vb = map(self.proj_seq_len, zip((q[:,:,:-1], k[:,:,:-1], v[:,:,:-1]), qkv_projs))
		elif y is not None and n_groups > 0:

			q = self.do_block_merge(y[0], q)
			k = self.do_block_merge(y[1], k)
			v = self.do_block_merge(y[2], v)

		x = self.do_att(q, k, v)
		if self.its_time and n_groups > 0:
			vb = self.do_att(qb, kb, vb, group=True)
			y = (qb, kb, vb)
			y = self.block_drop(y[0]), self.block_drop(y[1]), self.block_drop(y[2])
		else:
			# If true, then remove synthetic tokens to clean the sequence.
			if x.size(3) > self.group_t and n_groups > 0:
				x = torch.cat((x[:,:,:1,:-self.k_apx], x[:,:,1:,self.k_apx:]), dim=2)
			y = None
		x = x.contiguous().view(B, self.nheads, x.size(2) * x.size(3), self.hsize).transpose(2, 1).contiguous().view(B, T, C)
		x = self.resid_dropout(self.c_proj(x))
		return x, y


class CausalSelfAttention3(nn.Module):
	def __init__(self, idx: int) -> NoReturn:
		super().__init__()
		assert config.dim % config.nheads == 0, 'embeds size is not divisible to the nheads'
		self.idx = idx
		self.dim = config.dim
		self.nheads = config.nheads
		self.pos_method = config.pos
		self.hsize = self.dim // self.nheads
		self.k_apx = 1
		self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=config.bias)
		self.c_proj = nn.Linear(self.dim, self.dim, bias=config.bias)
		self.dropout = config.dropout
		self.resid_dropout = nn.Dropout(self.dropout)
		self.block_drop = nn.Dropout(self.dropout)

		if self.pos_method == 'dynamic':
			self.pos_win = config.pos_win
			self.dim_snip = self.dim // self.pos_win
			self.pos_coef = nn.Parameter(torch.tensor(data=0.8))
			self.lnq = RMSNorm(self.dim)
			self.lnk = RMSNorm(self.dim)

		self.n_groups = config.ngroups
		self.group_t = (config.block_size // self.n_groups) # tokens per group
		self.its_time = config.nlayers % 2 ^ ((self.idx + 1) % 2)
		self.q_proj = nn.Parameter(init_(torch.zeros(self.group_t, self.k_apx)))
		self.k_proj = nn.Parameter(init_(torch.zeros(self.group_t, self.k_apx)))
		self.v_proj = nn.Parameter(init_(torch.zeros(self.group_t, self.k_apx)))
		self.proj_seq_len = lambda args: torch.einsum('bhgtc,tk->bhgkc', *args)


	def create_dype_v2(self, x: Tensor) -> Tensor:
		snip = x[:,:,:self.dim_snip].flatten(1)
		snip = F.pad(snip, (self.dim - self.dim_snip, 0), value=1.0)
		pos_emb = snip.unfold(1, self.dim, self.dim_snip)
		# Blend
		heads = pos_emb.view(x.size(0), x.size(1), self.nheads, self.hsize)
		comb = heads.transpose(2, 3).contiguous().view(x.size(0), x.size(1), -1)
		return comb * self.pos_coef


	def do_att(self, q: Tensor, k: Tensor, v: Tensor, group: bool = False) -> Tensor:
		return F.scaled_dot_product_attention(q, k, v, 
			attn_mask=None,
			dropout_p=config.dropout if (self.training and not group) else 0,
			is_causal=True,
		)


	def do_block_merge(self, xblock: Tensor, x: Tensor) -> Tensor:
		other_blocks = torch.cat((xblock, x[:,:,1:,:]), dim=3)
		first_block = torch.cat((x[:,:,:1], xblock[:,:,:1]), dim=3)
		x = torch.cat((first_block, other_blocks), dim=2)
		return x

	def forward(self,
		x: Tensor,
		y: Union[Tensor, None] = None,
		freqs_cis: Union[Tensor, None] = None,
	) -> Tuple[Tensor, Union[Tensor, None]]:

		B, T, C = x.size()
		n_groups = min(T // self.group_t, self.n_groups)
		q, k, v = self.c_attn(x).split(self.dim, dim=2)

		if self.pos_method == 'dynamic':
			dype = self.create_dype_v2(v)
			q = q + self.lnq(dype)
			k = k + self.lnk(dype)

		q = q.view(B, T, self.nheads, self.hsize).transpose(1, 2).contiguous()
		k = k.view(B, T, self.nheads, self.hsize).transpose(1, 2).contiguous()

		if self.pos_method == 'rope':
			q, k = apply_rotary_emb(q, k, freqs_cis)

		# Change shape (B, T, C) to (B, nh, ng, gt, C)
		q = q.view(B, n_groups, self.group_t, self.nheads, self.hsize).permute(0, 3, 1, 2, 4)
		k = k.view(B, n_groups, self.group_t, self.nheads, self.hsize).permute(0, 3, 1, 2, 4)
		v = v.view(B, n_groups, self.group_t, self.nheads, self.hsize).permute(0, 3, 1, 2, 4)
		if self.its_time and n_groups > 0:
			qkv_projs = (self.q_proj, self.k_proj, self.v_proj)
			qb, kb, vb = map(self.proj_seq_len, zip((q[:,:,:-1], k[:,:,:-1], v[:,:,:-1]), qkv_projs))
		elif y is not None and n_groups > 0:
			q = self.do_block_merge(y[0], q)
			k = self.do_block_merge(y[1], k)
			v = self.do_block_merge(y[2], v)

		x = torch.tensor([]).to(config.device)
		# No softmax? no scaling? how could you?
		for i in range(1 if q.size(3) > self.group_t else 0, self.group_t + (1 if q.size(3) > self.group_t else 0)):
			x = torch.cat(
				(
					x,
					# (q[:,:,:,i:i+1] * k[:,:,:,:i+1]).sum(dim=-1).unsqueeze(-1).mT @ v[:,:,:,:i + 1]
					torch.einsum('abcfe,abcde->abcf', k[:,:,:,:i+1], q[:,:,:,i:i+1]).unsqueeze(-1).mT @ v[:,:,:,:i + 1]
				),
				dim=3,
			)
		if self.its_time and n_groups > 0:
			vb = self.do_att(qb, kb, vb, group=True)
			y = (qb, kb, vb)
			y = self.block_drop(y[0]), self.block_drop(y[1]), self.block_drop(y[2])
		else:
			# If true, then remove synthetic tokens to clean the sequence.
			if x.size(3) > self.group_t and n_groups > 0:
				x = torch.cat((x[:,:,:1,:-self.k_apx], x[:,:,1:,self.k_apx:]), dim=2)
			y = None
		x = x.contiguous().view(B, self.nheads, x.size(2) * x.size(3), self.hsize).transpose(2, 1).contiguous().view(B, T, C)
		x = self.resid_dropout(self.c_proj(x))
		return x, y


class NonLinear(nn.Module):
	def __init__(self) -> NoReturn:
		super().__init__()
		self.dim = config.dim
		self.w1 = nn.Linear(self.dim, 4 * self.dim, bias=config.bias)
		self.w2 = nn.Linear(self.dim, 4 * self.dim, bias=config.bias)
		self.w3 = nn.Linear(4 * self.dim, self.dim, bias=config.bias)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x: Tensor) -> Tensor:
		return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
	def __init__(self,
		idx: int,
		alpha: float = 1.0,
	):
		super().__init__()
		assert config.dim % config.nheads == 0, 'embeds size is not divisible to the nheads'
		self.idx = idx
		self.alpha = alpha
		self.dim = config.dim
		self.head_size = self.dim // config.nheads
		self.dropout = config.dropout
		self.ffn = NonLinear()
		self.ln1 = RMSNorm(self.dim)
		self.ln2 = RMSNorm(self.dim)
		if config.attention == 1:
			self.attn = CausalSelfAttention(self.idx)
		elif config.attention == 2:
			self.attn = CausalSelfAttention2(self.idx)
		else:
			self.attn = CausalSelfAttention3(self.idx)

	def forward(self,
		x: Tensor,
		y: Union[Tensor, None],
		freqs_cis: Union[Tensor, None] = None,
	) -> Tuple[Tensor, Union[Tensor, None]]:

		head_out, y = self.attn(self.ln1(x), y, freqs_cis=freqs_cis)
		head_out = x + head_out
		hidden_state = head_out * self.alpha + self.ffn(self.ln2(head_out))
		return hidden_state, y


class Transformer(nn.Module):
	def __init__(self) -> NoReturn:
		super().__init__()
		self.dim = config.dim
		self.pos_method = config.pos
		self.freqs_cis = None
		self.freqs_cis_test = None

		if self.pos_method == 'rope':
			self.freqs_cis = precompute_freqs_cis(self.dim // config.nheads, config.block_size)


		self.stack = nn.ModuleDict(dict(
			tok_embs=nn.Embedding(config.vocab_size, self.dim),
			pos_embs=nn.Embedding(config.block_size, self.dim) if self.pos_method == 'learnable' else None,
			dropout=nn.Dropout(config.dropout),
			dropout_pos=nn.Dropout(config.dropout_pos) if self.pos_method == 'dynamic' else None,
			ln1=RMSNorm(self.dim),
			lm_head=nn.Linear(self.dim, config.vocab_size, bias=False),
		))

		self.alpha = 1.0 if not config.deepnorm else math.pow(2.0 * config.nlayers, 0.25)
		self.blocks = nn.ModuleList([Block(idx, self.alpha) for idx in range(config.nlayers)])

		self.stack.tok_embs.weight = self.stack.lm_head.weight

		self.apply(self.norm_weights)
		if config.deepnorm:
			self._deepnorm()

		self.count_params = self.num_params() / 1e6
		config.parameters = self.count_params

		print("Number of parameters: %.2fM" % (self.count_params,))


	def _deepnorm(self) -> NoReturn:
		'''
			https://arxiv.org/pdf/2203.00555.pdf
		'''
		init_scale = math.pow(8.0 * config.nlayers, 0.25)
		for name, p in self.named_parameters():
			if (
				'w1' in name
				or 'w2' in name
				or 'w3' in name
				or 'c_proj' in name
				or 'c_attn' in name
			):
				p.data.div_(init_scale)


	def num_params(self) -> int:
		n_params = sum(p.numel() for p in self.parameters())
		# n_params -= self.stack.pos_embs.weight.numel()
		n_params -= self.stack.tok_embs.weight.numel()
		return n_params


	def norm_weights(self, module):
		if isinstance(module, nn.Linear) and not config.deepnorm:
			if config.init_weight == 'normal_':
				nn.init.normal_(module.weight, mean=0.0, std=0.02)
			else:
				nn.init.xavier_uniform_(module.weight, gain=1 / math.sqrt(2))
			if module.bias is not None:
				 nn.init.constant_(module.bias, 0.001)
		elif isinstance(module, nn.Embedding):
			if config.init_weight == 'normal_':
				nn.init.normal_(module.weight, mean=0.0, std=0.02)
			else:
				nn.init.xavier_uniform_(module.weight, gain=1 / math.sqrt(2))
		elif isinstance(module, nn.LayerNorm):
			nn.init.zeros_(module.bias)
			nn.init.ones_(module.weight)


	def forward(self, 
		seq: Tensor,
		targets: Union[Tensor, None] = None,
	) -> tuple[Tensor, Tensor]:

		B, T = seq.shape
		x = self.stack.tok_embs(seq)

		freqs_cis = None if self.pos_method != 'rope' else self.freqs_cis[:T].to(seq.device)
		x = self.stack.dropout(x)

		y = None
		for i, block in enumerate(self.blocks):
			x, y = block(x, y, freqs_cis=freqs_cis)

		if targets is None:
			x = x[:,-1]

		x = self.stack.ln1(x)
		logits = self.stack.lm_head(x)

		if targets is None:
			loss = None
		else:
			logits = logits.view(-1, config.vocab_size)
			loss = F.cross_entropy(logits, targets.flatten())

		return logits, loss


	def autocomplete(self, 
		idx: Tensor,
		_len: int = 10,
		temperature: float = 1.0,
		top_k: int = None,
	) -> Tensor:
		config.mode = 'inference'
		bsize = config.block_size
		for _ in range(_len):
			idx_cond = idx if idx.size(1) <= bsize else idx[:, -bsize:]
			logits, _ = self(idx_cond)
			logits = logits / temperature
			probs = F.softmax(logits, dim=-1)
			if top_k is not None:
				v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
				logits[logits < v[:, [-1]]] = -float('Inf')
			next_idx = torch.multinomial(probs, num_samples=1)
			idx = torch.cat((idx, next_idx), dim=1)
		return idx
