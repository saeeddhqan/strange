
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import NoReturn, ClassVar, Union, Optional, Tuple
import math


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
	):
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


class CausalSelfAttention(nn.Module):
	def __init__(self, idx: int):
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

		self.flash = config.flash_attention
		if not self.flash:
			self.register_buffer('bias', torch.tril(torch.ones(self.block_size, self.block_size))
										.view(1, 1, self.block_size, self.block_size))

	def forward(self,
		x: Tensor,
		y: None,
		freqs_cis: Optional[Union[Tensor, None]] = None,
		) -> Tuple[Tensor, None]:
		B, T, C = x.size()
		q, k, v  = self.c_attn(x).split(self.dim, dim=2)

		if self.pos_method == 'rope':
			q = q.view(B, T, self.nheads, self.hsize)
			k = k.view(B, T, self.nheads, self.hsize)
			v = v.view(B, T, self.nheads, self.hsize).transpose(1, 2)
			q, k = apply_rotary_emb(q, k, freqs_cis)
			q = q.transpose(1, 2)
			k = k.transpose(1, 2)
		else:
			k = k.view(B, T, self.nheads, self.hsize).transpose(1, 2)
			q = q.view(B, T, self.nheads, self.hsize).transpose(1, 2)
			v = v.view(B, T, self.nheads, self.hsize).transpose(1, 2)

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
			# att = self.attn_dropout(att)
			y = att @ v
		
		y = y.transpose(1, 2).contiguous().view(B, T, C)

		y = self.resid_dropout(self.c_proj(y))
		return y, None


class CausalSelfAttention2(nn.Module):
	def __init__(self, idx: int):
		super().__init__()
		assert config.dim % config.nheads == 0, 'embeds size is not divisible to the nheads'
		self.idx = idx
		self.dim = config.dim
		self.nheads = config.nheads
		self.pos_method = config.pos
		self.hsize = self.dim // self.nheads

		self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=config.bias)
		self.c_proj = nn.Linear(self.dim, self.dim, bias=config.bias)

		self.dropout = config.dropout
		self.resid_dropout = nn.Dropout(self.dropout)
		self.block_drop = nn.Dropout(self.dropout)

		self.n_groups = config.ngroups
		self.group_t = (config.block_size // self.n_groups) # tokens per group
		self.its_time = config.nlayers % 2 ^ ((self.idx + 1) % 2)

	def do_att(self, q: Tensor, k: Tensor, v: Tensor, group: bool = False) -> Tensor:
		return torch.nn.functional.scaled_dot_product_attention(q, k, v, 
			attn_mask=None,
			dropout_p=config.dropout if (self.training and not group) else 0,
			is_causal=True,
		)

	def do_block_merge(self, xblock: Tensor, x: Tensor) -> Tensor:
		other_blocks = torch.cat((xblock, x[:,:,1:,:]), dim=3)
		first_block = torch.cat((x[:,:,:1], xblock[:,:,:1,-1:]), dim=3)
		x = torch.cat((first_block, other_blocks), dim=2)
		return x

	def forward(self,
		x: Tensor,
		y: Union[Tensor, None] = None,
		freqs_cis: Union[Tensor, None] = None,
	) -> Tensor:
		B, T, C = x.size()
		n_groups = min(T // self.group_t, self.n_groups)
		q, k, v  = self.c_attn(x).split(self.dim, dim=2)
		
		if self.pos_emb == 'rope':
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
			q = torch.cat((q, q.mean(dim=3).unsqueeze(3)), dim=3)
			k = torch.cat((k, k.mean(dim=3).unsqueeze(3)), dim=3)
			v = torch.cat((v, v.mean(dim=3).unsqueeze(3)), dim=3)
		elif y is not None and n_groups > 0:
			# Embed synthetic tokens at the beginning of blocks so that tokens can communicate with it
			q = self.do_block_merge(y[0], q)
			k = self.do_block_merge(y[1], k)
			v = self.do_block_merge(y[2], v)

		x = self.do_att(q, k, v)
		if self.its_time and n_groups > 0:
			# remove last block from q, k, v
			q, k = q[:,:,:-1,-1], k[:,:,:-1,-1]
			# One communication between synthetic tokens to share information between groups
			v = self.do_att(
				q,
				k,
				x[:,:,:-1,-1],
				group=True,
			).unsqueeze(3)
			y = (q.unsqueeze(3), k.unsqueeze(3), v)
			y = self.block_drop(y[0]), self.block_drop(y[1]), self.block_drop(y[2])
			x = x[:,:,:,:-1] # crop footprints(blocks)
		else:
			# If true, then remove synthetic tokens to clean the sequence.
			if x.size(3) > self.group_t and n_groups > 0:
				x = torch.cat((x[:,:,:1,:-1], x[:,:,1:,1:]), dim=2)
			y = None
		x = x.contiguous().view(B, self.nheads, x.size(2) * x.size(3), self.hsize).transpose(2, 1).contiguous().view(B, T, C)
		x = self.resid_dropout(self.c_proj(x))
		return x, y


class NonLinear(nn.Module):
	def __init__(self):
		super().__init__()
		self.dim = config.dim
		self.w1 = nn.Linear(self.dim, 4 * self.dim, bias=config.bias)
		self.w2 = nn.Linear(self.dim, 4 * self.dim, bias=config.bias)
		self.w3 = nn.Linear(4 * self.dim, self.dim, bias=config.bias)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x: Tensor):
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
		self.causal_self_attention = CausalSelfAttention2(self.idx) if config.attention == 2 else CausalSelfAttention(self.idx)

	def forward(self,
		x: Tensor,
		y: Union[Tensor, None],
		freqs_cis: Union[Tensor, None] = None,
	) -> Tuple[Tensor, Union[Tensor, None]]:

		head_out, y = self.causal_self_attention(self.ln1(x), y, freqs_cis=freqs_cis)
		head_out = x + head_out
		hidden_state = head_out + self.ffn(self.ln2(head_out))
		return hidden_state, y


class Transformer(nn.Module):
	def __init__(self) -> NoReturn:
		super().__init__()
		self.dim = config.dim
		self.pos_method = config.pos
		self.ngroups = config.ngroups
		self.pos_win = config.pos_win
		self.dim_snip = self.dim // self.pos_win

		if self.pos_method == 'rope':
			self.register_buffer('freqs_cis', precompute_freqs_cis(self.dim // config.nheads, config.block_size * 2)) # double for making it dynamism
		else:
			self.freqs_cis = None

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
		self.pos_coef = nn.Parameter(torch.tensor(data=0.5)) if self.pos_method == 'dynamic' else None

		self.apply(self.norm_weights)
		if config.deepnorm:
			self._deepnorm()

		self.count_params = self.num_params() / 1e6
		config.parameters = self.count_params

		print("Number of parameters: %.2fM" % (self.count_params,))


	def _deepnorm(self):
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
		x = self.stack.tok_embs(seq) # (B,T,C)

		# Dynamic pos embedding
		if self.pos_method == 'dynamic':
			pos_emb = x[:,:,:self.dim_snip].flatten(1) # (B, n)
			pos_emb = F.pad(pos_emb, (self.dim - self.dim_snip, 0), value=0) # (B, n+)
			pos_emb = self.stack.dropout_pos(
				pos_emb.unfold(1, self.dim, self.dim_snip) * self.pos_coef,
			) # (B, T, C)
		elif self.pos_method == 'learnable':
			arange = torch.arange(T, device=seq.device)
			pos_emb = self.stack.pos_embs(arange)

		x = x + pos_emb if self.pos_method != 'rope' else x

		freqs_cis = None if self.pos_method != 'rope' else self.freqs_cis[:T].to(seq.device)
		x = self.stack.dropout(x)

		y = None
		for i, block in enumerate(self.blocks):
			x, y = block(x, y, freqs_cis=freqs_cis)

		if targets is None:
			x = x[:,-1]

		x = self.stack.ln1(x)
		logits = self.stack.lm_head(x) # (batch, block_size, vocab_size)

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
