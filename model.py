
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

	def _norm(self, x):
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

	def forward(self, x):
		output = self._norm(x.float()).type_as(x)
		return (output * self.weight)


class CausalSelfAttention(nn.Module):
	def __init__(self, idx: int):
		super().__init__()
		assert config.dim % config.nheads == 0, 'embeds size is not divisible to the nheads'
		self.dim = config.dim
		self.nheads = config.nheads
		self.dropout = 0.1
		self.hsize = self.dim // self.nheads
		self.block_size = config.block_size

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
		assert config.dim % config.nheads == 0, 'embeds size is not divisible to the nheads'
		self.idx = idx
		self.dim = config.dim
		self.hsize = self.dim // config.nheads

		self.c_attn_q = nn.Linear(self.dim, self.dim, bias=False)
		self.c_proj = nn.Linear(self.dim, self.dim, bias=False)
		self.attn_dropout = nn.Dropout(0.1)
		self.resid_dropout = nn.Dropout(0.1)

		self.n_head = config.nheads
		self.dropout = config.dropout
		self.scale = 1.0 / math.sqrt(self.hsize)


	def forward(self, x, k, v):
		B, T, C = x.size()
		train = config.mode == 'train'

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
		self.dim = config.dim
		self.w1 = nn.Linear(self.dim, 4 * self.dim, bias=config.bias)
		self.w2 = nn.Linear(4 * self.dim, self.dim, bias=config.bias)
		self.w3 = nn.Linear(self.dim, 4 * self.dim, bias=config.bias)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x: Tensor):
		return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
	def __init__(self, idx: int, alpha: float):
		super().__init__()
		self.idx = idx
		self.dim = config.dim
		self.heads = CausalSelfAttention(idx) if idx == 0 else SecondSelfAttention(idx)
		self.ffn = NonLinear()
		self.ln1 = RMSNorm(self.dim)
		self.ln2 = RMSNorm(self.dim)

	def forward(self, x, k=None, v=None):
		hidden_state = x + self.heads(self.ln1(x), k, v)
		hidden_state = hidden_state + self.ffn(self.ln2(hidden_state))
		return hidden_state


class Transformer(nn.Module):
	def __init__(self) -> NoReturn:
		super().__init__()
		self.dim = config.dim
		self.stack = nn.ModuleDict(dict(
			tok_embs=nn.Embedding(config.vocab_size, self.dim),
			pos_embs=nn.Embedding(config.block_size, self.dim),
			dropout=nn.Dropout(config.dropout),
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
				 nn.init.constant_(module.bias, 0.0)
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
		tok_emb = self.stack.tok_embs(seq) # (batch, block_size, embed_dim) (B,T,C)

		arange = torch.arange(T, device=seq.device)
		pos_emb = self.stack.pos_embs(arange)

		x = tok_emb + pos_emb

		x = self.stack.dropout(x)
		k, v = None, None
		for idx, block in enumerate(self.blocks):
			x = block(x, k, v)
			if idx == 0:
				k, v = block.heads.k, block.heads.v

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
		bsize = config.block_size
		for _ in range(_len):
			idx_cond = idx if idx.size(1) <= bsize else idx[:, -bsize:]
			logits, _ = self(idx_cond)
			logits = logits / temperature
			probs = F.softmax(logits, dim=-1)
			if top_k is not None:
				v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
				logits[logits < v[:, [-1]]] = float('-inf')
			next_idx = torch.multinomial(probs, num_samples=1)
			idx = torch.cat((idx, next_idx), dim=1)
		return idx
