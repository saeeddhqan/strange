
import torch, math
import torch.nn.functional as F
from torch import nn, Tensor
from typing import NoReturn, ClassVar, Union, Optional, Tuple
from transformers import AutoTokenizer 


config = None

class Data:
	def __init__(self, config: ClassVar) -> NoReturn:
		split = 8
		data = torch.tensor([])
		for chunk in range(split + 1):
			data = torch.cat((data, torch.load(f'{config.data_file}_p{chunk}.pt')))
		data = data.to(torch.long)

		model = 'mistralai/Mistral-7B-v0.1'
		self.tokenizer = AutoTokenizer.from_pretrained(model)

		self.encode = self.tokenizer.encode
		self.decode = lambda seq: self.tokenizer.decode(seq, skip_special_tokens=True)

		self.vocab_size = self.tokenizer.vocab_size
		config.vocab_size = self.vocab_size

		train_split = int(0.9 * len(data))

		self.train_data = data[:train_split]
		self.test_data = data[train_split:]

		self.block_size = config.block_size
		self.batch_size = config.batch_size


	def __len__(self) -> int:
		return self.vocab_size


	def get_batch(self, 
		idx: int, split: str = 'train',
		block_size: int = None,
		batch_size: int = -1,
	) -> tuple[Tensor, Tensor]:
		block_size = self.block_size if block_size is None else block_size
		batch_size = self.batch_size if batch_size == -1 else batch_size

		data = self.train_data if split == 'train' else self.test_data
		ix = torch.randint(len(data) - block_size, (batch_size,))
		x = torch.stack([data[i:i + block_size] for i in ix])
		y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
		return (x.pin_memory().to(config.device, non_blocking=True),
				y.pin_memory().to(config.device, non_blocking=True),
		)


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


class Attention(nn.Module):
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
		if self.pos_method == 'dynamic':
			self.pos_win = config.pos_win
			self.dim_snip = self.dim // self.pos_win
			self.pos_coef = nn.Parameter(torch.tensor(data=0.9))
			self.lnq = RMSNorm(self.dim)
			self.lnk = RMSNorm(self.dim)
		self.flash = config.flash_attention
		if not self.flash:
			self.register_buffer('bias', torch.tril(torch.ones(self.block_size, self.block_size))
										.view(1, 1, self.block_size, self.block_size))


	def create_dype(self, x: Tensor) -> Tensor:
		snip = x[:,:,:self.dim_snip].flatten(1)
		snip = F.pad(snip, (self.dim - self.dim_snip, 0), value=1.0)
		pos_emb = snip.unfold(1, self.dim, self.dim_snip)
		return pos_emb


	def create_dype_v2(self, x: Tensor) -> Tensor:
		snip = x[:,:,:self.dim_snip].flatten(1)
		snip = F.pad(snip, (self.dim - self.dim_snip, 0), value=1.0)
		pos_emb = snip.unfold(1, self.dim, self.dim_snip)
		# Blend
		heads = a.view(x.size(0), x.size(1), self.nheads, self.hsize)
		comb = heads.transpose(2, 3).contiguous().view(x.size(0), x.size(1), -1)
		return comb


	def forward(self,
		x: Tensor,
		freqs_cis: Optional[Tensor] = None,
	) -> Tuple[Tensor, None]:

		B, T, C = x.size()
		q, k, v  = self.c_attn(x).split(self.dim, dim=2)

		if self.pos_method == 'dynamic':
			dype = self.create_dype_v2(v) * self.pos_coef
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
		return y


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
		self.attn = Attention(self.idx)

	def forward(self,
		x: Tensor,
		freqs_cis: Union[Tensor, None] = None,
	) -> Tuple[Tensor, Union[Tensor, None]]:

		x = x + self.attn(self.ln1(x), freqs_cis=freqs_cis)
		x = x * self.alpha + self.ffn(self.ln2(x))
		return x


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
		n_params -= self.stack.tok_embs.weight.numel()
		return n_params


	def norm_weights(self, module):
		if isinstance(module, nn.Linear):
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

		for i, block in enumerate(self.blocks):
			x = block(x, freqs_cis=freqs_cis)

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
