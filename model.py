
import torch, math, sentencepiece, json
import torch.nn.functional as F
from torch import nn, Tensor
from typing import NoReturn, ClassVar, Union, Optional, Tuple


def init_(tensor):
	dim = tensor.shape[-1]
	std = 1 / math.sqrt(dim)
	tensor.uniform_(-std, std)
	return tensor

config = None

class Data:
	def __init__(self, config: ClassVar) -> NoReturn:
		data = torch.tensor([])
		parts = 256 + 1
		self.tokenizer = sentencepiece.SentencePieceProcessor(
			model_file='data/shakespeare-sp.model')
		self.encode = self.tokenizer.encode
		self.decode = lambda seq: self.tokenizer.decode(seq)
		self.vocab_size = self.tokenizer.vocab_size()
		with open(f'data/shakespeare.txt') as fp:
			encoded = self.encode(fp.read())
			data = torch.tensor(encoded)
		data = data.to(torch.long)
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
		block_size = None,
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

class FNetBlock(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
		return x

class Block(nn.Module):
	def __init__(self,
		idx: int,
	):
		super().__init__()
		assert config.dim % config.nheads == 0, 'embeds size is not divisible to the nheads'
		self.idx = idx
		self.dim = config.dim
		self.head_size = self.dim // config.nheads
		self.dropout = config.dropout
		self.ffn = NonLinear()
		self.ln1 = RMSNorm(self.dim)
		self.ln2 = RMSNorm(self.dim)
		self.attn = FNetBlock()

	def forward(self,
		x: Tensor,
	) -> Tuple[Tensor, Union[Tensor, None]]:

		x = x + self.attn(self.ln1(x))
		x = x + self.ffn(self.ln2(x))
		return x


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
		self.blocks = nn.ModuleList([Block(idx) for idx in range(config.nlayers)])
		self.stack.tok_embs.weight = self.stack.lm_head.weight
		self.apply(self.norm_weights)
		self.count_params = self.num_params() / 1e6
		config.parameters = self.count_params
		print("Number of parameters: %.2fM" % (self.count_params,))


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
		arange = torch.arange(T, device=seq.device)

		x = self.stack.dropout(x + self.stack.pos_embs(arange))

		for i, block in enumerate(self.blocks):
			x = block(x)

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
