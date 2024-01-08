
from dataclasses import dataclass
import torch, math, sentencepiece, json
import torch.nn.functional as F
from torch import nn, Tensor
from typing import NoReturn, ClassVar, Union, Optional, Tuple


config = None


@dataclass
class ConfigMamba:
	dim: int = 128
	nlayers: int = 2
	vocab_size: int = 0
	d_state: int = 256
	expand: int = 2
	dt_rank: Union[int, str] = 'auto'
	d_conv: int = 4 
	pad_vocab_size_multiple: int = 8
	conv_bias: bool = True
	bias: bool = False

conf_mamba = ConfigMamba()

class Data:
	def __init__(self, config: ClassVar) -> NoReturn:
		if config.token_type == 'token':
			data = torch.tensor([])
			self.tokenizer = sentencepiece.SentencePieceProcessor(
				model_file='data/shakespeare-sp.model')
			self.encode = self.tokenizer.encode
			self.decode = lambda seq: self.tokenizer.decode(seq)
			self.vocab_size = self.tokenizer.vocab_size()
			with open(f'data/shakespeare.txt') as fp:
				encoded = self.encode(fp.read())
			data = torch.tensor(encoded, dtype=torch.long)
		else:
			with open(config.data_file) as fp:
				text = fp.read()
			self.chars = sorted(list(set(text)))
			self.vocab_size = len(self.chars)
			self.stoi = {c:i for i,c in enumerate(self.chars)}
			self.itos = {i:c for c,i in self.stoi.items()}
			self.encode = lambda s: [self.stoi[x] for x in s]
			self.decode = lambda e: ''.join([self.itos[x] for x in e])
			data = torch.tensor(self.encode(text), dtype=torch.long)

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

	def _norm2(self, x) -> Tensor:
		# transnormer
		return (x / torch.norm(x, p=2)) / torch.sqrt(torch.tensor(x.size(-1))).to(x.device)

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


class MambaBlock(nn.Module):
	def __init__(self):
		super().__init__()
		# self.A = nn.Linear(conf_mamba.d_state, conf_mamba.d_state)
		# self.B = nn.Linear(conf_mamba.dim, conf_mamba.d_state)
		self.AB = nn.Linear(conf_mamba.dim + conf_mamba.d_state, conf_mamba.d_state)
		self.C = nn.Linear(conf_mamba.d_state, conf_mamba.dim)
		self.D = nn.Linear(conf_mamba.dim, conf_mamba.dim)
		self.in_proj = nn.Linear(conf_mamba.dim, conf_mamba.dim)
		self.out_proj = nn.Linear(conf_mamba.dim, conf_mamba.dim)
		self.latent = torch.zeros((1, conf_mamba.d_state))
		self.lnorm = RMSNorm(conf_mamba.d_state)

	def forward(self, x):
		B, T, C = x.shape
		x = self.in_proj(x)
		latent = self.latent.expand(B, -1).to(x.device)
		ys = []
		for i in range(T):
			latent = self.lnorm(self.AB(torch.cat((latent, x[:, i]), dim=1)))
			# latent = self.lnorm(self.A(latent)) + self.B(x[:, i])
			ys.append(self.C(latent))
		ys = torch.stack(ys, dim=1) + self.D(x)
		return self.out_proj(ys)


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
		self.attn = MambaBlock()

	def forward(self,
		x: Tensor,
	) -> Tuple[Tensor, Union[Tensor, None]]:
		if config.health > 0 and config.mode == 'train':
			config.layers_health[self.idx]['pre_layer'] = x.norm(2).item()
		x = x + self.attn(self.ln1(x))
		x = x + self.ffn(self.ln2(x))
		if config.health > 0 and config.mode == 'train':
			config.layers_health[self.idx]['post_layer'] = x.norm(2).item()
		return x


class Transformer(nn.Module):
	def __init__(self) -> NoReturn:
		super().__init__()
		self.dim = config.dim
		self.stack = nn.ModuleDict(dict(
			tok_embs=nn.Embedding(config.vocab_size, self.dim),
			# pos_embs=nn.Embedding(config.block_size, self.dim),
			dropout=nn.Dropout(config.dropout),
			ln1=RMSNorm(self.dim),
			lm_head=nn.Linear(self.dim, config.vocab_size, bias=False),
		))
		self.blocks = nn.ModuleList([Block(idx) for idx in range(config.nlayers)])
		self.stack.tok_embs.weight = self.stack.lm_head.weight
		self.apply(self.norm_weights)
		self.count_params = self.num_params() / 1e6
		config.parameters = self.count_params
		print("Number of parameters: %.3fM" % (self.count_params,))


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
		# arange = torch.arange(T, device=seq.device)

		# x = self.stack.dropout(x + self.stack.pos_embs(arange))
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
		for _ in range(_len):
			idx_cond = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size:]
			logits, _ = self(idx_cond)
			logits = logits / temperature
			probs = F.softmax(logits, dim=-1)
			if top_k is not None:
				v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
				logits[logits < v[:, [-1]]] = -float('Inf')
			next_idx = torch.multinomial(probs, num_samples=1)
			idx = torch.cat((idx, next_idx), dim=1)
		return idx
