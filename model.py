
from einops import rearrange, repeat, einsum
from dataclasses import dataclass
import torch, math, sentencepiece, json
import torch.nn.functional as F
from torch import nn, Tensor
from typing import NoReturn, ClassVar, Union, Optional, Tuple
from pscan import pscan, pscan_faster


config = None


@dataclass
class ConfigMamba:
	dim: int = 256
	nlayers: int = 6
	vocab_size: int = 0
	d_state: int = 16
	expand: int = 2
	dt_rank: Union[int, str] = 'auto'
	d_conv: int = 4 
	d_inner: int = 0
	pad_vocab_size_multiple: int = 8
	conv_bias: bool = True
	bias: bool = False
	group: bool = True
	ngroups: int = 32

	def __post_init__(self):
		self.d_inner = int(self.expand * self.dim)
		
		if self.dt_rank == 'auto':
			self.dt_rank = math.ceil(self.dim / 16)


conf_mamba = ConfigMamba()

class Data:
	def __init__(self, config: ClassVar) -> NoReturn:
		if config.token_type == 'token':
			data = torch.tensor([])
			if True:
				sp_model = 'data/wikisplit/wikisplit-sp.model'
				parts = 62
				for part in range(parts):
					with open(f'data/wikisplit/train_p{part}.json') as fp:
						load = torch.tensor(json.load(fp))
						data = torch.cat((data, load), dim=0)
				data = data.to(torch.long)
			else:
				sp_model = 'data/shakespeare-sp.model'
				with open(f'data/shakespeare.txt') as fp:
					encoded = self.encode(fp.read())
				data = torch.tensor(encoded, dtype=torch.long)

			self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=sp_model)
			self.encode = self.tokenizer.encode
			self.decode = lambda seq: self.tokenizer.decode(seq)
			self.vocab_size = self.tokenizer.vocab_size()
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

	def forward(self, x) -> Tensor:
		output = self._norm(x.float()).type_as(x)
		return (output * self.weight)


class MambaBlock(nn.Module):
	def __init__(self, idx: int):
		super().__init__()
		self.idx = idx
		self.d_state = conf_mamba.d_state
		self.d_in = conf_mamba.d_inner
		self.dim = conf_mamba.dim
		self.in_proj = nn.Linear(self.dim, self.d_in * 2, bias=conf_mamba.bias)

		self.conv1d = nn.Conv1d(
			in_channels=self.d_in,
			out_channels=self.d_in,
			bias=conf_mamba.conv_bias,
			kernel_size=conf_mamba.d_conv,
			groups=self.d_in,
			padding=conf_mamba.d_conv - 1,
		)
		self.x_proj = nn.Linear(self.d_in, conf_mamba.dt_rank + conf_mamba.d_state * 2, bias=False)
		self.dt_proj = nn.Linear(conf_mamba.dt_rank, self.d_in, bias=True)
		self.out_proj = nn.Linear(self.d_in, self.dim, bias=config.bias)
		self.ngroups = conf_mamba.ngroups

		A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_in)
		self.A_log = nn.Parameter(torch.log(A))
		self.D = nn.Parameter(torch.ones(self.d_in))
		self.resid_drop = nn.Dropout(0.0)
		if conf_mamba.group:
			self.hot_loop = self.group_block
			self.ng = conf_mamba.ngroups
			self.scale = 1.0 / math.sqrt(self.d_state * self.d_in)
			self.c_attn = nn.Linear(self.d_state, 3 * self.d_state, bias=config.bias)
			# self.c_proj = nn.Linear(self.d_state, self.d_state, bias=config.bias)
			# self.ln1 = RMSNorm(self.d_state)
		else:
			self.hot_loop = self.vanilla_block


	def forward(self, x: Tensor, latent: Tensor) -> Tensor:
		b, l, d = x.shape

		x_res = self.in_proj(x)

		x, res = x_res.split(split_size=[self.d_in, self.d_in], dim=-1)
		x = self.conv1d(x.mT)[:, :, :l]
		x = F.silu(x.mT)

		d_in, n = self.A_log.shape
		A = -torch.exp(self.A_log.float())
		D = self.D.float()
		x_dbl = self.x_proj(x)
		delta, B, C = x_dbl.split(split_size=[conf_mamba.dt_rank, n, n], dim=-1)
		delta = F.softplus(self.dt_proj(delta))

		y, latent = self.hot_loop(x, delta, A, B, C, D, latent)
		y = y * F.silu(res)

		return self.out_proj(y), latent


	def vanilla_block_seq(self, u: Tensor, delta: Tensor,
		A: Tensor, B: Tensor, C: Tensor, D: Tensor,
		latent: Tensor | None = None,
	) -> Tensor:
		b, l, d_in = u.shape
		n = A.shape[1]
		
		deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
		deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

		x = torch.zeros((b, d_in, n), device=deltaA.device)
		ys = []
		for i in range(l):
			x = deltaA[:, i] * x + deltaB_u[:, i]
			y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
			ys.append(y)
		y = torch.stack(ys, dim=1)
		
		y = y + u * D

		return y, None


	def vanilla_block(self, u: Tensor, delta: Tensor,
		A: Tensor, B: Tensor, C: Tensor, D: Tensor,
		latent: Tensor | None = None,
	) -> Tensor:
		b, l, d_in = u.shape
		n = A.shape[1]
		
		deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
		deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

		hs = pscan(deltaA, deltaB_u)
		y = (hs @ C.unsqueeze(-1)).squeeze() # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
		y = y + u * D

		return y, None


	def group_block(self, u: Tensor, delta: Tensor,
		A: Tensor, B: Tensor, C: Tensor, D: Tensor,
		latent: Tensor | None = None,
	) -> Tensor:
		b, l, d_in = u.shape
		n = A.shape[1]
		pg = l // self.ng # tokens per group

		A = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n')) # delta A
		B = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n') # delta B

		latent = torch.zeros((b, self.ng, d_in, n), device=u.device) if self.idx == 0 else latent
		A = A.view(b, self.ng, pg, d_in, n).contiguous()
		B = B.view(b, self.ng, pg, d_in, n).contiguous()
		C = C.view(b, self.ng, pg, -1).contiguous()

		B = torch.cat(((B[:,:,0] + latent).unsqueeze(2), B[:,:,1:]), dim=2)
		hs = pscan_faster(A, B)
		y = (hs @ C.unsqueeze(-1)).squeeze().view(b, l, d_in) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

		y = y + u * D

		return y, self.latent_communication(latent, pg)


	def group_block_seq(self, u: Tensor, delta: Tensor,
		A: Tensor, B: Tensor, C: Tensor, D: Tensor,
		latent: Tensor | None = None,
	) -> Tensor:
		b, l, d_in = u.shape
		n = A.shape[1]
		pg = l // self.ng # tokens per group

		A = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n')) # delta A
		B = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n') # delta B

		latent = torch.zeros((b, self.ng, d_in, n), device=u.device) if self.idx == 0 else latent
		A = A.view(b, self.ng, pg, d_in, n).contiguous()
		B = B.view(b, self.ng, pg, d_in, n).contiguous()
		C = C.view(b, self.ng, pg, -1).contiguous()
		ys = []
		for i in range(pg):
			latent = A[:, :, i] * latent + B[:, :, i]
			y = einsum(latent, C[:, :, i], 'b g d_in n, b g n -> b g d_in')
			ys.append(y)
		ys = torch.stack(ys, dim=2).view(b, l, d_in).contiguous()
		ys = ys + u * D
		return ys, self.latent_communication(latent, pg)


	def latent_communication(self, latent: Tensor, pg: int):
		b = latent.size(0)
		xq, xk, xv = self.c_attn(latent[:,:-1]).split(self.d_state, dim=3)
		xq = xq.contiguous().view(b, self.ng - 1, -1)
		xk = xk.contiguous().view(b, self.ng - 1, -1)
		xv = xv.contiguous().view(b, self.ng - 1, -1)
		# simple, but group quad
		latent = F.scaled_dot_product_attention(xq, xk, xv, 
			attn_mask=None,
			# dropout_p=config.dropout if (self.training) else 0,
			is_causal=True)
		# Sequential attention
		# latent = torch.tensor([]).to(config.device)
		# for i in range(self.ng - 1):
		# 	scores = (xq[:, i:i + 1] * xk[:, :i + 1]).sum(dim=-1).unsqueeze(-1).mT * self.scale
		# 	scores = F.softmax(scores, dim=-1)
		# 	latent = torch.cat((latent, scores @ xv[:,:i + 1]), dim=1)
		latent = latent.view(b, self.ng - 1, self.d_in, self.d_state) # c_proj?

		return torch.cat((torch.zeros((b, 1, self.d_in, self.d_state), device=xq.device), latent), dim=1)


class Block(nn.Module):
	def __init__(self,
		idx: int,
	):
		super().__init__()
		self.idx = idx
		self.dim = config.dim
		self.ln1 = RMSNorm(self.dim)
		self.communicate = MambaBlock(self.idx)

	def forward(self,
		x: Tensor,
		latent: Tensor,
	) -> Tuple[Tensor, Union[Tensor, None]]:
		u, latent = self.communicate(self.ln1(x), latent)
		x = u + x
		return x, latent


class Transformer(nn.Module):
	def __init__(self) -> NoReturn:
		super().__init__()
		self.dim = config.dim
		self.stack = nn.ModuleDict(dict(
			tok_embs=nn.Embedding(config.vocab_size, self.dim),
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

		x = self.stack.dropout(self.stack.tok_embs(seq))

		latent = None
		for i, block in enumerate(self.blocks):
			x, latent = block(x, latent)

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
