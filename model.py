
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from datasets import load_dataset
from typing import NoReturn, ClassVar, Union, Optional
import math

config = None

class Data:
	def __init__(self, config: ClassVar) -> NoReturn:
		'''
			Given a data file, we load it, convert it to a tensor, and split it to train and test data.
			Parameters
			----------
			config: ClassVar
				config instance
		'''
		model_name = 'roneneldan/TinyStories-Instruct-33M'
		dataset_name = 'roneneldan/TinyStories'

		def prepare_split(split, story_len):
			dataset = load_dataset(dataset_name, split=split)
			dataset = dataset.select(range(story_len))
			tok = '\n~\n'.join(dataset['text'])
			return tok

		text_train = prepare_split('train', 20000)
		text_test = prepare_split('validation', 2000)

		self.chars = sorted(list(set(text_train + text_test)))
		self.vocab_size = len(self.chars)
		config.vocab_size = self.vocab_size
		self.stoi = {c:i for i,c in enumerate(self.chars)}
		self.itos = {i:c for c,i in self.stoi.items()}
		self.encode = lambda s: [self.stoi[x] for x in s]
		self.decode = lambda e: ''.join([self.itos[x] for x in e])

		self.train_data = torch.tensor(self.encode(text_train), dtype=torch.long)
		self.test_data = torch.tensor(self.encode(text_test), dtype=torch.long)
		if config.device == 'cuda':
			self.train_data = self.train_data.pin_memory().to(config.device, non_blocking=True)
			self.test_data = self.test_data.pin_memory().to(config.device, non_blocking=True)

		self.block_size = config.block_size
		self.batch_size = config.batch_size


	def __len__(self):
		return self.vocab_size


	def get_batch(self, 
		idx: int, split: str = 'train',
		batch_size: int = -1,
	) -> tuple[Tensor, Tensor]:
		'''
			Given a source for data(train|test), it returns a random chunk from the source.
			Parameters
			----------
			idx: int
				we don't use it here
			split: str
				source of data
			batch_size: int
				batch size
			Returns
			-------
			x: Tensor
				input sequence
			y: Tensor
				target sequence
		'''
		data = self.train_data if split == 'train' else self.test_data
		batch_size = self.batch_size if batch_size == -1 else batch_size
		ix = torch.randint(len(data) - (self.block_size + 1), (batch_size,))
		x = torch.stack([data[i:i + self.block_size] for i in ix])
		y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
		return x, y


# From llama
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


# From nanoGPT
class CausalSelfAttention(nn.Module):
	def __init__(self, idx: int):
		super().__init__()
		assert config.embeds_size % config.num_heads == 0, 'embeds size is not divisible to the num_heads'
		self.dim = config.embeds_size
		config.head_size = self.dim // config.num_heads
		self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=False)
		self.c_proj = nn.Linear(self.dim, self.dim, bias=False)
		self.attn_dropout = nn.Dropout(config.dropout)
		self.resid_dropout = nn.Dropout(config.dropout)
		self.n_head = config.num_heads
		self.dropout = config.dropout
		self.flash = config.flash_attention and hasattr(torch.nn.functional, 'scaled_dot_product_attention')
		if not self.flash:
			self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
										.view(1, 1, config.block_size, config.block_size))

	def forward(self, x):
		B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
		q, k, v  = self.c_attn(x).split(self.dim, dim=2)
		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
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
			# att = self.attn_dropout(att) # NOTE: my preference. it causes values of early tokens nan
			y = att @ v # (B, nh, T, T) x (B, nho, T, hs) -> (B, nh, T, hs)
		
		y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

		# output projection
		y = self.resid_dropout(self.c_proj(y))
		return y


class NonLinear(nn.Module):
	'''Based on llama2 code'''
	def __init__(self):
		super().__init__()
		self.dim = config.embeds_size
		self.w1 = nn.Linear(self.dim, 4 * self.dim, bias=config.bias) # bias=False in llama
		self.w2 = nn.Linear(4 * self.dim, self.dim, bias=config.bias) # bias=False in llama
		self.w3 = nn.Linear(self.dim, 4 * self.dim, bias=config.bias) # bias=False in llama
		self.dropout = nn.Dropout(config.dropout)
	def forward(self, x: Tensor):
		'''
			Init forward method.
		'''
		return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
	def __init__(self, idx: int):
		super().__init__()
		assert config.embeds_size % config.num_heads == 0, 'embeds size is not divisible to the num_heads'
		self.idx = idx
		self.dim = config.embeds_size
		self.n_heads = config.num_heads
		self.n_layers = config.num_layers
		self.head_size = self.dim // self.n_heads
		self.c_attn_v = nn.Linear(self.dim, self.dim, config.bias)
		self.c_attn_qk = nn.Linear(self.dim, self.dim * 2, True)
		# self.c_attn_q = nn.Linear(self.dim, self.dim, True)
		# self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=config.bias)
		self.c_proj = nn.Linear(self.dim, self.dim, bias=config.bias)
		self.dropout = config.dropout
		self.resid_dropout = nn.Dropout(self.dropout)
		self.block_drop = nn.Dropout(self.dropout)
		# self.n_groups = int(config.block_size ** 0.5)
		self.n_groups = 2
		self.per_group = (config.block_size // self.n_groups)
		self.odd_even = self.n_layers % 2
		self.ffn = NonLinear()
		self.ln1 = RMSNorm(self.dim)
		self.ln2 = RMSNorm(self.dim)
		self.ln3 = RMSNorm(self.head_size)
		self.flash = config.flash_attention
		# self.causal_self_attention = CausalSelfAttention(self.idx)

	def forward(self, x, y=None):
		B, T, C = x.shape

		# Pre LN
		if y is not None:
			y = (self.block_drop(y[0]), self.block_drop(y[1]), self.block_drop(y[2]))
			y = (self.ln3(y[0]), self.ln3(y[1]), self.ln3(y[2]))
		head_out, y = self.causal_self_attention(self.ln1(x), y)
		# head_out = self.causal_self_attention(self.ln1(x))
		res_con = x + head_out
		hidden_state = res_con + self.ffn(self.ln2(res_con))


		if config.health > 0 and config.mode == 'train':
			config.layers_health[self.idx]['pre_layer'] = x.norm(2).item()
			config.layers_health[self.idx]['post_attention'] = head_out.norm(2).item()

		return hidden_state, y

	def do_att(self, q, k, v):
		return torch.nn.functional.scaled_dot_product_attention(q, k, v, 
				attn_mask=None,
				dropout_p=0,
				# dropout_p=config.dropout if self.training else 0,
				is_causal=True,
			)

	def do_block_merge(self, xblock, x):
		other_blocks = torch.cat((xblock, x[:,:,1:,:]), dim=3)
		first_block = torch.cat((x[:,:,:1], xblock[:,:,:1,-1:]), dim=3)
		x = torch.cat((first_block, other_blocks), dim=2)
		return x

	def causal_self_attention(self, x, y):
		B, T, C = x.size()
		v = self.c_attn_v(x)
		q, k = self.c_attn_qk(v).split(self.dim, dim=2)
		# q = self.c_attn_q(v) # Note
		# q, k, v  = self.c_attn(x).split(self.dim, dim=2)
		its_time = self.odd_even ^ ((self.idx + 1) % 2)
		n_groups = self.n_groups

		if self.idx == 0 and T % self.per_group != 0 and T > self.per_group:
			remain = self.per_group - (T % self.per_group) 
			comp = remain * self.dim
			T = T + remain
			pad = torch.zeros(B, remain, embeds_size).to(x.device)
			q = torch.cat((q, pad), dim=1)
			k = torch.cat((k, pad), dim=1)
			v = torch.cat((v, pad), dim=1)
			del pad, comp, remain

		n_groups = min(T // self.per_group, self.n_groups)
		q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
		k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
		v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
		if n_groups > 1:
			q = q.view(B, self.n_heads, n_groups, -1, self.head_size) # (B, nh, ng, gs, hs)
			k = k.view(B, self.n_heads, n_groups, -1, self.head_size) # (B, nh, ng, gs, hs)
			v = v.view(B, self.n_heads, n_groups, -1, self.head_size) # (B, nh, ng, gs, hs)
			if its_time and q.size(2):
				qblock = q.mean(dim=3).unsqueeze(3)
				kblock = k.mean(dim=3).unsqueeze(3)
				vblock = v.mean(dim=3).unsqueeze(3)
				q = torch.cat((q, qblock), dim=3)
				k = torch.cat((k, kblock), dim=3)
				v = torch.cat((v, vblock), dim=3)
			elif y is not None:
				qblock, kblock, vblock = y
				q = self.do_block_merge(qblock, q)
				k = self.do_block_merge(qblock, k)
				v = self.do_block_merge(vblock, v)

			T += 1

		# x = v + self.do_att(q, k, v)
		x = self.do_att(q, k, v)
		if x.dim() > 4 and its_time: # Try run block attention in every layer.
			# remove last block from q, k, v
			q, k = q[:,:,:-1,-1], k[:,:,:-1,-1]
			v = self.do_att(
				q,
				k,
				x[:,:,:-1,-1:].view(B, self.n_heads, -1, self.head_size),
			).unsqueeze(3)
			y = (q.unsqueeze(3), k.unsqueeze(3), v)
			x = x[:,:,:,:-1] # crop footprints(blocks)
			T -= 1
		else:
			if not its_time and x.dim() > 4:
				x = torch.cat((x[:,:,:1,:-1], x[:,:,1:,1:]), dim=2)
				T -= 1
			y = None
		x = x.contiguous().view(B, self.n_heads, -1, x.size(-1))
		x = x.transpose(1, 2).contiguous().view(B, T, C)
		x = self.resid_dropout(self.c_proj(x))
		return x, y


class Transformer(nn.Module):
	def __init__(self) -> NoReturn:
		super().__init__()
		self.dim = config.embeds_size
		self.pad = 8
		self.eps_dim = self.pad * 4
		self.stack = nn.ModuleDict(dict(
			tok_embs=nn.Embedding(config.vocab_size, self.dim - self.eps_dim),
			pos_embs=nn.Embedding(config.block_size, self.dim - self.eps_dim),
			eps_tok_embs=nn.Embedding(config.vocab_size + 1, 4),
			eps_pos_embs=nn.Embedding(config.block_size, 4),
			dropout=nn.Dropout(config.dropout),
			ln1=RMSNorm(self.dim),
			lm_head=nn.Linear(self.dim, config.vocab_size, bias=False),
		))
		self.blocks = nn.ModuleList([Block(idx) for idx in range(config.num_layers)])
		# self.stack.tok_embs.weight = self.stack.lm_head.weight
		self.apply(self.norm_weights)
		self.count_params = self.num_params() / 1e6
		config.parameters = self.count_params

		for name, p in self.named_parameters():
			if name.endswith('c_proj.weight'):
				torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

		print("Number of parameters: %.2fM" % (self.count_params,))


	def num_params(self) -> int:
		'''
			Calculates number of parameters.
			It does not consider token and position embeddings.
			Parameters
			----------

			Returns
			-------
			n_params: int
				number of parameters 
		'''
		n_params = sum(p.numel() for p in self.parameters())
		n_params -= self.stack.pos_embs.weight.numel()
		n_params -= self.stack.tok_embs.weight.numel()
		return n_params


	def norm_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
		elif isinstance(module, nn.LayerNorm):
			torch.nn.init.zeros_(module.bias)
			torch.nn.init.ones_(module.weight)


	def forward(self, 
		seq: Tensor,
		targets: Union[Tensor, None] = None,
	) -> tuple[Tensor, Tensor]:
		'''
			Given a sequence, and targets, it trains the 
			arch and returns logits and loss if target != None
			Parameters
			----------
			seq: Tensor
				input sequence, shape (batch, block_size)
			targets:
				target sequence, shape (batch, block_size)
			Returns
			-------
			logits: Tensor
				the network output, shape (batch, block_size, vocab_size)
			loss: Tensor
				training loss if targets is provided
		'''
		B, T = seq.shape
		tok_emb = self.stack.tok_embs(seq) # (batch, block_size, embed_dim) (B,T,C)
		arange = torch.arange(T, device=seq.device)
		pos_emb = self.stack.pos_embs(arange)

		xseq = F.pad(seq + 1, (self.pad, 0)).unfold(1, self.pad, 1)[:,:-1,:]
		bseq = F.pad(arange, (self.pad, 0)).unfold(0, self.pad, 1)[1:,:]
		eps_embs = self.stack.eps_tok_embs(xseq)
		eps_pos_embs = self.stack.eps_pos_embs(bseq)
		eps_embs = eps_embs + eps_pos_embs
		# eps_comb = torch.cat([
		# 	eps_embs.view(B, T, -1),
		# 	eps_pos_embs.view(1, T, -1).expand(B, T, -1)],
		# 	dim=-1,
		# )
		x = torch.cat([tok_emb + pos_emb, eps_embs.view(B, T, -1)], dim=-1)
		# x = tok_emb + pos_emb
		x = self.stack.dropout(x)
		y = None
		for block in self.blocks:
			x, y = block(x,y)

		if targets is None: # for autocemplete method
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
	) -> Tensor:
		'''
			Given a sequence, it autoregressively predicts the next token
			Greedy method, no top-k and nucleus
			Parameters
			----------
			idx: Tensor
				input sequence, shape (1, n)
			_len:
				length of sequence you want to create
			Returns
			-------
			idx: Tensor
				the output sequence, shape (1, n+_len)
		'''
		config.mode = 'inference'
		bsize = config.block_size
		for _ in range(_len):
			idx_cond = idx[:, -bsize:] # crop it
			with config.autocast:
				logits, _ = self(idx_cond)
			# logits = logits[:, -1, :] # only care about the last logit
			probs = F.softmax(logits, dim=-1) # view is for arch twelve
			# It selects samples from probs. The higher the prob, the more the chance of being selected
			next_idx = torch.multinomial(probs, num_samples=1) # (B, 1) one prediction for each batch
			idx = torch.cat((idx, next_idx), dim=1)
		return idx

