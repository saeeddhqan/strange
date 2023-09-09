
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
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
		if config.device == 'cuda':
			data = data.pin_memory().to(config.device, non_blocking=True)

		train_split = int(0.9 * len(data))
		self.train_data = data[:train_split]
		self.test_data = data[train_split:]
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


# class CausalSelfAttention2(nn.Module):
# 	def __init__(self, idx: int):
# 		super().__init__()
# 		assert config.embeds_size % config.num_heads == 0, 'embeds size is not divisible to the num_heads'
# 		self.idx = idx
# 		self.dim = config.embeds_size
# 		self.n_heads = config.num_heads
# 		self.n_layers = config.num_layers
# 		self.head_size = self.dim // self.n_heads
# 		self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=config.bias)
# 		self.c_proj = nn.Linear(self.dim, self.dim, bias=config.bias)
# 		self.dropout = config.dropout
# 		# self.attn_dropout = nn.Dropout(self.dropout)
# 		self.resid_dropout = nn.Dropout(self.dropout)
# 		# self.n_groups = 64
# 		self.n_groups = int(config.block_size ** 0.5)
# 		self.per_group = (config.block_size // self.n_groups)
# 		self.odd_even = self.n_layers % 2

# 		self.flash = config.flash_attention and hasattr(torch.nn.functional, 'scaled_dot_product_attention')

# 		self.register_buffer('bias', torch.tril(torch.ones(self.per_group + 1, self.per_group + 1))
# 									.view(1, 1, 1, self.per_group + 1, self.per_group + 1))
# 		self.register_buffer('bias2', torch.tril(torch.ones(self.n_groups, self.n_groups))
# 									.view(1, 1, self.n_groups, self.n_groups))
# 		self.register_buffer('bias3', torch.tril(torch.ones(self.per_group, self.per_group))
# 									.view(1, 1, self.per_group, self.per_group))


# 	def do_att(self, q, k, v, bias):
# 		if self.flash:
# 			y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
# 				attn_mask=None,
# 				dropout_p=0,
# 				is_causal=True,
# 			)
# 		else:
# 			att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
# 			att = att.masked_fill(bias == 0, float('-inf'))
# 			att = F.softmax(att, dim=-1)
# 			# att = self.attn_dropout(att) # Find a replacement for it.
# 			y = att @ v # (B, nh, T, T) x (B, nho, T, hs) -> (B, nh, T, hs)
# 		return y


# 	def forward(self, x):
# 		B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

# 		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
# 		q, k, v  = self.c_attn(x).split(self.dim, dim=2)
# 		odd_head = self.idx % 2
# 		its_time = (self.odd_even ^ odd_head)
# 		n_groups = self.n_groups
# 		if self.idx == 0 and T % self.per_group != 0:
# 			remain = self.per_group - (T % self.per_group)
# 			comp = remain * self.dim
# 			T = T + remain
# 			pad = torch.zeros(B, remain, self.dim).to(x.device)
# 			# Think about this during inference
# 			q = torch.cat((q, pad), dim=1)
# 			k = torch.cat((k, pad), dim=1)
# 			v = torch.cat((v, pad), dim=1)
# 			del pad

# 		n_groups = min(T // self.per_group, self.n_groups)
# 		q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
# 		k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
# 		v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
# 		# print('\tafter c_attn out:', v[0][0][0][:20])

# 		if n_groups > 1:
# 			n_groups = T // (self.per_group + 1) if (its_time and self.idx != 0) else n_groups
# 			q = q.view(B, self.n_heads, n_groups, -1, self.head_size) # (B, nh, ng, gs, hs)
# 			k = k.view(B, self.n_heads, n_groups, -1, self.head_size) # (B, nh, ng, gs, hs)
# 			v = v.view(B, self.n_heads, n_groups, -1, self.head_size) # (B, nh, ng, gs, hs)
# 			if not its_time and q.size(2):
# 				qblock = q.mean(dim=3).unsqueeze(3)
# 				kblock = k.mean(dim=3).unsqueeze(3)
# 				vblock = v.mean(dim=3).unsqueeze(3)
# 				q = torch.cat((q, qblock), dim=3)
# 				k = torch.cat((k, kblock), dim=3)
# 				v = torch.cat((v, vblock), dim=3)
# 				T += n_groups
# 				n_groups = min(T // self.per_group, self.n_groups)
# 			bias = self.bias[:,:,:,:q.size(3),:q.size(3)]
# 		else:
# 			bias = self.bias3[:,:,:self.per_group,:self.per_group]
# 		# print('\tafter pad,fold,mean out:', v[0][0][0][0][:20])
# 		# So far so good. let's check the following line
# 		x = v + self.do_att(q, k, v, bias)
# 		# x = self.do_att(q, k, v, bias)
# 		# So far so good. let's check the following block
# 		if x.dim() > 4 and not its_time:
# 			bsize = q.size(2)
# 			blocks = x[:,:,:,-1:].view(B, self.n_heads, -1, 1, self.head_size) + self.do_att(
# 				q[:,:,:,-1],
# 				k[:,:,:,-1],
# 				x[:,:,:,-1:].view(B, self.n_heads, -1, self.head_size),
# 				self.bias2[:,:,:bsize,:bsize],
# 			).unsqueeze(3)
# 			r = torch.cat((blocks[:,:,:-1,:], x[:,:,1:,:-1]), dim=3)
# 			x = torch.cat((x[:,:,:1], r), dim=2)
# 		else:
# 			if self.idx != 0 and x.dim() > 4:
# 				x = x[:,:,:,1:]
# 				T -= max(0, x.size(2))
# 		x = x.contiguous().view(B, self.n_heads, -1, x.size(-1))
# 		x = x.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
# 		# output projection
# 		x = self.resid_dropout(self.c_proj(x))
# 		return x


class NonLinear(nn.Module):
	'''Based on llama2 code'''
	def __init__(self):
		super().__init__()
		self.dim = config.embeds_size
		self.w1 = nn.Linear(self.dim, 4 * self.dim, bias=config.bias) # bias=False in llama
		self.w2 = nn.Linear(4 * self.dim, self.dim, bias=config.bias) # bias=False in llama
		self.w3 = nn.Linear(self.dim, 4 * self.dim, bias=config.bias) # bias=False in llama

	def forward(self, x: Tensor):
		'''
			Init forward method.
		'''
		return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
	def __init__(self, idx: int):
		super().__init__()
		self.idx = idx
		self.seq_size = config.block_size
		self.dim = config.embeds_size
		self.heads = CausalSelfAttention(idx)
		self.ffn = NonLinear()
		self.ln1 = RMSNorm(self.dim)
		self.ln2 = RMSNorm(self.dim)
		# self.ln1 = nn.LayerNorm(self.dim)
		# self.ln2 = nn.LayerNorm(self.dim)


	def forward(self, x):
		'''
			Init forward method.

			Parameters
			----------
			x: Tensor
				input tensor
			Returns
			-------
			hidden_state: Tensor
				output tensor
		'''
		# Pre LN
		# head_out = self.heads(self.ln1(x))
		# hidden_state = head_out + self.ffn(self.ln2(head_out))
		# Pre LN with original head
		head_out = x + self.heads(self.ln1(x))
		hidden_state = head_out + self.ffn(self.ln1(x))
		# Post LN
		# head_out = self.ln1(self.heads(x))
		# hidden_state = self.ln2(head_out + self.ffn(head_out))
		# B2C LN
		# head_out = self.ln1(self.heads(x))
		# hidden_state = self.ln2(head_out + self.ffn(head_out))

		if config.health > 0 and config.mode == 'train':
			config.layers_health[self.idx]['pre_layer'] = x.norm(2).item()
			config.layers_health[self.idx]['post_attention'] = head_out.norm(2).item()

		return hidden_state


class Transformer(nn.Module):
	def __init__(self) -> NoReturn:
		super().__init__()
		self.dim = config.embeds_size
		self.stack = nn.ModuleDict(dict(
			tok_embs=nn.Embedding(config.vocab_size, self.dim),
			pos_embs=nn.Embedding(config.block_size, self.dim),
			dropout=nn.Dropout(config.dropout),
			ln1=RMSNorm(self.dim),
			blocks=nn.ModuleList([Block(idx) for idx in range(config.num_layers)]),
			lm_head=nn.Linear(self.dim, config.vocab_size, bias=False),
		))
		self.stack.tok_embs.weight = self.stack.lm_head.weight
		self.apply(self.norm_weights)
		self.count_params = self.num_params() / 1e6
		config.parameters = self.count_params

		for name, p in self.named_parameters():
			if name.endswith('c_proj.weight'):
				torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))
			if name.endswith('c_attn.weight'):
				torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))
			if name.endswith('c_attn.bias'):
				torch.nn.init.zeros_(p)
			if name.endswith('c_proj.bias'):
				torch.nn.init.zeros_(p)

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
		x = self.stack
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

		x = tok_emb + pos_emb
		x = self.stack.dropout(x)

		for block in self.stack.blocks:
			x = block(x)

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

