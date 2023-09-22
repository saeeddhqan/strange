# eps_emb:
		# self.pad = 8
		# self.eps_dim = self.pad * 4
		# self.stack = nn.ModuleDict(dict(
		# 	tok_embs=nn.Embedding(config.vocab_size, self.dim - self.eps_dim),
		# 	pos_embs=nn.Embedding(config.block_size, self.dim - self.eps_dim),
		# 	eps_tok_embs=nn.Embedding(config.vocab_size + 1, 4),
		# 	eps_pos_embs=nn.Embedding(config.block_size, 4),
		# xseq = F.pad(seq + 1, (self.pad, 0)).unfold(1, self.pad, 1)[:,:-1,:]
		# bseq = F.pad(arange, (self.pad, 0)).unfold(0, self.pad, 1)[1:,:]
		# eps_embs = self.stack.eps_embs(xseq)
		# eps_pos_embs = self.stack.eps_pos_embs(bseq)
		# eps_comb = torch.cat([
		# 	eps_embs.view(B, T, -1),
		# 	eps_pos_embs.view(1, T, -1).expand(B, T, -1)],
		# 	dim=-1,
		# )
		# x = torch.cat([tok_emb + pos_emb, eps_comb], dim=-1)

# layer rotation:
		# rotate = epoch%config.num_layers if (torch.rand(1).item() < 0.15 and self.training) else 0
		# for i in range(config.num_layers):
			# idx = (rotate + i) % config.num_layers
			# x = self.stack.blocks[idx](x)

# layer skipper:
			# if (
			# 	i == config.nlayers-1
			# 	and torch.rand(1).item() < 0.1
			# 	and self.training
			# ):
			# 	continue
# layer wise lr:
			# if config.lw_lr:
			# 	learning_rates = config.layer_wise_lr(config.lr, config.nlayers, config.lw_lr_factor)
			# 	learning_rates.reverse()
			# 	params = [{'params': self.model.stack.parameters(), 'lr': config.lr}]
			# 	params_layers = [
			# 		{
			# 			'params': self.model.blocks[x].parameters(),
			# 			'lr': learning_rates[x],
			# 		}
			# 		for x in range(config.nlayers)
			# 	]
			# 	params.extend(params_layers)

			# self.optimizer = torch.optim.AdamW(
			# 	self.model.parameters() if not config.lw_lr else params,
			# 	lr=config.lr,
			# 	# amsgrad=True, # Found amsgrad better.
			# 	# betas=(config.beta1, config.beta2),
			# 	fused=use_fused,
			# )
# dimension-wise qkv:
		# self.q = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 1, self.dim)), gain=1/math.sqrt(2))
		# self.k = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 1, self.dim)), gain=1/math.sqrt(2))
		# self.v = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 1, self.dim)), gain=1/math.sqrt(2))
		# q, k, v  = x * self.q, x * self.k, x * self.v
# activation function
			# class MyAct(nn.Module):
			# 	def __init__(self):
			# 		super().__init__()
			# 		self.alt = nn.Parameter(torch.tensor(data=0.0))
			# 	def forward(self, x):
			# 		# return torch.where(x>=0.0, (x * (1 - (1 / (1 + x)))), self.alt)
			# 		return torch.where(x>=0.0, (x * ((1 / (1 + math.e ** -x)))), self.alt)
