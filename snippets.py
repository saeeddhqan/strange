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
# addition accuracy(in generator function):
		# try:
		# 	dsplit = [(x.split('+')[0], x.split('+')[1].split('=')[0], int(x.split('=')[1])) for x in decoded.split('\n') if len(x) == 9]
		# 	corr = 0
		# 	accum = len(dsplit)
		# 	for i in range(len(dsplit)):
		# 		n1 = int(dsplit[i][0])
		# 		n2 = int(dsplit[i][1])
		# 		ans = dsplit[i][2]
		# 		cor = n1 + n2
		# 		if cor == ans:
		# 			corr += 1
		# 	if config.tensorboard:
		# 		self.tensorboard_writer.add_scalar('accuracy', corr/accum, epoch, new_style=True)
		# 	print('all=', accum, ',correct=', corr, ',accuracy=', corr/accum)
		# except:
		# 	pass
# SoTA:
		# if self.pos_method == 'dynamic':
		# 	q = q + self.lnq(self.create_dype_v4(v) * self.pos_coef)
		# 	k = k + self.lnk(self.create_dype_v4(v) * self.pos_coef)
# v321:
		# def create_dype_v1(self, x: Tensor) -> Tensor:
		# 	snip = x[:,:,:,:self.dim_snip].flatten(2)
		# 	snip = F.pad(snip, (self.hsize - self.dim_snip, 0), value=0)
		# 	pos_emb = snip.unfold(2, self.hsize, self.dim_snip)
		# 	return pos_emb
		# def create_dype_v2(self, x: Tensor) -> Tensor:
		# 	pos_emb = self.pos_dropout(
		# 		F.pad(
		# 			x[:,:,:,:self.cmax].flatten(2),
		# 			(self.pad_size, 0),
		# 			value=1.0,
		# 		)[:,:,config.dypes[:x.size(2)]]
		# 	)
		# 	return pos_emb

		# def create_dype_v3(self, x: Tensor) -> Tensor:
		# 	pos_emb = self.pos_dropout(
		# 		F.pad(
		# 			x[:,:,:self.cmax].flatten(1),
		# 			(self.pad_size, 0),
		# 			value=1.0,
		# 		)[:,config.dypes[:x.size(1)]]
		# 	)
		# 	return pos_emb