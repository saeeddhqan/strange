# eps_emb:
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