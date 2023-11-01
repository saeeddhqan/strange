import torch


def create_prob_dist(w, C, decay_factor):
	prob_dist = torch.exp(-decay_factor * torch.arange(w, dtype=torch.float32))
	prob_dist = (prob_dist / prob_dist.sum()) * C
	print(prob_dist)
	ltz = torch.where(prob_dist < 1.0)
	if ltz[0].size(0) > 0:
		ltz = ltz[0][0]
		reqm = (prob_dist[:ltz] - prob_dist[:ltz].to(torch.int)).sum().to(torch.int)
		reql = int(prob_dist[ltz:].sum())
		prob_dist = prob_dist.to(torch.int)
		# Add to tail
		prob_dist[ltz:ltz+reql] += 1
		# Add to head
		prob_dist[:reqm] += 1
	else:
		prob_dist = prob_dist.to(torch.int)
	remain = C - prob_dist.sum()
	prob_dist[:remain] = prob_dist[:remain] + torch.ones(remain)
	return prob_dist[prob_dist > 0]


def create_mask_range(C, w, decay_factor):
	prob_dist = create_prob_dist(w, C, decay_factor).flip(0)
	Cmax = prob_dist.max().item()
	mask = (torch.arange(Cmax)[None, :] < prob_dist[:, None]).to(torch.int)
	mask[mask == 0] = -(Cmax + 1) # To make sure they won't be >= 0 by addition
	mask = mask * torch.arange(0, Cmax).view(1, -1)
	mask = mask + torch.arange(0, Cmax * mask.size(0), Cmax).view(1, -1).T
	mask = mask.view(1, -1)
	mask = mask[mask >= 0].unsqueeze(0)
	return mask, Cmax


def create_embs(mask_range, T, Cmax):
	dype = mask_range.expand(T, -1) + torch.arange(0, Cmax * T, Cmax).view(1, -1).T
	return dype
