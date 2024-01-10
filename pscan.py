import math

import torch


class PScan(torch.autograd.Function):
	@staticmethod
	def pscan(A, X):
		B, D, L, _ = A.size()
		num_steps = int(math.log2(L))

		Aa = A
		Xa = X
		for k in range(num_steps):
			T = 2 * (Xa.size(2) // 2)

			Aa = Aa[:, :, :T].view(B, D, T//2, 2, -1)
			Xa = Xa[:, :, :T].view(B, D, T//2, 2, -1)
			
			Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
			Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

			Aa = Aa[:, :, :, 1]
			Xa = Xa[:, :, :, 1]

		for k in range(num_steps-1, -1, -1):
			Aa = A[:, :, 2**k-1:L:2**k]
			Xa = X[:, :, 2**k-1:L:2**k]

			T = 2 * (Xa.size(2) // 2)

			if T < Xa.size(2):
				Xa[:, :, -1].add_(Aa[:, :, -1].mul(Xa[:, :, -2]))
				Aa[:, :, -1].mul_(Aa[:, :, -2])

			Aa = Aa[:, :, :T].view(B, D, T//2, 2, -1)
			Xa = Xa[:, :, :T].view(B, D, T//2, 2, -1)

			Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
			Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])


	@staticmethod
	def forward(ctx, A_in, X_in):
		A = A_in.clone() # (B, L, D, N)
		X = X_in.clone() # (B, L, D, N)
		
		A = A.transpose(2, 1) # (B, D, L, N)
		X = X.transpose(2, 1) # (B, D, L, N)

		PScan.pscan(A, X)

		ctx.save_for_backward(A_in, X)

		return X.transpose(2, 1)


	@staticmethod
	def backward(ctx, grad_output_in):
		A_in, X = ctx.saved_tensors

		A = A_in.clone()

		A = A.transpose(2, 1) # (B, D, L, N)
		A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
		grad_output_b = grad_output_in.transpose(2, 1)

		grad_output_b = grad_output_b.flip(2)
		PScan.pscan(A, grad_output_b)
		grad_output_b = grad_output_b.flip(2)

		Q = torch.zeros_like(X)
		Q[:, :, 1:].add_(X[:, :, :-1] * grad_output_b[:, :, 1:])

		return Q.transpose(2, 1), grad_output_b.transpose(2, 1)


class PScanFaster(torch.autograd.Function):
	@staticmethod
	def pscan(A, X):
		B, G, D, L, _ = A.size()
		num_steps = int(math.log2(L))

		Aa = A
		Xa = X
		for k in range(num_steps):
			T = 2 * (Xa.size(3) // 2)

			Aa = Aa[:, :, :, :T].view(B, G, D, T//2, 2, -1)
			Xa = Xa[:, :, :, :T].view(B, G, D, T//2, 2, -1)

			Xa[:, :, :, :, 1].add_(Aa[:, :, :, :, 1].mul(Xa[:, :, :, :, 0]))
			Aa[:, :, :, :, 1].mul_(Aa[:, :, :, :, 0])

			Aa = Aa[:, :, :, :, 1]
			Xa = Xa[:, :, :, :, 1]

		for k in range(num_steps-1, -1, -1):
			Aa = A[:, :, :, 2**k-1:L:2**k]
			Xa = X[:, :, :, 2**k-1:L:2**k]

			T = 2 * (Xa.size(3) // 2)

			if T < Xa.size(3):
				Xa[:, :, :, -1].add_(Aa[:, :, :, -1].mul(Xa[:, :, :, -2]))
				Aa[:, :, :, -1].mul_(Aa[:, :, :, -2])

			Aa = Aa[:, :, :, :T].view(B, G, D, T//2, 2, -1)
			Xa = Xa[:, :, :, :T].view(B, G, D, T//2, 2, -1)

			Xa[:, :, :, 1:, 0].add_(Aa[:, :, :, 1:, 0].mul(Xa[:, :, :, :-1, 1]))
			Aa[:, :, :, 1:, 0].mul_(Aa[:, :, :, :-1, 1])


	@staticmethod
	def forward(ctx, A_in, X_in):
		A = A_in.clone().transpose(3, 2) # (B, G, L, D, N) -> (B, D, G, L, N)
		X = X_in.clone().transpose(3, 2) # (B, G, L, D, N) -> (B, D, G, L, N)
		PScanFaster.pscan(A, X)

		ctx.save_for_backward(A_in, X)
		return X.transpose(3, 2)


	@staticmethod
	def backward(ctx, grad_output_in):
		A_in, X = ctx.saved_tensors

		A = A_in.clone()

		A = A.transpose(3, 2) # (B, D, L, N)
		A = torch.cat((A[:, :, :, :1], A[:, :, :, 1:].flip(3)), dim=3)
		grad_output_b = grad_output_in.transpose(3, 2)

		grad_output_b = grad_output_b.flip(3)
		PScanFaster.pscan(A, grad_output_b)
		grad_output_b = grad_output_b.flip(3)

		Q = torch.zeros_like(X)
		Q[:, :, :, 1:].add_(X[:, :, :, :-1] * grad_output_b[:, :, :, 1:])

		return Q.transpose(3, 2), grad_output_b.transpose(3, 2)

pscan = PScan.apply
pscan_faster = PScanFaster.apply
