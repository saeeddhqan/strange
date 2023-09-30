'''
Contains main methods for training a model.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import wandb
import argparse
import time
import random
import model
import math
from contextlib import nullcontext
from typing import Union, Optional, Iterable, Any, NoReturn, ClassVar



def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def get_lr(epoch, warmup_iters=2000, lr_decay_iters=3250, min_lr=1e-4, lr=1e-3):
	# 1) linear warmup for warmup_iters steps
	if epoch < warmup_iters:
		return lr
	# 2) if it > lr_decay_iters, return min learning rate
	if epoch > lr_decay_iters:
		return min_lr
	# 3) in between, use cosine decay down to min learning rate
	decay_ratio = (epoch - warmup_iters) / (lr_decay_iters - warmup_iters)
	assert 0 <= decay_ratio <= 1
	coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
	return min_lr + coeff * (lr - min_lr)


set_seed(1244)
block_size = 64
dim = 128
params = {
	'block_size': block_size,
	'lr': 1e-3, # Learning rate
	'min_lr': 1e-4, # Min learning rate
	'beta1': 0.9,
	'beta2': 0.99,
	'decay_lr': False,
	'eval_step': 250, # Every n step, we do an evaluation.
	'iterations': 5000, # Like epochs
	'eval_iterations': 200, # Do n step(s), and calculate loss.
	'batch_size': 64,
	'nlayers': 2,
	'nheads': 4,
	'ngroups': 8,
	'dropout': 0.1,
	'dim': dim,
	'weight_decay': 0.001,
	'stop_loss': 1.4, # When can we stop training? once the stop_loss is <= n and once epochs are done.
	'vocab_size': 0,
	'device': 'cuda' if torch.cuda.is_available() else 'cpu',
	'variation': 'stable', # When we change something, change this to distinguish different variations.
	'workdir': 'workdir',
	'data_file': 'data/shakespeare.txt',
	'load': '',
	'action': 'train',
	'mode': 'train',
	'data_load': None,
	'wandb': False,
	'tensorboard': False,
	'parameters': None,
	'details': '',
	'compile': False,
	'dtype': 'float32',
	'autocast': None,
	'flash_attention': True,
	'bias': False,
	'deepnorm': False,
	'init_weight': '_normal',
	'topk': -1,
	'health': False, # Monitor gradients in tensorboard
	'pos': 'learnable', # rope, dynamic, learnable
}


def after_conf_init():
	'''
		boring
	'''
	if config.device == 'cuda':
		config.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else config.dtype
	ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
	config.autocast = nullcontext() if config.device == 'cpu' else torch.amp.autocast(device_type=config.device, dtype=ptdtype)
	config.topk = None if config.topk <= 0 else config.topk


class Config:
	def __init__(self, data_dict: dict) -> NoReturn:
		'''
			Given a data_dict, the class treats each key/val as an object.
			Parameters
			----------
			data_dict: dict
				a dict that key is a property and value is its value
		'''
		self.__data_dict__ = data_dict

	def __getattr__(self, k: Union[int, str, bytes]) -> Any:
		'''
			Given a key, it returns its data if it exists, otherwise None.
			Parameters
			----------
			k: str
				key
			Returns
			-------
			v: Union[any type]
				the value of the k
		'''
		if k in self.__data_dict__:
			return self.__data_dict__[k]
		else:
			raise ValueError(f"'{k}' does not exist.")

	def __setattr__(self, k: Union[int, str, bytes], v: Any) -> NoReturn:
		if k == '__data_dict__':
			super().__setattr__(k, v)
		else:
			self.__data_dict__[k] = v

	def __delattr__(self, k: Union[int, str, bytes]) -> NoReturn:
		'''
			Given a key, it deletes it from data dict if it exists.
			Parameters
			----------
			k: str
				key that needs to be removed
		'''
		if k in self.__data_dict__:
			del self.__data_dict__[k]
		else:
			raise ValueError(f"'{k}' does not exist.")

	def set_args(self, args: argparse.Namespace) -> NoReturn:
		'''
			Given an object of argparse, the method adds all the KVs to the data.
			Parameters
			----------
			args: argparse.Namespace
				parsed args object
		'''
		for kv in args._get_kwargs():
			k,v = kv
			self.__setattr__(k, v)
		after_conf_init()

	def get_model_params(self, abstract: bool = False) -> dict:
		'''
			Returns a dictionary that contains model parameters.
			Parameters
			----------
			abstract: bool
				True if you want to remove metadata from dictionary.
		'''
		if abstract:
			filters = (
				'data_load', 'action', 'load', 'workdir',
				'wandb', 'tensorboard', 'details', 'data_file',
				'variation', 'device', 'mode', 'autocast',
				'healthcare', 'flash_attention', 'compile',
				'init_weight', 'health'
			)
		else:
			filters = ('data_load', 'load', 'iterations', 'autocast', 'health')
		params = {}
		for k in self.__data_dict__:
			if k not in filters:
				params[k] = self.__data_dict__[k]
		return params

	def set_model_params(self, params: dict) -> NoReturn:
		'''
			Returns a dictionary that contains model parameters.
			Parameters
			----------
			params: dict
				Key value parameters.
		'''

		filters = (
			'data_load', 'action', 'load', 'workdir', 'mode', 'health')
		for k in params:
			if k not in filters:
				self.__data_dict__[k] = params[k]


class ManageModel:
	def __init__(self, model: ClassVar = None) -> NoReturn:
		'''
			Parameters
			----------
			model: Union[ClassVar, None]
				model instance
		'''
		self.model = model
		self.optimizer = None
		self.loss = {}
		self.elapsed_time = 0


	def stop_criterion(self, test_loss: float) -> bool:
		'''
			Stop criterion.
			Parameters
			----------
			test_loss: float
				The test loss
			Returns
			-------
			bool:
				True if the stop criterion is met, False otherwise.
		'''
		return True if test_loss <= config.stop_loss else False


	def load_model(self, path: str) -> NoReturn:
		'''
			Load a model from path
			Parameters
			----------
			path: str
				Path to the model


		'''
		if not os.path.exists(path):
			print(f"Path '{path}' does not exist.")
			exit()
		checkpoint = torch.load(path)
		config.set_model_params(checkpoint['config'])
		config.data_load = model.Data(config)
		config.vocab_size = len(config.data_load)
		model.config = config
		self.model = model.Transformer()
		self.model.load_state_dict(checkpoint['model'])


	def net_health(self, epoch: int, lr: float) -> NoReturn:
		'''
			Gradients. Needs to be run after each iter.
			Parameters
			----------
			epoch: int
				current epoch
			lr: float
				current learning rate
		'''
		if config.tensorboard:
			for name, param in self.model.named_parameters():
				if param.grad is not None:
					self.tensorboard_writer.add_histogram(name + '/grad', param.grad, global_step=epoch)
			self.tensorboard_writer.flush()


	def pre_train(self) -> NoReturn:
		'''
			Prepare the language model for training.
			Init optimizer, tensorboard, wandb, dirs, model, etc.
		'''
		self.model.train()
		self.model.to(config.device)

		if self.optimizer is None:
			use_fused = config.device == 'cuda'

			self.optimizer = torch.optim.AdamW(
				self.model.parameters(),
				lr=config.lr,
				# amsgrad=True, # Found amsgrad better.
				# betas=(config.beta1, config.beta2),
				fused=use_fused,
			)

		variation = f"{config.variation}_{config.nlayers}nl_{config.nheads}nh_{config.dim}d_{config.dropout}do_{config.block_size}bs_{int(config.deepnorm)}dn_{config.lr}lr_{int(config.decay_lr)}dlr_{config.ngroups}ng_{config.pos}"
		if config.tensorboard:
			self.tensorboard_writer = SummaryWriter(
				comment='_' + variation,
				filename_suffix='',
			)
		if config.wandb:
			self.wandb_init = wandb.init(
				project='Wizard Cloak',
				name=variation,
				config=config.get_model_params(),
			)
		self.path_format = os.path.join(
			config.workdir,
			f"model_{variation}",
		)

		if config.wandb:
			self.wandb_init.watch(self.model, log='all')

		os.makedirs(config.workdir, exist_ok=True)


	def pre_test(self) -> NoReturn:
		'''
			Prepare the language model for testing.
		'''
		self.model.eval()
		self.model.to(config.device)


	def post_train(self) -> NoReturn:
		'''
			Tasks that relate to the after training happen here.

		'''
		if config.tensorboard:
			hyparams = config.get_model_params(abstract=True)
			metrics = {}
			hyparams['test_loss'] = self.loss['test'].item()
			hyparams['train_loss'] = self.loss['train'].item()
			hyparams['elapsed_time'] = round(self.elapsed_time / 60, 4)
			hyparams['parameters'] = config.parameters
			for i in hyparams:
				self.tensorboard_writer.add_text(i, str(hyparams[i]))
			self.tensorboard_writer.flush()
			self.tensorboard_writer.close()
		if config.wandb:
			wandb.log({
				'meta/params': config.parameters,
				'meta/elapsed_time': round(self.elapsed_time / 60, 4)
			})


	def post_test(self) -> NoReturn:
		pass


	@torch.no_grad()
	def calculate_loss(self, length: int) -> dict[str, int]:
		'''
			We select eval_iterations chunks from both train and test data
			and save their losses. All in all, evaluating the perf
			of the model on train and test data. Learnt from nanoGPT
			Parameters
			----------

			Returns
			-------
			loss: dict
				testing process loss
		'''

		self.model.eval()

		out = {}
		for split in ('train', 'test'):
			# A tensor to capture the losses
			losses = torch.zeros(config.eval_iterations)
			for k in range(config.eval_iterations):
				X, y = config.data_load.get_batch(0, split, block_size=length)
				with config.autocast:
					_, loss = self.model(X, y)
				losses[k] = loss.item()
			out[split] = losses.mean()

		self.model.train()

		return out


	@torch.no_grad()
	def test(self, epoch: int) -> NoReturn:
		'''
			Generate a sequence, calculate loss, and log
			Parameters
			----------
			epoch: int
				current epoch
		'''
		state = config.mode
		config.mode = 'inference'
		seq, elapsed, elapsed_per_token = self.generator(epoch=epoch)
		print(seq)
		print('-' * 10)
		print(f"[{epoch}] > Elapsed: {elapsed}")
		print(f"[{epoch}] > Elapsed per character: {elapsed_per_token}")
		self.loss = self.calculate_loss(config.block_size)
		test_loss = round(self.loss['test'].item(), 4)
		train_loss = round(self.loss['train'].item(), 4)
		print(f"[{epoch}] > train: {train_loss}, test: {test_loss}")
		print('-' * 30)
		if config.tensorboard:
			self.tensorboard_writer.add_scalar('train_loss', train_loss, epoch, new_style=True)
			self.tensorboard_writer.add_scalar('test_loss', test_loss, epoch, new_style=True)
			self.tensorboard_writer.flush()
		if config.wandb:
			wandb.log({
				'train/loss': train_loss,
				'test/loss': test_loss,
				'iter': epoch,
			})
		config.mode = state


	def train_chunk(self, epoch: int, test_cond: bool) -> float:
		'''
			A method for getting a chunk of data, run the model on it, and do training steps.
			Parameters
			----------
			epoch: int
				epoch
			test_cond: bool
				test condition
			Returns
			-------
			loss: float
				training process loss
		'''
		epoch_loss = 0

		if config.decay_lr:
			lr = get_lr(
				epoch + 1,
				lr_decay_iters=config.iterations,
				lr=config.lr,
				min_lr=config.min_lr,
			)

			for param_group in self.optimizer.param_groups:
				param_group['lr'] = lr

		lr = config.lr if not config.decay_lr else lr

		X, y = config.data_load.get_batch(epoch)
		start = time.time()
		with config.autocast:
			pred, loss = self.model(X, y)
		self.optimizer.zero_grad(set_to_none=True)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(
			self.model.parameters(),
			1.0,
		)

		self.optimizer.step()
		stop = time.time()
		self.elapsed_time += stop - start

		if config.health and test_cond:
			self.net_health(epoch, lr)


		return epoch_loss


	def train_procedure(self, epoch: int) -> bool:
		'''
			Running one iteration.
			Parameters
			----------
			epoch: int
				epoch
			Returns
			-------
			bool:
				specifies whether the training should continue or not.
		'''
		test_cond = epoch % config.eval_step == config.eval_step - 1 # NOTE: no
		self.train_chunk(epoch, test_cond)

		# If it's not the right time to test the model.
		if not test_cond:
			return True

		self.test(epoch)

		path = self.path_format + f"_{epoch}.pt"

		torch.save({
			'model': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'config': config.get_model_params(),
			'train_loss': self.loss['train'],
			'test_loss': self.loss['test'],
			'epoch': epoch,
			}, path)

		if self.stop_criterion(self.loss['test']):
			return False
		return True


	def train(self) -> NoReturn:
		'''
			Training process.
		'''

		self.pre_train()

		for epoch in range(config.iterations):
			try:
				if not self.train_procedure(epoch):
					print(f"The test loss is <= {config.stop_loss}. Stopping...")
					break
			except KeyboardInterrupt:
				print(f"Keyboard interrupt at epoch {epoch}.")
				break

		self.post_train()


	@torch.no_grad()
	def generator(self, seq_len: int = 100, epoch: int = 0) -> tuple[str, float, float]:
		'''
			Generate a sequence with seq_len length and return it
			along with time elapsed.
			Parameters
			----------
			seq_len: int
				sequence length you want to create
			Returns
			-------
			decoded: str
				generated sequence
			took: float
				elapsed time to generate the sequence
			took_per_token: float
				elapsed time to generate each token
		'''
		self.pre_test()

		X, _ = config.data_load.get_batch(0, 'test', batch_size=1)
		start = time.time()
		with config.autocast:
			generated = self.model.autocomplete(X, seq_len, top_k=config.topk)
		end = time.time()
		decoded = config.data_load.decode(generated[0].tolist())
		took = end - start
		took_per_token = took / len(decoded)

		self.post_test()

		return decoded, took, took_per_token


if __name__ == '__main__':
	config = Config(params)
	parser = argparse.ArgumentParser()
	parser.add_argument('--action', '-a', type=str, help='train, and test', required=True)
	parser.add_argument('--device', type=str, default=config.device, help=f"device type, default {config.device}")
	parser.add_argument('--workdir', type=str, default=config.workdir, help=f"directory to save models, default {config.device}")
	parser.add_argument('--load', type=str, default=config.load, help='path to a model to start with')
	parser.add_argument('--data-file', type=str, default=config.data_file, help=f"input data file, default {config.data_file}")
	parser.add_argument('--variation', '-v', type=str, default=config.variation, help=f"model variation, default {config.variation}")
	parser.add_argument('--details', type=str, help=f"model details, default {config.details}")
	parser.add_argument('--iterations', '-i', type=int, default=config.iterations, help=f"number of training iterations, default {config.iterations}")
	parser.add_argument('--lr', '-lr', type=float, default=config.lr, help=f"learning rate, default {config.lr}")
	parser.add_argument('--min-lr', '-ml', type=float, default=config.min_lr, help=f"minimum learning rate, default {config.min_lr}")
	parser.add_argument('--dropout', '-do', type=float, default=config.dropout, help=f"dropout prob, default {config.dropout}")
	parser.add_argument('--nlayers', '-nl', type=int, default=config.nlayers, help=f"number of blocks, default {config.nlayers}")
	parser.add_argument('--nheads', '-nh', type=int, default=config.nheads, help=f"number of heads, default {config.nheads}")
	parser.add_argument('--dim', '-d', type=int, default=config.dim, help=f"embedding size, default {config.dim}")
	parser.add_argument('--block-size', '-bs', type=int, default=config.block_size, help=f"length input sequence, default {config.block_size}")
	parser.add_argument('--batch-size', '-b', type=int, default=config.batch_size, help=f"batch size, default {config.batch_size}")
	parser.add_argument('--topk', type=int, default=config.topk, help=f"topk sampling, default {config.topk}")
	parser.add_argument('--stop-loss', type=float, default=config.stop_loss, help=f"training stops when test loss is <= a treshold, default {config.stop_loss}")
	parser.add_argument('--health', action='store_true', default=config.health, help=f"gradient health monitoring, default {config.health}")
	parser.add_argument('--wandb', action='store_true', default=config.wandb, help=f"use wandb for visualization, default {config.wandb}")
	parser.add_argument('--tensorboard', action='store_true', default=config.tensorboard, help=f"use tensorboard for visualization, default {config.tensorboard}")
	parser.add_argument('--compile', action='store_true', default=config.compile, help=f"compile the model for faster training, default {config.compile}")
	parser.add_argument('--decay-lr', action='store_true', default=config.decay_lr, help=f"decay learning rate, default {config.decay_lr}")
	parser.add_argument('--deepnorm', action='store_true', default=config.deepnorm, help=f"use deep layer normalizer, default {config.deepnorm}")
	args = parser.parse_args()

	config.set_args(args)
	task = ManageModel()

	match config.action:
		case 'train':
			config.mode = 'train'
			if config.load != '':
				task.load_model(config.load)
			else:
				config.data_load = model.Data(config)
				model.config = config
				model = model.Transformer()
				task.model = torch.compile(model) if config.compile else model
			task.train()
		case 'test':
			config.mode = 'inference'
			task.load_model(config.load)
			seq, elapsed, elapsed_per_token = task.generator(500)
			print(seq)
			print('-' * 12)
			print('Elapsed:', elapsed)
			print('Elapsed per character:', elapsed_per_token)
		case _:
			print('Invalid action.')
