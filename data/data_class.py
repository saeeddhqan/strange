class Data:
	def __init__(self, config: ClassVar) -> NoReturn:
		data = torch.tensor([])
		parts = 2
		for part in range(parts):
			with open(f'data/politic4k_p{part}.json') as fp:
				load = torch.tensor(json.load(fp))
				data = torch.cat((data, load), dim=0)
		data = data.to(torch.long)
		self.tokenizer = sentencepiece.SentencePieceProcessor(model_file='data/politic4k-sp.model')
		self.encode = self.tokenizer.encode
		self.decode = lambda seq: self.tokenizer.decode(seq)
		self.vocab_size = self.tokenizer.vocab_size()
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