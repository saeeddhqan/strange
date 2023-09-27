```bash

python main.py -a train -d <dim> -nl <num-layers> -nh <num-heads> -bs <block-size> -v 'your_small_desc' --tensorboard --wandb --compile --decay-lr -lr 1e-3 -ml <min-learning-rate>

```


#### Example

```bash

python main.py -a train -lr 2e-3 -ml 1e-4 --decay-lr --tensorboard -v 'orig_expand' --data-file data/addition.txt

```