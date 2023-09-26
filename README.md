```bash

python main.py -a train -d <dim> -nl <num-layers> -nh <num-heads> -bs <block-size> -v 'your_small_desc' --tensorboard --wandb --compile --decay-lr -lr 1e-3 -ml <min-learning-rate>

```


#### Example

```bash

python main.py -a train -nl 4 -nh 4 -d 320 -lr 2e-3 -ml 1e-4 --decay-lr

```