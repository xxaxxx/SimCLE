#!/bin/bash
# Author vim
# Created Time: 2021-01-27 12:01
#--master_addr=10.193.7.37
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nnodes=3 --master_addr=10.193.7.38 --node_rank=2  --nproc_per_node=8   --master_port 29501 simcle_distillation.py
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nnodes=1  --node_rank=0  --nproc_per_node=1   --master_port 29501 simcle_distillation.py