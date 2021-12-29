#!/bin/bash
# Author vim
# Created Time: 2021-01-27 12:01
#--master_addr=10.193.7.37
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nnodes=2 --master_addr=10.193.7.37 --node_rank=1  --nproc_per_node=8   --master_port 29501 simcle_contrastive.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nnodes=1  --node_rank=0  --nproc_per_node=8   --master_port 29501 simcle_contrastive_zh.py \
                  --save_model_path 'models/simcle_roberta_zh_12_3_6kw/' \
                  --batch_size 200  \
                  --eval_steps 250 \
                  --queue_len 200000 \
                  --lr 0.00005 \
                  --model_name princeton-nlp/sup-simcse-roberta-large  \
                  --epochs 10  \
                  --task 'contrastive' \
                  --eval_name 'eval_stsb_spearman' \
                  --temp 0.0 \
                  --dropout 0.1 \
#                  --zh_data_path /share/nlp-aic/wmt/train.all.zh \
#                  --en_data_path /share/nlp-aic/wmt/train.all.en \
