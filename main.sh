#!/bin/bash
# Author vim
# Created Time: 2021-01-27 12:01
#--master_addr=10.193.7.37
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nnodes=2 --master_addr=10.193.7.37 --node_rank=1  --nproc_per_node=8   --master_port 29501 simcle_contrastive.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nnodes=1  --node_rank=0  --nproc_per_node=8   --master_port 29501 simcle_contrastive.py \
                  --save_model_path 'models/plus_bert_large_12_12_wp_epo2/' \
                  --batch_size 64  \
                  --eval_steps 250 \
                  --queue_len 200000 \
                  --lr 0.00005 \
                  --model_name bert-large-uncased \
                  --epochs 2  \
                  --task 'plus' \
                  --zh_pmp '../mix_SCSE/zh_roberta_model_12_2_wmt/model.pth' \
                  --eval_name 'eval_stsb_spearman' \
                  --temp 0.05 \
                  --dropout 0.1 \
                  --zh_data_path /share/nlp-aic/wmt/train.all.zh \
                  --en_data_path /share/nlp-aic/wmt/train.all.en \
