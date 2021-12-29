import os 
from transformers import BertTokenizer
import torch
import clip
from torch import nn
import sys
from transformers.optimization import AdamW, get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import argparse
from dataset import BackgroundGenerator
import numpy as np
import torch.nn.functional as F
from transformer_v2 import Transformer
from sklearn.metrics import f1_score
from PIL import Image
import json
import math
from torch.cuda.amp import autocast,GradScaler
from test_clip import load_model
import csv
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict

from tensorboardX import SummaryWriter
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

device=torch.device(0)

def evaluate(
    model,
    tokenizer,
    eval_senteval_transfer: bool = False,
):

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        sentences = [' '.join(s) for s in batch]
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
        )
        for k in batch:
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            
            pooler_output = outputs.pooler_output
        return pooler_output.cpu()

    # Set params for SentEval (fastmode)
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                        'tenacity': 3, 'epoch_size': 2}

    se = senteval.engine.SE(params, batcher, prepare)
    tasks = ['STSBenchmark', 'SICKRelatedness']
    if eval_senteval_transfer:     
        tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
    model.eval()
    results = se.eval(tasks)
    stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
    sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

    metrics = {"eval_stsb_spearman": stsb_spearman, "eval_sickr_spearman": sickr_spearman, "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2} 
    if eval_senteval_transfer:
        avg_transfer = 0
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            avg_transfer += results[task]['devacc']
            metrics['eval_{}'.format(task)] = results[task]['devacc']
        avg_transfer /= 7
        metrics['eval_avg_transfer'] = avg_transfer

    return metrics


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

def to_int(items):
    return list(map(int,items))

def collate_fn(batch):
    res=defaultdict(list)
    
    for item in batch:
        for it in item:
            res[it].append(item[it])
    for item in res:
        res[item]=torch.cat(res[item],dim=0)
       
    return res
    
class MyDataset(Dataset):
    def __init__(self, data_path,tokenize,local_rank,datas_num):
        super(MyDataset, self).__init__()
        csv_file=open(data_path)
        reader=csv.reader(csv_file)
        self.texts=[item for item in reader]
        self.tokenize = tokenize

    def __getitem__(self, index):
        texts=self.texts[index]
        seqs=self.tokenize(texts, padding='max_length',max_length=50, truncation=True, return_tensors="pt")
        return seqs



    def __len__(self):
        return len(self.texts)


def main(local_rank):
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:'+str(local_rank))

    data_path='data/nli_for_simcse.csv'
    model_name=args.model_name
    save_model_path=args.save_model_path
    pmp=args.pmp
    batch_size=args.bs
    lr=args.lr
    
#     tokenizer = AutoTokenizer.from_pretrained("roberta-base")
#     bert_model = AutoModel.from_pretrained("roberta-base").to(device)
#     bert_model.load_state_dict(torch.load('../SimCLE/models/simcle_roberta_base_10_16_6kw/model_0_12000steps.pth',map_location=device))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    try:
        bert_model.load_state_dict(torch.load(pmp,map_location='cpu'))
    except:
        bert_model.proj=nn.Linear(768,1024)
    
    dataset = MyDataset(data_path,tokenizer,local_rank=args.local_rank,datas_num=0)
    print('datas has been processed')
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader=DataLoaderX(dataset=dataset,batch_size=batch_size,local_rank=local_rank,num_workers=10,shuffle=False,collate_fn=collate_fn,sampler=train_sampler)  

    model=load_model()
    model.bert_model=bert_model
    
  #  model.logit_scale = nn.Parameter(torch.ones([])*0.05)
    print(model.logit_scale.data)

    del model.transformer
    model=model.to(device)
    model.train()
    
    img_queue=torch.Tensor([])
    text_queue=torch.Tensor([])
    
    model = nn.parallel.DistributedDataParallel(model,broadcast_buffers=False,device_ids=[local_rank],find_unused_parameters=True)
    
    accumulate_back_steps=1
    num_epoch = args.epochs
    optim = AdamW(model.parameters(),lr=lr, eps=1e-8, betas=(0.9, 0.98))
    scheduler = get_cosine_schedule_with_warmup(
    optim, num_warmup_steps=0.1 * len(dataloader), num_training_steps=len(dataloader)*num_epoch)

    steps=0
    loss_accumulate=0
    scaler = GradScaler()
    max_norm=1.0
    max_result=0
    
    for epoch in range(num_epoch):
        for sample_zh in tqdm(dataloader):

            sample_zh=dict(sample_zh)
            optim.zero_grad()
            with autocast():
                loss,img_queue=model(sample_zh,img_queue,steps)
            if local_rank ==0:
                print(loss,img_queue.shape)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            steps+=1
                
            if dist.get_rank()==0 and steps% 250==0:
                model.module.bert_model.eval()
                results=evaluate(model.module.bert_model,tokenizer)
                print(results)
                if results['eval_stsb_spearman'] > max_result:
                    max_result = results['eval_stsb_spearman']
                    
                    if not os.path.exists(save_model_path):
                        os.mkdir(save_model_path)
                    torch.save(model.module.bert_model.state_dict(), \
                               save_model_path+'model_'+str(epoch)+'_{}steps.pth'.format(steps))
                model.train()
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--bs', type=int, default=128, help='local_rank')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='local_rank')
    parser.add_argument('--save_model_path', type=str, default='', help='local_rank')
    parser.add_argument('--pmp', type=str, default='', help='local_rank')
    parser.add_argument('--lr', type=float, default=0.00001, help='local_rank')
    parser.add_argument('--epochs', type=int, default=2, help='local_rank')
    query_len=2000
    args = parser.parse_args()
    print(args)
    main(args.local_rank)
