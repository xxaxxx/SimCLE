import os 
from transformers import BertTokenizer
import torch
from torch import nn
import sys
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import argparse
from prefetch_generator import BackgroundGenerator
import numpy as np
from torch.cuda.amp import autocast,GradScaler
from transformers import AutoModel, AutoTokenizer, AutoConfig
from model import SimCLE
import json
from copy import deepcopy as c
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
            for k in range(len(self.batch)):
                for item in self.batch[k]:
                    self.batch[k][item]=self.batch[k][item].to(device=self.local_rank)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--local_rank',type=int,help='local_rank')
    parser.add_argument('--batch_size',type=int,default=400,help='')
    parser.add_argument('--zh_data_path',type=str,default='../datasets/trans_dataset/train_all.zh',help='local_rank')
    parser.add_argument('--en_data_path',type=str,default='../datasets/trans_dataset/train_all.en',help='local_rank')
    parser.add_argument('--max_len',type=int,default=50,help='')
    parser.add_argument('--save_model_path',type=str,default='simcle_bert_9_26/',help='local_rank')
    parser.add_argument('--eval_steps',type=int,default=125,help='')
    parser.add_argument('--queue_len',type=int,default=100000,help='')
    parser.add_argument('--epochs',type=int,default=3,help='')
    parser.add_argument('--lr',type=float,default=0.0002,help='')
    parser.add_argument('--temp',type=float,default=0.05,help='')
    parser.add_argument('--dropout',type=float,default=0.1,help='')
    parser.add_argument('--model_name',type=str,default='bert-base-uncased',help='local_rank')
    parser.add_argument('--zh_pmp',type=str,default='bert-base-uncased',help='local_rank')
    parser.add_argument('--task',type=str,default='contrastive',help='local_rank')
    parser.add_argument('--eval_name',type=str,default='contrastive',help='local_rank')

    return parser

    
class MyDataset(Dataset):
    def __init__(self, text_zh,text_en,tokenize_zh,tokenize_en):
        super(MyDataset, self).__init__()
        self.text_zh = open(text_zh).readlines()
        self.text_en = open(text_en).readlines()
        
        self.tokenize_zh = tokenize_zh
        self.tokenize_en = tokenize_en
        
    def mask_tokens(
        self, inputs, special_tokens_mask = None
    ):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = self.tokenize_en.get_special_tokens_mask(labels, already_has_special_tokens=True)
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenize_en.convert_tokens_to_ids(self.tokenize_en.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenize_en), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def __getitem__(self, index):
        sample_zh = self.text_zh[index].strip()
        sample_en = self.text_en[index].strip()
        zh_inputs = self.tokenize_zh(sample_zh,max_length=50,padding='max_length', truncation=True, return_tensors="pt")
        en_inputs = self.tokenize_en(sample_en,max_length=50,padding='max_length', truncation=True, return_tensors="pt")
        for zh,en in zip(zh_inputs,en_inputs):
            zh_inputs[zh]=zh_inputs[zh].squeeze(0)
            en_inputs[en]=en_inputs[en].squeeze(0)
        return zh_inputs,en_inputs
    
    
    def __len__(self):
        return len(self.text_zh)
    
    
def load_bert_model():
    
    tokenizer_zh = AutoTokenizer.from_pretrained("bert-base-chinese")
    model_zh = AutoModel.from_pretrained("bert-base-chinese")
    
    tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_en = AutoModel.from_pretrained("bert-base-uncased")
    
    return tokenizer_zh, tokenizer_en, model_zh, model_en
    

def load_roberta_model():
    tokenizer_zh = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
    try:
        model_zh = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
        try:
            model_zh.load_state_dict(torch.load(args.zh_pmp,map_location='cpu'))
        except:
            model_zh.proj=nn.Linear(768,1024)
            model_zh.load_state_dict(torch.load(args.zh_pmp,map_location='cpu'))
    except:
        model_zh = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        model_zh.embeddings.word_embedding =nn.Embedding(tokenizer_zh.vocab_size,1024,padding_idx=tokenizer_zh.pad_token_id)
        model_zh.load_state_dict(torch.load(args.zh_pmp,map_location='cpu'))
    
    tokenizer_en = AutoTokenizer.from_pretrained(args.model_name)
    
    model_en=AutoModel.from_pretrained(args.model_name)
#     config.attention_probs_dropout_prob=args.dropout
#     config.hidden_dropout_prob=args.dropout
#     model_en = AutoModel.from_config(config)
#     bert=AutoModel.from_pretrained(args.model_name)
#     model_en.load_state_dict(bert.state_dict())
    
    return tokenizer_zh, tokenizer_en, model_zh, model_en




def main(args):
    
    local_rank=args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:'+str(local_rank))
    
    max_len=args.max_len

    if dist.get_rank()==0:
        print('start load model...')
    tokenizer_zh, tokenizer_en, model_zh, model_en=load_roberta_model()
  #  tokenizer_zh, tokenizer_en, model_zh, model_en=load_bert_model()
    torch.cuda.manual_seed(7)

    if dist.get_rank()==0:
        print('start process datas...')
    dataset = MyDataset(args.zh_data_path, args.en_data_path, tokenizer_zh, tokenizer_en)
    if dist.get_rank()==0:
        print('datas has been processed')
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset=dataset,batch_size=args.batch_size,num_workers=0,shuffle=False,sampler=train_sampler) 

    model=SimCLE(q=model_en, k=model_zh,vocab_size=len(tokenizer_en.vocab),temp=args.temp)
    model = model.to(device)
    
    if args.task=='contrastive':
        model.forward=model.forward_single
    else:
        model.forward=model.forward_distillation
    
    # memory queue，用来存储中文的embedding，扩大对比学习的size
    queue=torch.Tensor([])
        
    model = nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=True)
    
    scaler = GradScaler()
    optim = AdamW(model.parameters(),lr=args.lr, eps=1e-8, betas=(0.9, 0.98))
    num_epoch = args.epochs
    scheduler = get_cosine_schedule_with_warmup(
    optim, num_warmup_steps=0.1 * len(dataloader), num_training_steps=len(dataloader)*num_epoch)

    steps=0
    writer=SummaryWriter(log_dir='simcle/')
    
    if dist.get_rank()==0:
        print('start training...')
        print('arguments:', args)
    
    max_norm=1.0
    max_result=0
    pre_epoch=0
    
    for epoch in range(num_epoch):
        for zh_inputs,en_inputs in tqdm(dataloader):
            
            optim.zero_grad()
            with autocast():
                loss,queue=model(text=(zh_inputs,en_inputs,None,None)\
                                 ,queue=queue,steps=steps,queue_len=args.queue_len)
            
            if dist.get_rank()==0:
                print(loss,steps)
                writer.add_scalar('simcle/loss', loss, steps)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            steps+=1
            if dist.get_rank()==0 and steps% args.eval_steps==0:

                results=evaluate(model.module.q_transformer,tokenizer_en)
                print(results)
                if results[args.eval_name] > max_result:
                    pre_epoch=epoch
                    max_result = results[args.eval_name]
                    if not os.path.exists(args.save_model_path):
                        os.mkdir(args.save_model_path)
                    os.system('rm '+args.save_model_path+'model_'+str(epoch)+'*')
                    torch.save(model.module.q_transformer.state_dict(), \
                               args.save_model_path+'model_'+str(epoch)+'_{}steps.pth'.format(steps))
                model.train()
    dist.destroy_process_group()


if __name__ == "__main__":
    
    parser=get_parser()
    args = parser.parse_args()
    main(args)
