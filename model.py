from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import f1_score,accuracy_score
import torch.distributed as dist
from copy import deepcopy as c

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class SimCLE(nn.Module):
    def __init__(self,
                 q=None,
                 k=None,
                 temp=0.05,
                 vocab_size=10000
                 ):
        super().__init__()

        self.q_transformer=q
        self.k_transformer=k
        
#         for params in self.k_transformer.named_parameters():
#             params[1].requires_grad=False
        
        self.q_transformer.proj=nn.Linear(768,1024)
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.f1_score = None
        self.temp=temp
        self.lm_head=nn.Linear(768,vocab_size)
        self.vocab_size=vocab_size
        

    def criterion(self, query, key, temp, queue, steps, queue_len=20000):
        
        # get labels for contrastive loss
        labels = torch.arange(query.shape[0])
        
        # nomalization
        key = key / key.norm(dim=-1, keepdim=True)
        query = query / query.norm(dim=-1, keepdim=True)
        tmp=key[:query.shape[0]]
        key = torch.cat([key, queue.to(query.device)], dim=0)
        
        # caculate the similarity scores
      #  scores =  torch.einsum('ab,cb->ac', query, key)/self.temp
        scores =  torch.einsum('ab,cb->ac', query, key)*temp
        
        # get the loss of image-to-text and text-to-image
        loss = F.cross_entropy(scores, labels.to(scores.device))
      #  loss += F.cross_entropy(scores.T, labels.to(scores.device))
        
        # update the queue
        queue=torch.cat([tmp,queue],dim=0)
        queue=queue[:queue_len]
       
        if steps % 10 == 0 and dist.get_rank()==0:
            pred = scores.argmax(dim=-1)
            f1=f1_score(labels.cpu(), pred.cpu(), average='micro')
            self.f1_score = torch.tensor(f1)
            print('f1_score:',f1)
        return loss, queue.detach().cpu()
    
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.q_transformer.parameters(), self.k_transformer.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    def gather(self,tensor):
        tensor_list=[torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=tensor_list, tensor=tensor.contiguous())
        tensor_list[dist.get_rank()]=tensor
        tensor_list=torch.cat(tensor_list,dim=0)
        return tensor_list

    def gather_neg(self,tensor):
        tensor_list=[torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=tensor_list, tensor=tensor.contiguous())
        tensor_list.pop(dist.get_rank())
        tensor_list=torch.cat(tensor_list,dim=0)
        return tensor_list

    def forward_distillation(self, text, queue, steps, queue_len):
        zh_inputs,en_inputs,_,_=text
        
        q_features = self.q_transformer(**en_inputs,output_hidden_states=True, return_dict=True).pooler_output
        
        with torch.no_grad():
            k_features = self.k_transformer(**zh_inputs, output_hidden_states=True, return_dict=True).pooler_output
        
        if q_features.shape[-1]!=k_features.shape[-1]:
            q_features=self.proj(q_features)
        
        loss = F.mse_loss(q_features, k_features)
        return loss, queue

    def forward_plus(self,  text, queue, steps, queue_len):
        zh_inputs,en_inputs,mlm_inputs,mlm_labels=text
        
        q_features = self.q_transformer(**en_inputs,output_hidden_states=True, return_dict=True).pooler_output
        
        if mlm_inputs is not None:
            outputs = self.q_transformer(**mlm_inputs,output_hidden_states=True, return_dict=True).last_hidden_state
            outputs = self.lm_head(outputs).view(-1,self.vocab_size)
        
        with torch.no_grad():
            k_features = self.k_transformer(**zh_inputs, output_hidden_states=True, return_dict=True).pooler_output
        
        if q_features.shape[-1]!=k_features.shape[-1]:
            q_features=self.q_transformer.proj(q_features)
        neg=self.gather_neg(q_features)
        k_features=torch.cat([k_features,neg],dim=0)
        
        temp = self.logit_scale.exp()
        loss, img_queue = self.criterion(q_features, k_features, temp, queue, steps,queue_len=queue_len)
        if mlm_inputs is not None:
            loss_mlm = F.cross_entropy(outputs, mlm_labels.view(-1))
            loss+=loss_mlm*0.1
        
        return loss, img_queue
    
    def forward_single(self,  text, queue, steps, queue_len):
        zh_inputs,en_inputs,mlm_inputs,mlm_labels=text
        
        q_features = self.q_transformer(**en_inputs,output_hidden_states=True, return_dict=True).pooler_output
        
        if mlm_inputs is not None:
            outputs = self.q_transformer(**mlm_inputs,output_hidden_states=True, return_dict=True).last_hidden_state
            outputs = self.lm_head(outputs).view(-1,self.vocab_size)
        
        with torch.no_grad():
            k_features = self.k_transformer(**zh_inputs, output_hidden_states=True, return_dict=True).pooler_output
        
        if q_features.shape[-1]!=k_features.shape[-1]:
            q_features=self.q_transformer.proj(q_features)
        
        temp = self.logit_scale.exp()
        loss, img_queue = self.criterion(q_features, k_features, temp, queue, steps,queue_len=queue_len)
        if mlm_inputs is not None:
            loss_mlm = F.cross_entropy(outputs, mlm_labels.view(-1))
            loss+=loss_mlm*0.1
        
        return loss, img_queue
    
    def forward_2(self,  text, queue, steps, queue_len):
        if steps==1 and dist.get_rank()==0:
            print('ok')
        zh_inputs,en_inputs,_,_=text
        q_features = self.q_transformer(**en_inputs,output_hidden_states=True, return_dict=True).pooler_output
        with torch.no_grad():
            k_features = self.k_transformer(**zh_inputs,output_hidden_states=True, return_dict=True).pooler_output
        if q_features.shape[-1]!=k_features.shape[-1]:
            q_features=self.q_transformer.proj(q_features)
        q_features=self.gather(q_features)
        k_features=self.gather(k_features)
        
        temp = self.logit_scale.exp()
        loss, img_queue = self.criterion(q_features, k_features, temp, queue, steps,queue_len=0)
        return loss, img_queue

    def forward(self,  text, queue, steps, queue_len):
        zh_inputs,en_inputs,mlm_inputs,mlm_labels=text
        
        q_features = self.q_transformer(**en_inputs,output_hidden_states=True, return_dict=True).pooler_output
        k_features = self.k_transformer(**zh_inputs, output_hidden_states=True, return_dict=True).pooler_output
        
        if q_features.shape[-1]!=k_features.shape[-1]:
            q_features=self.q_transformer.proj(q_features)
        q_neg=self.gather_neg(q_features)
        k_neg=self.gather_neg(k_features)
        k_features=torch.cat([k_features,q_neg,k_neg],dim=0)
        
        temp = self.logit_scale.exp()
        loss, img_queue = self.criterion(q_features, k_features, temp, queue, steps,queue_len=0)

       # loss_2, img_queue = self.criterion(k_features, orch.cat([q_features,q_neg,k_neg],dim=0), temp, queue, steps,queue_len=0)
       # loss=loss_1+loss_2
        return loss, img_queue


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    print(transformer_width, transformer_heads, transformer_layers)
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
