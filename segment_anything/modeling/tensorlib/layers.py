#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import math
from typing import Optional, List

class TDLayer():
    def __init__(
        self,
        type,
        config,
        dropout
    ):
        self.type = type
        self.task_num = config['task_num']
        self.config = config
        self.td_dropout = dropout
        
        if dropout > 0.:
            self.td_dropout = nn.Dropout(dropout)
        else:
            self.td_dropout = lambda x: x

def tucker_decomposition(num_embeddings, embedding_dim, config):
    G = nn.Parameter(torch.zeros([config['R1'], config['R2'], config['R3']], requires_grad=True))
    U1 = nn.Parameter(torch.zeros([num_embeddings, config['R1']], requires_grad=True))
    U2 = nn.Parameter(torch.zeros([embedding_dim, config['R2']], requires_grad=True))
    U3 = nn.Parameter(torch.zeros([config['task_num'], config['R3']], requires_grad=True))
    nn.init.zeros_(G)
    nn.init.normal_(U1)
    nn.init.normal_(U2)
    nn.init.normal_(U3)
    return G, U1, U2, U3

class TDEmbedding(nn.Embedding, TDLayer):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        config,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        TDLayer.__init__(self, type=config['type'], config=config, dropout=0)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.task_num = config['task_num']
        self.scaling = config['scaling']
        # Actual trainable parameters
        if self.type == 'tucker':
            self.td_G, self.td_U1, self.td_U2, self.td_U3 = tucker_decomposition(num_embeddings, embedding_dim, config)
        self.weight.requires_grad = False

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
    
    def generate_tensor(self):
        if self.type == 'tucker':
            tensor = torch.einsum('pqv,ip->iqv', self.td_G, self.td_U1)
            tensor = torch.einsum('iqv,jq->ijv', tensor, self.td_U2)
            tensor = torch.einsum('ijv,kv->ijk', tensor, self.td_U3)
            tensor = tensor.permute((2, 1, 0))
        return tensor * self.scaling
        
    def forward(self, x: torch.Tensor, task_idx: int = -1):
        result = nn.Embedding.forward(self, x)
        tenser = self.generate_tensor()
        if task_idx != -1:
            after_A = F.embedding(
                x, tenser[task_idx], self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
        else:
            b, h, w, c = x.shape
            x = x.reshape(self.task_num, -1, h, w, c)
            after_A = torch.concat([F.embedding(
                x[_], tenser[_], self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            ) for _ in range(self.task_num)], dim=0)
        result += after_A
        return result


class TDLinear(nn.Linear, TDLayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        config,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        TDLayer.__init__(self, type=config['type'], config=config, dropout=config['dropout_rate'])
        self.task_num = config['task_num']
        self.scaling = config['scaling']
        # Actual trainable parameters
        if self.type == 'tucker':
            self.td_G, self.td_U1, self.td_U2, self.td_U3 = tucker_decomposition(in_features, out_features, config)
            
        self.weight.requires_grad = False

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)   

    def generate_tensor(self):
        if self.type == 'tucker':
            tensor = torch.einsum('pqv,ip->iqv', self.td_G, self.td_U1)
            tensor = torch.einsum('iqv,jq->ijv', tensor, self.td_U2)
            tensor = torch.einsum('ijv,kv->ijk', tensor, self.td_U3)
            tensor = tensor.permute((2, 1, 0))
        return tensor * self.scaling

    def forward(self, x: torch.Tensor, task_idx: int = -1):
        tensor = self.generate_tensor()
        result = F.linear(x, self.weight, bias=self.bias)
        if task_idx != -1:
            result += F.linear(self.td_dropout(x), tensor[task_idx])
        else:
            b, h, w, c = x.shape
            x = x.reshape(self.task_num, -1, h, w, c)
            after_A = torch.concat([F.linear(self.td_dropout(x[_]), tensor[_])
                                    for _ in range(self.task_num)]
                                   , dim=0)
            result += after_A
        return result

class QKVLinear(nn.Linear, TDLayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        enable_qkv: List[bool],
        config,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        TDLayer.__init__(self, type=config['type'], config=config, dropout=config['dropout_rate'])
        assert out_features % len(enable_qkv) == 0, \
            'The length of enable_qkv must divide out_features'
        self.enable_qkv = enable_qkv
        self.task_num = config['task_num']
        self.scaling = config['scaling']
        
        # Actual trainable parameters
        if any(enable_qkv):
            if self.type == 'tucker':
                self.td_G, self.td_U1, self.td_U2, self.td_U3 = tucker_decomposition(in_features, out_features // len(enable_qkv) * sum(enable_qkv), config)
            self.weight.requires_grad = False
            # Compute the indices
            self.ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_qkv), -1)
            self.ind[enable_qkv, :] = True
            self.ind = self.ind.view(-1)
    
    def generate_tensor(self):
        if self.type == 'tucker':
            tensor = torch.einsum('pqv,ip->iqv', self.td_G, self.td_U1)
            tensor = torch.einsum('iqv,jq->ijv', tensor, self.td_U2)
            tensor = torch.einsum('ijv,kv->ijk', tensor, self.td_U3)
            tensor = tensor.permute((2, 1, 0))
        return tensor * self.scaling

    def zero_pad(self, x):
        result = x.new_zeros(*x.shape[:-2],len(self.ind), *x.shape[-1:])
        result[:, self.ind, :] = x
        return result

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)     

    def forward(self, x: torch.Tensor, task_idx: int=-1):
        tensor = self.generate_tensor()
        tensor = self.zero_pad(tensor)
        result = F.linear(x, self.weight, bias=self.bias)
        if task_idx != -1:
            result += F.linear(self.td_dropout(x), tensor[task_idx])
        else:
            b, h, w, c = x.shape
            x = x.reshape(self.task_num, -1, h, w, c)
            after_A = torch.concat([F.linear(self.td_dropout(x[_]), tensor[_])
                                    for _ in range(self.task_num)]
                                   , dim=0)
            result += after_A
        return result