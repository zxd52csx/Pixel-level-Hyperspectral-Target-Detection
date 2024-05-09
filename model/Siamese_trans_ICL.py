import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import copy
import time
import numpy as np
import random
from scipy.sparse import block_diag

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class MultiheadAttention(nn.Module):
    def __init__(self, input_size, num_heads, channels):
        super(MultiheadAttention, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads

        # 初始化为线性层参数
        self.query_linear = nn.Linear(input_size, channels)
        self.key_linear = nn.Linear(input_size, channels)
        self.value_linear = nn.Linear(input_size, channels)

        # 最后的线性层参数
        self.final_linear = nn.Linear(input_size, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.linear = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        # 输入x的维度为(batch_size, sequence_length, input_size)
        batch_size, sequence_length, input_size = x.size()

        # 注意力计算
        query = self.query_linear(x).view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)  # (batch_size, num_heads, sequence_length, input_size/num_heads)
        key = self.key_linear(x).view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)  # (batch_size, num_heads, sequence_length, input_size/num_heads)
        value = self.value_linear(x).view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)  # (batch_size, num_heads, sequence_length, input_size/num_heads)

        attention_scores = torch.matmul(query, key.transpose(2, 3)) / torch.sqrt(torch.tensor(input_size / self.num_heads))  # (batch_size, num_heads, sequence_length, sequence_length)
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, sequence_length, sequence_length)
        weighted_sum = torch.matmul(attention_weights, value)  # (batch_size, num_heads, sequence_length, input_size/num_heads)

        # 组合多头注意力
        concat_attention = weighted_sum.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)  # (batch_size, sequence_length, input_size)

        # 应用最后的线性层
        output = self.final_linear(concat_attention)

        # 残差
        add_norm1 = self.norm1(output + x)

        ori_shape = add_norm1.shape
        ff = self.linear(add_norm1.reshape(-1, 32)).reshape(ori_shape)

        add_norm2 = self.norm2(ff + add_norm1)
        return add_norm2
    


class siamese_trans(nn.Module):

    def __init__(self, input_size, sia=True, num_detectors=4):
        super(siamese_trans, self).__init__()
        self.input_size = input_size
        self.num_detectors = num_detectors
        setup_seed((torch.rand(1) * 10000).int().item())
        self.net = nn.ModuleList()
        self.trans1 = nn.ModuleList()
        self.trans2 = nn.ModuleList()
        self.extra_token = nn.Parameter(torch.ones(32))
        self.mlp = nn.ModuleList()

        self.encoder = nn.ModuleList()
        self.mmt = 0.9
        dim = 32
        self.increment = [0.5 for _ in range(2)]
        for _ in range(len(self.increment)):
            if _ != len(self.increment) - 1:
                if _ == 0:
                    input = input_size
                else:
                    input = dim
                self.encoder +=[
                    nn.Linear(input, dim, bias=False),
                    nn.BatchNorm1d(dim, affine=True, momentum=self.mmt),
                    nn.Sigmoid(),
                    ]
            else:
                num_2 = dim
                self.encoder +=[
                    nn.Linear(dim, num_2, bias=False),
                    nn.BatchNorm1d(num_2, affine=True, momentum=self.mmt),
                    ]
        self.encoder += [
            nn.Linear(num_2, dim),
            # nn.Linear(input_size, dim),
        ]
        input_size = dim


        self.net += [
            nn.Conv1d(1, 32, 3, 1, 1),
            nn.Linear(32, 32),            
        ]

        self.trans1 = MultiheadAttention(32 * 1, 4, input_size)
        self.trans2 = MultiheadAttention(32 * 1, 4, input_size)
        self.ICLM = nn.ModuleList([nn.BatchNorm1d(32 **2),
                                   nn.BatchNorm1d(100)])
        
        self.mlp += [
                        nn.Linear((input_size) * 32, 100),
                        nn.BatchNorm1d(100),
                        nn.Linear(100, 100)
                     ]

    def ICL(self, feat, batchsize, layer,):
        ndim = feat.ndim
        if ndim > 2:
            b, l ,c = feat.shape
            feat = feat.reshape(b, -1)
        if batchsize > 0:
            prior_feat = feat[:batchsize]
            num_pixel = feat.shape[0] - batchsize
            ri = int(self.increment[0] * num_pixel / prior_feat.shape[0])
            mean_old = layer.running_mean
            var_old = layer.running_var
            # 下面这个是为了更新running mean 和 running var
            # prior_feat = prior_feat.repeat(ri, 1)  
            # feature_aug = torch.cat([feat, prior_feat])
            # feature_aug_var = torch.cat([feat, prior_feat])
            # _ = layer(feature_aug)
            # mean_new = (1 - self.mmt) * mean_old + self.mmt * (feature_aug.mean(dim=0))
            # var_new = (1 - self.mmt) * var_old + self.mmt * feature_aug_var.std(dim=0)**2
            # 
            mean_num = feat.shape[0] + prior_feat.shape[0]*ri
            feat_mean = (feat.sum(dim=0) + prior_feat.sum(dim=0) * ri) / mean_num
            feat_val = (((feat - feat_mean[None])**2).sum(dim=0) + ((prior_feat - feat_mean[None])**2).sum(dim=0)) / mean_num
            mean_new = (1 - self.mmt) * mean_old + self.mmt * feat_mean
            var_new = (1 - self.mmt) * var_old + self.mmt * feat_val           
            layer.running_mean = mean_new.detach()
            layer.running_var = var_new.detach()
            # 手动实现的BN
            feat = (feat - mean_new[None]) / ((var_new[None] + 1e-5)**0.5)
        else:
            feat = layer(feat)
        if ndim>2:
            feat =feat.reshape(b, l, c)
        return feat

    def forward(self, x, train=True):
        ensemble_feat = torch.cat(x)
        batchsize = x[0].shape[0]
        num_pixel = x[1].shape[0]
        for index, layer in enumerate(self.encoder):
            if layer._get_name() == 'BatchNorm1d':
                ensemble_feat = self.ICL(ensemble_feat, batchsize, layer)
            else:
                ensemble_feat = layer(ensemble_feat)
        # 选择是否在trans中也加入隐式对比学习
        # feat = ensemble_feat[:batchsize]
        random = (ensemble_feat[batchsize:].shape[0] * torch.rand(10000)).long()
        feat = torch.cat([ensemble_feat[:batchsize], ensemble_feat[batchsize:][random]])
        batch_size = feat.shape[0]
        feat = self.net[0](feat[:, None, :]).permute(0,2,1).reshape(-1, 32)
        feat = self.net[1](feat).reshape(batch_size, -1, 32)
        feat = self.trans1(feat)
        feat = self.ICL(feat, batchsize, self.ICLM[0])

        feat = feat.reshape(batch_size, -1)
        for layer in self.mlp:
            if layer._get_name() == 'BatchNorm1d':
                feat = self.ICL(feat, batchsize, self.ICLM[1])
            else:
                feat = layer(feat) # 
        similar = feat.softmax(dim=-1)
        return similar[:batchsize,:1]

    def loss(self, pred):   
        pred = pred.squeeze(-1)    
        loss = -1 * pred.log()
        return loss.mean()
    
    def detect(self, x):
        for index, layer in enumerate(self.encoder):
            x = layer(x)
        feat = x
        batch_size = feat.shape[0]
        feat = self.net[0](feat[:, None, :]).permute(0,2,1).reshape(-1, 32)
        feat = self.net[1](feat).reshape(batch_size, -1, 32)
        feat = self.trans1(feat)
        feat = self.ICL(feat, 0, self.ICLM[0])
        feat = feat.reshape(batch_size, -1)
        for layer in self.mlp:
            if layer._get_name() == 'BatchNorm1d':
                feat = self.ICL(feat, 0, self.ICLM[1])
            else:
                feat = layer(feat) # 
        similar = feat.softmax(dim=-1)
        return similar[:,:1]
