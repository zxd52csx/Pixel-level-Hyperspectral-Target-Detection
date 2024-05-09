import matplotlib.pyplot as plt
import scipy as sp
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.autograd as autograd
import time

class ICLM(nn.Module):

    def __init__(self, channel,r, mmt):
        super(ICLM, self).__init__()

        self.module = nn.BatchNorm1d(channel)
        self.mmt = mmt
        self.r = r

    def forward(self, feat, batchsize=0):
        if batchsize == 0:
            mean_old = self.module.running_mean
            var_old = self.module.running_var
            feat = (feat - mean_old[None]) / ((var_old[None] + 1e-5)**0.5)
            return feat
        else:
            prior_feat = feat[:batchsize]
            num_pixel = feat.shape[0] - batchsize
            mean_old = self.module.running_mean
            var_old = self.module.running_var
            ri = int(self.r * num_pixel / prior_feat.shape[0])
            # realization #1 (faster) 
            mean_num = feat.shape[0] + prior_feat.shape[0]*ri
            feat_mean = (feat.sum(dim=0) + prior_feat.sum(dim=0) * ri) / mean_num
            feat_val = (((feat - feat_mean[None])**2).sum(dim=0) + ((prior_feat - feat_mean[None])**2).sum(dim=0) * ri) / mean_num
            mean_new = (1 - self.mmt) * mean_old + self.mmt * feat_mean
            var_new = (1 - self.mmt) * var_old + self.mmt * feat_val          
            self.module.running_mean = mean_new.detach()
            self.module.running_var = var_new.detach()

            # # realization #2
            # prior_feat = prior_feat.repeat(ri, 1)  
            # feature_aug = torch.cat([feat, prior_feat])
            # feature_aug_var = torch.cat([feat, prior_feat])
            # _ = self.module(feature_aug)
            # mean_new = (1 - self.mmt) * mean_old + self.mmt * (feature_aug.mean(dim=0))
            # var_new = (1 - self.mmt) * var_old + self.mmt * feature_aug_var.std(dim=0)**2
            
            feat = (feat - mean_new[None]) / ((var_new[None] + 1e-5)**0.5)
        return feat
    
class Linear_m(nn.Module):
    def __init__(self, input, output, bias=False):
        super(Linear_m, self).__init__()
        self.model = nn.Linear(input, output, bias=bias)
    
    def forward(self, x):
        return self.model(x)
    
class mynet_decoder(nn.Module):
    def __init__(self, band, endmember_num):
        super(mynet_decoder, self).__init__()
        self.layer = nn.Linear(endmember_num, band)
   
    def loss_deep_clustering(self, input, real_ab):
        weight = self.layer.weight
        recon = torch.matmul(real_ab, weight.T)
        recon = recon / recon.norm(2, dim=1, keepdim=True)
        mse = torch.mean(((torch.abs(recon - input + 1e-8))**2).sum(dim=1))
        ab_bk = real_ab
        orth = torch.matmul(ab_bk.transpose(1,0), ab_bk)/torch.matmul(ab_bk.norm(2,dim=0).view(-1,1),ab_bk.norm(2,dim=0).view(1,-1))
        orth2 = orth[0,1:]

        return mse + orth2.mean()

class FCbDT(nn.Module):
    def __init__(self, input, hidden_depth,increment, gt):
        super(FCbDT, self).__init__()
        self.hidden_depth = hidden_depth
        self.encoder = nn.ModuleList()
        self.increment = increment
        self.mmt = 0.9
        self.encoder += [
            nn.LayerNorm(input,elementwise_affine=True),
            nn.Linear(input, hidden_depth, bias=False),
            ICLM(hidden_depth, increment[0], self.mmt),
            nn.Sigmoid(),
        ]
        for _ in range(len(self.increment) - 1):
            if _ == len(self.increment) - 2:
                self.encoder +=[
                    nn.Linear(hidden_depth, hidden_depth, bias=False),
                    ICLM(hidden_depth, increment[0], self.mmt),
                    nn.Linear(hidden_depth, 100),
                    ]
            else:
                self.encoder +=[
                    nn.Linear(hidden_depth, hidden_depth, bias=False),
                    ICLM(hidden_depth, increment[0], self.mmt),
                    nn.Sigmoid(),
                    ]
        for layer in self.encoder:
            if layer._get_name == 'Linear':
                torch.nn.init.normal_(layer.weight, mean=0.1, std=0.)
        self.cc = 3
        self.gt = gt
        self.spatial = nn.Conv2d(1, self.cc**2, self.cc, padding=int(self.cc / 2))
        with torch.no_grad():
            self.spatial.weight = nn.Parameter(self.spatial.weight.clamp(0,0))
            for i in range(self.cc**2):
                x, y = i // self.cc, i % self.cc
                self.spatial.weight[i, :, x, y] = nn.Parameter(1 + self.spatial.weight[i, :, x, y]) 
    

    def detect(self, Inscene_HSI):
        x = Inscene_HSI
        for layer in self.encoder:
            x = layer(x)
        x = x.softmax(dim=-1)
        return x[:,0]

    def forward(self, x, Tc_mask):
        if Tc_mask.sum() > 0:
            LSSC = True
        else:
            LSSC = False

        prior, neg_data = x[0], x[1]
        num_pixel = neg_data.shape[0]
        ensemble_feat = torch.cat([x[0], x[1]])
        batchsize = prior.shape[0]

        sl_list = [0]
        feat_list = []

        for index, layer in enumerate(self.encoder):
            if layer._get_name() == 'ICLM':
                ensemble_feat = layer(ensemble_feat, batchsize)
            else:
                ensemble_feat = layer(ensemble_feat)
            if layer._get_name() == 'Linear' and LSSC:
                spatia_shape = (1, self.gt.shape[0], self.gt.shape[1], -1)
                feat_spatia = ensemble_feat[batchsize:batchsize+num_pixel].reshape(spatia_shape).permute(0,3,1,2).contiguous()
                feat_spatia = feat_spatia[-1][:,None]
                unlabel_num = feat_spatia.shape[-1] * feat_spatia.shape[-2]
                if unlabel_num == self.gt.shape[1] * self.gt.shape[0]:
                    self.spatial.eval()
                    feat_spatia = self.spatial(feat_spatia)
                    feat_spatia_candi = feat_spatia.permute(2, 3, 1, 0).reshape(-1, self.cc**2, feat_spatia.shape[0])[Tc_mask>0]
                    feat_spatia_candi = feat_spatia_candi.softmax(dim=-1)
                    feat_list.append(feat_spatia_candi)
        if LSSC:
            neighboring_scores = feat_list[-1]
            center_s = neighboring_scores[:,int((self.cc**2)/2):int((self.cc**2)/2)+1].detach()
            for f in feat_list[:-1]:
                delta_s = (center_s - neighboring_scores)[:,:,0]
                flag = (delta_s < 0).float()[:,:,None]
                neighboring_feat = f.detach()
                center_feat = f[:,int((self.cc**2)/2):int((self.cc**2)/2)+1].expand_as(neighboring_feat).contiguous()
                spatial_loss = (neighboring_feat * center_feat).sum(dim=-1).abs() / neighboring_feat.norm(2, dim=-1) / center_feat.norm(2, dim=-1)
                spatial_loss = -1 * spatial_loss.log()
                spatial_loss = (spatial_loss * flag[:,:,0]).sum() / max((flag[:,:,0]).sum(), 1)
                sl_list.append(spatial_loss)
        ensemble_feat = ensemble_feat.softmax(dim=-1)
        if LSSC:
            LSSC_loss = sum(sl_list)
        else:
            LSSC_loss = 0
        return ensemble_feat, LSSC_loss
    

    def loss(self, batch, ab_pre):
        if batch > 1:
            hbatch = int(batch/2)
            loss = -1 * ab_pre[:hbatch,0].log().mean()
            for i in range(hbatch):
                loss += -1 * ab_pre[hbatch: hbatch+i+1,i+1].log().mean()
            loss = loss / batch
        else:
            loss = -1 * ab_pre[:batch,0].log().mean()
        return loss
    

class MultiheadAttention(nn.Module):
    def __init__(self, input_size, num_heads, channels, incre, mmt, preprocess=False):
        super(MultiheadAttention, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.preprocess = preprocess
        if preprocess:
            self.net = nn.ModuleList()
            self.net += [
                nn.Linear(1, 32),
                nn.Linear(32, 32),            
            ]

        # 初始化为线性层参数
        self.query_linear = nn.Linear(input_size, channels)
        self.key_linear = nn.Linear(input_size, channels)
        self.value_linear = nn.Linear(input_size, channels)

        # 最后的线性层参数
        self.final_linear = nn.Linear(channels, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.linear = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ICLM = ICLM(channels **2, incre, mmt)

    def forward(self, x, prior_num=0):
        if self.preprocess:
            x = x[:,:,None]
            for layer in self.net:
                x = layer(x)
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
        add_norm2 = self.ICLM(add_norm2.reshape(batch_size, -1), prior_num)
        return add_norm2


class STbDT(nn.Module):
    def __init__(self, input, hidden_depth, increment, gt):
        super(STbDT, self).__init__()
        self.hidden_depth = hidden_depth
        self.encoder = nn.ModuleList()
        self.increment = [increment[0] for _ in range(2)]
        self.mmt = 0.9
        self.encoder += [
            nn.LayerNorm(input,elementwise_affine=True),
            Linear_m(input, hidden_depth, bias=False),
            ICLM(hidden_depth, increment[0], self.mmt),
            nn.Sigmoid(),
        ]
        for _ in range(len(self.increment) - 1):
            if _ != len(self.increment) - 2:
                self.encoder +=[
                    Linear_m(hidden_depth, hidden_depth, bias=False),
                    ICLM(hidden_depth, increment[0], self.mmt),
                    nn.Sigmoid(),
                    ]
            else:                
                self.encoder +=[
                    Linear_m(hidden_depth, hidden_depth, bias=False),
                    ICLM(hidden_depth, increment[0], self.mmt),
                    Linear_m(hidden_depth, 32),
                    ]

        for layer in self.encoder:
            if layer._get_name == 'Linear':
                torch.nn.init.normal_(layer.weight, mean=0.1, std=0.)
        self.encoder += [MultiheadAttention(32, 4, 32, increment[0], self.mmt, preprocess=True)]
        self.encoder += [
                        nn.Linear(32*32, 100),
                        ICLM(100, increment[0], self.mmt), 
                        Linear_m(100, 100)
                     ]
        self.net_depth = len(self.encoder)
        self.cc = 3
        self.gt = gt
        self.spatial = nn.Conv2d(1, self.cc**2, self.cc, padding=int(self.cc / 2))
        with torch.no_grad():
            self.spatial.weight = nn.Parameter(self.spatial.weight.clamp(0,0))
            for i in range(self.cc**2):
                x, y = i // self.cc, i % self.cc
                self.spatial.weight[i, :, x, y] = nn.Parameter(1 + self.spatial.weight[i, :, x, y]) 
    

    def detect(self, Inscene_HSI):
        x = Inscene_HSI
        for layer in self.encoder:
            x = layer(x)
        x = x.softmax(dim=-1)
        return x[:,0]

    def forward(self, x, Tc_mask):
        if Tc_mask.sum() > 0:
            LSSC = True
        else:
            LSSC = False
        prior, neg_data = x[0], x[1]
        num_pixel = neg_data.shape[0]
        ensemble_feat = torch.cat([x[0], x[1]])
        batchsize = prior.shape[0]

        sl_list = [0]
        feat_list = []

        for index, layer in enumerate(self.encoder):
            if layer._get_name() in ['ICLM', 'MultiheadAttention']:
                ensemble_feat = layer(ensemble_feat, batchsize)
            else:
                ensemble_feat = layer(ensemble_feat)
            if layer._get_name() == 'Linear_m' and LSSC:
                spatia_shape = (1, self.gt.shape[0], self.gt.shape[1], -1)
                feat_spatia = ensemble_feat[batchsize:batchsize+num_pixel].reshape(spatia_shape).permute(0,3,1,2).contiguous()
                feat_spatia = feat_spatia[-1][:,None]
                unlabel_num = feat_spatia.shape[-1] * feat_spatia.shape[-2]
                if unlabel_num == self.gt.shape[1] * self.gt.shape[0]:
                    self.spatial.eval()
                    feat_spatia = self.spatial(feat_spatia)
                    feat_spatia_candi = feat_spatia.permute(2, 3, 1, 0).reshape(-1, self.cc**2, feat_spatia.shape[0])[Tc_mask>0]
                    feat_spatia_candi = feat_spatia_candi.softmax(dim=-1)
                    feat_list.append(feat_spatia_candi)
        if LSSC:
            neighboring_scores = feat_list[-1]
            center_s = neighboring_scores[:,int((self.cc**2)/2):int((self.cc**2)/2)+1].detach()
            for f in feat_list[:-1]:
                delta_s = (center_s - neighboring_scores)[:,:,0]
                flag = (delta_s < 0).float()[:,:,None]
                neighboring_feat = f.detach()
                center_feat = f[:,int((self.cc**2)/2):int((self.cc**2)/2)+1].expand_as(neighboring_feat).contiguous()
                spatial_loss = (neighboring_feat * center_feat).sum(dim=-1).abs() / neighboring_feat.norm(2, dim=-1) / center_feat.norm(2, dim=-1)
                spatial_loss = -1 * spatial_loss.log()
                spatial_loss = (spatial_loss * flag[:,:,0]).sum() / max((flag[:,:,0]).sum(), 1)
                sl_list.append(spatial_loss)
        if LSSC:
            LSSC_loss = sum(sl_list)
        else:
            LSSC_loss = 0
        ensemble_feat = ensemble_feat.softmax(dim=-1)
        return ensemble_feat, LSSC_loss



    def loss(self, batch, ab_pre):
        if batch > 1:
            hbatch = int(batch/2)
            loss = -1 * ab_pre[:hbatch,0].log().mean()
            for i in range(hbatch):
                loss += -1 * ab_pre[hbatch: hbatch+i+1,i+1].log().mean()
            loss = loss / batch
        else:
            loss = -1 * ab_pre[:batch,0].log().mean()
        return loss
    




