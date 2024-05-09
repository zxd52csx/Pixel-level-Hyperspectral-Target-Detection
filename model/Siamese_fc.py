import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.sparse import block_diag

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ensemble_linear(nn.Module):
    def __init__(self, input_size, output_size, num_detectors, input_flag=False):
        super(ensemble_linear, self).__init__()
        mask = tuple([np.ones((output_size, input_size)) for _ in range(num_detectors)])
        mask_array = block_diag(mask).toarray()
        self.input_size = input_size
        self.num = num_detectors
        self.output_size = output_size
        self.input = input_flag
        self.linear_mask = nn.Linear(input_size * num_detectors, output_size * num_detectors, bias=False)
        self.mask = torch.Tensor(mask_array).cuda()


    def forward(self, x):
        x = torch.matmul(self.linear_mask.weight * self.mask, x.transpose(1, 0))
        x = x.transpose(1, 0)
        return x

class siamese_fc_cos_assemble(nn.Module):

    def __init__(self, input_size, sia=True, num_detectors=4):
        super(siamese_fc_cos_assemble, self).__init__()
        self.input_size = input_size
        self.num_detectors = num_detectors
        setup_seed((torch.rand(1) * 10000).int().item())
        self.net1 = nn.ModuleList()
        num_1 = input_size
        num_2 = 32

        self.net1 += [
            nn.BatchNorm1d(input_size * num_detectors, affine=True),
            ensemble_linear(input_size, num_1, num_detectors),
            nn.BatchNorm1d(num_1 * num_detectors, affine=True),
            nn.Sigmoid(),
            ensemble_linear(num_1, num_2, num_detectors),
            nn.BatchNorm1d(num_2 * num_detectors, affine=True),
            nn.Sigmoid(),
        ]

        for l in self.net1.parameters():
            if l.ndim > 1:
                torch.nn.init.normal_(l,mean=0,std=0.0001)

    def forward(self, x, train=True):
        target = x[1]
        candidate = x[0]
        candidate = candidate.reshape(-1, candidate.shape[-1])
        target_num = target.shape[0]
        if not train:
            target = torch.cat([target for _ in range(self.num_detectors)], dim=-1)
            candidate = torch.cat([candidate for _ in range(self.num_detectors)], dim=-1)
        feature = torch.cat([target, candidate])
        for layer in self.net1:
            feature = layer(feature)
        target = feature[:target_num].reshape(target_num, self.num_detectors, -1)
        candidate = feature[target_num:].reshape(target_num, self.num_detectors, -1)
        similarity = torch.sum(target * candidate, dim=-1) / (
                1e-8 + torch.norm(target, 2, dim=-1) * torch.norm(candidate, 2, dim=-1))
        similarity = similarity.abs().clamp(None, 1)
        return similarity, feature

    def loss(self, label, pred):
        assert label.ndim == pred.ndim
        loss = -(label * pred.log() + (1 - label) * (1 - pred).clamp(1e-5,None).log())
        return loss.mean()
