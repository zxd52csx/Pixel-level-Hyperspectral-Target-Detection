import os
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as spo
import cv2
from dataset.comparison_methods import classic_detectors
import copy
from .comparison_methods import *

class HTD_dataset(Dataset):
    def __init__(self, img_dir, img_name, img_refer=[], eo=False, prior_transform=['dialation'], divide=1):
        self.prior_transform = prior_transform
        self.img_refer = img_refer
        self.refer_spectra = []
        row, col = self.parse_inscene(img_dir, img_name, divide)
        self.img = self.img / np.linalg.norm(self.img, 2, axis=1, keepdims=True)
        self.test_img = self.img.reshape(row, col, -1, order='F')
        if eo:
            classic_results = classic_detectors(copy.deepcopy(self.img), copy.deepcopy(self.prior), row, col)
            self.classic_results = classic_results
        self.background_end = self.img


    def parse_inscene(self, img_dir, img_name, proportion):
        img_path = os.path.join(img_dir, img_name + '.mat')
        data = spo.loadmat(img_path)
        self.img = data['img'].T
        self.img = self.img / np.linalg.norm(self.img, 2, axis=1)[:, np.newaxis]
        self.groundtruth = data['groundtruth']
        row, col = self.groundtruth.shape[0], self.groundtruth.shape[1]
        if 'part' in self.prior_transform:
            print('mode 1: Select proportion:1/{} of target pixels.'.format(proportion))
            self.parse_refer(img_dir, img_name, proportion=proportion)
            self.prior = np.stack(self.refer_spectra).mean(axis=0)
        elif 'MT' in self.prior_transform:
            print('mode 2: Use prior from different HSI for current HSI. Reference HSIs: {}'.format(self.img_refer))
            for img_ in self.img_refer:
                self.parse_refer(img_dir, img_, proportion=proportion)
            self.prior = np.stack(self.refer_spectra).mean(axis=0)
        else:
            print('mode 3: Averge all the pixels for prior.')
            self.prior = self.img[self.groundtruth.reshape(-1, order='F')>0].mean(axis=0)
        self.prior /= np.linalg.norm(self.prior, 2).clip(1e-10,None)
        return row, col
        
    def parse_refer(self, img_dir, img_name, proportion=1):
        img_path = os.path.join(img_dir, img_name + '.mat')
        data = spo.loadmat(img_path)
        img = data['img'].T
        groundtruth = data['groundtruth']
        img = img / np.linalg.norm(img, 2, axis=1, keepdims=True)

        num, label,_, centroid = cv2.connectedComponentsWithStats(groundtruth)
        if len(centroid) > 1:
            centroid = centroid[::proportion]
        groundtruth = groundtruth.clip(0,0)
        label = [i[np.newaxis] for i in label]
        gt = np.concatenate(label)
        for i, c in enumerate(centroid):
            gt[int(c[1]), int(c[0])] = 2 * i
            groundtruth[int(c[1]), int(c[0])] = 1

        prior_spectrum = (img)[groundtruth.reshape(-1, order='F')>0].mean(axis=0)
        self.refer_spectra.append(prior_spectrum)


    def __len__(self):
        return self.background_end.shape[0]
    
    def __getitem__(self, idx):
        spectra = self.background_end[idx]
        spectra = spectra / np.linalg.norm(spectra, 2)      
        norm = np.linalg.norm(spectra, 2)
        pseudo_target = norm * 1 * self.prior / np.linalg.norm(self.prior, 2)
        pseudo_target = pseudo_target / np.linalg.norm(pseudo_target, 2)
        return torch.Tensor(pseudo_target).cuda(), torch.Tensor(spectra).cuda()