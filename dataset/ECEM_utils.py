#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Oct 2018 @author: Rui ZHAO
The following code builds some useful classes and tools.

"""

import os.path as op
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import math
import cv2


class Data(object):

    def __init__(self, filepath, erode_num=1,snr=None):
        self.name = op.splitext(op.split(filepath)[1])[0]
        if snr: self.name = self.name+'_noise'
        self.data = scio.loadmat(filepath)  # load data
        self.img = np.array(self.data['X'], dtype=np.float64)  # image
        self.grt = np.array(self.data['groundtruth'], dtype=np.float64)  # groundtruth
        X = self.img
        groundtruth = self.grt
        imgH = groundtruth.shape[0]
        imgW = groundtruth.shape[1]
        X = X.reshape(X.shape[0], imgH, imgW, order='F')
        X = X[:, 16:121, 80:185]
        X = X.reshape(X.shape[0], -1, order='F')
        groundtruth = groundtruth[16:121, 80:185]
        self.img = X
        self.grt = groundtruth
        kernel = np.ones((2, 2), np.uint8)
        if erode_num == 1:
            groundtruth2 = cv2.erode(self.grt, kernel, iterations=1)
        elif erode_num == 2:
            groundtruth2 = cv2.erode(self.grt, kernel, iterations=1)
            groundtruth2 = cv2.erode(groundtruth2, kernel, iterations=1)
        if erode_num != 0:
            groundtruth_flatten = groundtruth2.reshape(-1, order='F')
            self.tgt = self.img.T[groundtruth_flatten > 0].mean(axis=0)[:,np.newaxis]
        else:
            self.tgt = np.array(self.data['d'], dtype=np.float64)  # target

        if snr:
            self.add_noise(snr)  # add noise

    def add_noise(self, snr):
        for i in range(self.img.shape[1]):   # add noise
            self.img[:, i] += wgn(self.img[:, i], snr)


class Detector(object):

    def __init__(self):
        self.data = []

    def load_data(self, img_data):
        self.data = img_data
        self.name = img_data.name
        self.img = img_data.img
        self.tgt = img_data.tgt
        self.grt = img_data.grt

    def show(self, results, names):
        imgshow = [self.img[1,].reshape(self.grt.shape, order='F'), self.grt]
        nameshow = ['image(first band)', 'groundtruth'] + names
        for item in results:
            imgshow.append(item.reshape(self.grt.shape, order='F'))
        k = math.ceil(len(imgshow) / 3) * 100 + 31
        for i in range(len(imgshow)):  # show image
            plt.subplot(k + i)
            plt.axis('off')
            plt.imshow(imgshow[i], cmap='gray')
            plt.title(nameshow[i])
        auc = plot_ROC(self.grt.reshape(-1, 1, order='F'), results, names)  # plot ROC curve
        return auc


def wgn(x, snr):
    # add white Gaussian noise to x with specific SNR
    snr = 10**(snr/10)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def dual_sigmoid(x):
    x = np.array(x)
    weights = 1.0 / (1.0 + np.exp(-x))
    return weights


def plot_ROC(test_labels, resultall, name):
    plt.subplots(num='ROC curve', figsize = [10,7])
    for i in range(len(resultall)):
        fpr, tpr, thresholds = metrics.roc_curve(
         test_labels, resultall[i], pos_label=1)  # caculate False alarm rate and Probability of detection
        auc = "%.5f" % metrics.auc(fpr, tpr)     # caculate AUC (Area Under the Curve)
        print('%s_AUC: %s'%(name[i],auc))
        if not i: my_plot = plt.semilogx if metrics.auc(fpr, tpr) > 0.9 else plt.plot
        my_plot(fpr, tpr, label=name[i])
        if i == len(resultall) -1:
            return auc
    plt.xlim([1e-5, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', facecolor='none', edgecolor='none')
    plt.title('ROC Curve')
    # plt.show()   # show ROC curve
    return 0 
