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


def mk_dirs(path):
    if os.path.exists(os.path.dirname(path)):
        os.mkdir(path)
    else:
        mk_dirs(os.path.dirname(path))

def ROC(test_labels, resultall, name, image_name, row, col, compare):
    if compare:
        method_num = len(name)
        num_per_row = 4
        row_n = method_num // num_per_row + 1

        fig, ax = plt.subplots(row_n, num_per_row)
        for id in range(int(np.ceil(method_num / num_per_row) * num_per_row)):
            if id < len(resultall):
                result = resultall[id]
                ax[id//num_per_row, id % num_per_row].imshow(result.reshape(row, col, order='F'))
                ax[id//num_per_row, id % num_per_row].set_title(name[id].replace('_', ''))
            ax[id//num_per_row, id % num_per_row].axis('off')
        plt.show()
        plt.close()
    else:
        plt.imshow(resultall[0].reshape(row, col, order='F'))
        plt.show()
        plt.close()

    auc_list = []
    for i in range(len(resultall)):
        fpr, tpr, thresholds = metrics.roc_curve(
         test_labels, resultall[i], pos_label=1)  # caculate False alarm rate and Probability of detection
        t = thresholds.clip(None, 1)
        auc_DF = metrics.auc(fpr, tpr)     # caculate AUC (Area Under the Curve)
        auc_Ft = metrics.auc(fpr, t)
        auc_Dt = metrics.auc(tpr, t)
        auc_TD = metrics.auc(tpr, t) + metrics.auc(fpr, tpr)
        auc_BS = metrics.auc(fpr, tpr) - metrics.auc(fpr, t)
        auc_TDBS = metrics.auc(tpr, t) - metrics.auc(fpr, t)
        auc_ODP = metrics.auc(tpr, t) + metrics.auc(fpr, tpr) - metrics.auc(fpr, t)
        auc_snpr = metrics.auc(tpr, t) / metrics.auc(fpr, t)
        auc_list.append([auc_DF, auc_Ft, auc_Dt, auc_TD, auc_BS, auc_TDBS, auc_ODP, auc_snpr])
        auc_str = '{}_AUC: DF:{:.4f}, Ft:{:.4f}, Dt:{:.4f}, TD:{:.4f}, BS:{:.4f}, TDBS:{:.4f}, ODP:{:.4f}, snpr:{:.4f}'.format(
            name[i],auc_DF, auc_Ft, auc_Dt, auc_TD, auc_BS, auc_TDBS, auc_ODP, auc_snpr)
        print(auc_str)
    return auc_list