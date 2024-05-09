from scipy.io import savemat
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import os
import matplotlib
import copy as cp
from matplotlib import projections, rcParams
# import pysptools
# rcParams['font.family'] = 'serif'
# config = {
#     "mathtext.fontset":'stix',
# }
# rcParams.update(config)
rcParams['font.size'] = 8
rcParams['font.weight'] = 'bold'
# rcParams['font.serif'] = ['Times New Roman']
# del matplotlib.font_manager.weight_dict['roman']
import numpy as np
import copy
from matplotlib.ticker import FixedLocator, FixedFormatter


def plot_ROC(test_labels, resultall, name, image_name,show=False, dir='rp'):
    mark_list = ['o', 'v','o', 'v','o', 'v', '*', 'x', 'D', 's', 'P', 'h','.', ',', '*', 'x', 'D', 's', 'P', 'h','.', ',', ]
    mark_size = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,2, 2,1, 1, 1, 1,1, 1, 1, ]
    x_label = [r'$\tau$', '$P_F$', r'$\tau$']
    y_label = ['$P_D$', '$P_D$', '$P_F$']



    fig = plt.figure(figsize=(6 * 4, 4),dpi=300)
    # fig = plt.figure(figsize=(5, 4),dpi=300)
    ax = fig.add_subplot(1,4,1, projection='3d')
    # ax = fig.add_subplot(1,1,1, projection='3d')
    ax.view_init(elev=20, azim=-50)
    # ax1 = plt.axes()
    plt.tick_params(labelsize=10)
    min_fpr = 1
    for i in range(len(resultall)):
        result_s = resultall[i]
        # result_s = ((result_s - result_s.min())  /  (result_s.max() - result_s.min()).clip(1e-10, None)).clip(None, 1 - 1e-10)
        fpr, tpr, thresholds = metrics.roc_curve(test_labels, result_s, pos_label=1,)
        min_fpr = min(fpr[fpr>0].min(), min_fpr)
    min_fpr_ori = copy.deepcopy(min_fpr)
    min_fpr = int(np.ceil(np.log10(1 / min_fpr)))
    fpr_min = 10**(-1*min_fpr)
    y_ticks = [i + 1 for i in range(int(min_fpr))]
    y_ticks_labels = [r'$10^{-%s}$'%(int(min_fpr-x)) for x in range(int(min_fpr))]
    # y_ticks_labels[0] = ['']
    for i in range(len(resultall)):
        fpr, tpr, thresholds = metrics.roc_curve(test_labels, resultall[i], pos_label=1)
        fpr = fpr.clip(fpr_min, None)
        fpr_log = np.log10(fpr[fpr>0])
        fpr = fpr_log + min_fpr
        # fpr = fpr / fpr.max()
        
        t = thresholds.clip(None, 1)
        if name[i]=='ICLTD':
            ax.plot(t, fpr, tpr,label=name[i],marker =  mark_list[i],markersize = 1, lw=1, color='black')
        else:
            ax.plot(t, fpr, tpr,label=name[i],marker =  mark_list[i],markersize = 1, lw=1)
        ax.grid(True)
        # plt.tick_params(which='major')
    # ax.plot([1, 1], [1, 0], [1, 1])
    # ax.plot([1, 1], [1, 1], [0, 1])
    # ax.plot([0, 1], [1, 1], [1, 1])
    ax.tick_params(direction='out')
    plt.title(image_name, fontproperties='Times New Roman')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    # plt.rcParams['xtick.direction'] = 'in'
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel('$P_F$')
    ax.set_zlabel('$P_D$')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    # ax.xaxis.set_ticks_position('both')
    # ax.yaxis.set_ticks_position('both')
    # ax.zaxis.set_ticks_position('both')
    # ax.grid(axi="x")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks_labels)
    y_locator = FixedLocator(y_ticks)
    y_formatter = FixedFormatter(y_ticks_labels)
    # plt.yscale('symlog')
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax.yaxis.set_major_locator(y_locator)
    # ax.yaxis.set_major_formatter(y_formatter)
    ax.zaxis.set_major_locator(plt.LinearLocator(6))
    # plt.show()
    y = np.linspace(0, min_fpr, 10)
    z = np.linspace(0, 1, 10)
    Y, Z = np.meshgrid(y, z)
    X = Y * 0 + 0
    ax.plot_surface(X, Y, Z, alpha=0.2, color='gray')
    y = np.linspace(0, min_fpr, 10)
    z = np.linspace(0, 1, 10)
    Y, Z = np.meshgrid(y, z)
    X = Y * 0 + 1
    ax.plot_surface(X, Y, Z, alpha=0.2, color='green')

    title_list = [('ROC' + r'$_{(P_{D}, \tau)}$'), ('ROC' + r'$_{(P_{D}, p_{F})}$'),('ROC' + r'$_{(P_{F}, \tau)}$')]
    # plt.figure()
    auc_list = []
    if image_name == 'MT-ABU-U1':
        A = 1
    for tt in range(3):
        plt.subplot(1, 4, tt+2)
        for i in range(len(resultall)):
            result_s = resultall[i]
            # result_s = ((result_s - result_s.min())  /  (result_s.max() - result_s.min()).clip(1e-10, None)).clip(None, 1 - 1e-10)
            fpr, tpr, thresholds = metrics.roc_curve(
            test_labels, result_s, pos_label=1)  # caculate False alarm rate and Probability of detection
            t = thresholds.clip(None, 1)
            t_ = copy.deepcopy(t)
            # t_ = t_.clip(0,1)
            auc = "%.5f" % (metrics.auc(t, tpr) + metrics.auc(fpr, tpr) - metrics.auc(t, fpr))     # caculate AUC (Area Under the Curve)
            if tt == 0:
                auc_list.append([float(auc),metrics.auc(t_, tpr), metrics.auc(fpr, tpr), metrics.auc(t_, fpr)])
            if not i: my_plot = plt.semilogx if metrics.auc(fpr, tpr) > 0.0 else plt.plot
            roc_list = [(t, tpr), (fpr, tpr), (t, fpr)]
            # roc_list = [(t, tpr), (fpr, tpr), (fpr, t)]
            if show:
                if tt ==1:
                    a= 1
                if name[i] == 'ICLTD':
                    if tt in [1]:
                        my_plot(roc_list[tt][0],roc_list[tt][1], label=name[i],marker = mark_list[i],markersize = 3, lw=1, color='black')
                    elif tt in [2]:
                        plt.semilogy(roc_list[tt][0],roc_list[tt][1],label=name[i],marker = mark_list[i],markersize = 3, lw=1,color='black')
                    else:
                        plt.plot(roc_list[tt][0],roc_list[tt][1],label=name[i],marker = mark_list[i],markersize = 3, lw=1,color='black')
                else:
                    if tt in [1]:
                        my_plot(roc_list[tt][0],roc_list[tt][1], label=name[i],marker = mark_list[i],markersize = 3, lw=1)
                    elif tt in [2]:
                        plt.semilogy(roc_list[tt][0],roc_list[tt][1],label=name[i],marker = mark_list[i],markersize = 3, lw=1)
                    else:
                        plt.plot(roc_list[tt][0],roc_list[tt][1],label=name[i],marker = mark_list[i],markersize = 3, lw=1)
                # plt.xlim([-1e-10, 1.0])
                # plt.ylim([-1e-10, 1.0])
                plt.xlabel(x_label[tt])
                plt.ylabel(y_label[tt])
                plt.tick_params(labelsize=12)
                # plt.title(image_name, fontproperties='Times New Roman')
                if os.path.exists(image_name + '.png'):
                    os.remove(image_name + '.png')
                plt.gcf().subplots_adjust(bottom=0.15)
                plt.gcf().subplots_adjust(left=0.15)
                # plt.title(title_list[tt])
        plt.grid()
    # plt.legend(loc=0, bbox_to_anchor=(1.05,1),borderaxespad = 0)
    plt.legend(loc=1, prop = {'size':9})
    # plt.savefig(dir + '/' + image_name + '.png', pad_inches=0.0, bbox_inches='tight')
    # # plt.savefig('rp/' + image_name + '.png',bbox_inches='tight', dpi=300, pad_inches=0.0)
    # plt.close()

    # plt.show()
    plt.savefig(dir + '/3D_' + image_name + '.png', bbox_inches='tight', dpi=300, pad_inches=0.28)
    plt.close()

    return auc_list


def box_plot(result_list, image_name, name_list,name_dict):
    groundtruth = result_list[0]
    groundtruth_flatten = groundtruth.reshape(-1, order='F')
    plt.figure()
    plt.tick_params(labelsize=10)
    fig, axes = plt.subplots(figsize=(6, 5),dpi=300)
    box_result = []
    for x in range(len(name_list)-1):
        if result_list[x+1].ndim >1:
            result_list[x+1] = result_list[x+1][:,0]
        result_list[x+1] = (result_list[x+1] - min(result_list[x+1].min(),0))  /  (result_list[x+1].max() - min(result_list[x+1].min(),0)).clip(1e-6, None)
        box_result.append(result_list[x+1][groundtruth_flatten>0])
        box_result.append(result_list[x+1][groundtruth_flatten == 0])
        if name_list[x+1]=='hcem':
            a=1
    # result_list[11] = result_list[10]
    box = axes.boxplot(box_result, showmeans=True, showfliers=True,patch_artist=True,medianprops={'color':'black'},labels = name_list[1:]+name_list[1:])
    # plt.ylim(0,0.1)
    fliers_num_list  = []
    for me in range(len(box_result) // 2):
        back_fliers = box['fliers'][me * 2 + 1]._y  # 背景像素中的异常值
        up_t = box['whiskers'][4 * me].get_ydata()[0]  # 目标的下四分位
        back_fliers_num = back_fliers[back_fliers > up_t].shape[0]
        fliers_num_list.append(back_fliers_num)
    print(fliers_num_list)
    axes.yaxis.grid(True) #在y轴上添加网格线
    axes.set_xticks([y + 1.5 for y in range(2*(len(name_list)-1)) if y%2 ==0])
    axes.set_ylabel('Normalized Detection Scores', fontweight='bold')
    axes.set_title(image_name + ' Dataset', fontweight='bold')
    for iif, patch in enumerate(box['boxes']):
        if iif %2 == 0:
            patch.set_facecolor('orangered')
        else:
            patch.set_facecolor('skyblue')
        # if iif == 1 and image_name in ['Cuprite']:
        #     print(box['boxes'][0]._facecolor)
        #     axes.legend((box['boxes'][0], box['boxes'][1]),('Target', 'Background'), loc=0)
    axes.legend((box['boxes'][0], box['boxes'][1]),('Target', 'Background'), loc=0,fontsize=12)
    plt.xticks(rotation=30)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig(os.path.join('output/plot', image_name + '_box.png'))
    plt.close()

import os
import os.path as osp
import cProfile, pstats

# import pysptools.util as util
# import pysptools.eea as eea
# import pysptools.abundance_maps as amp
import torch

def eea_ATGP(data, p):
    print('Testing ATGP')
    atgp = eea.ATGP()
    #U = atgp.extract(data, 8, normalize=True)
    U = atgp.extract(data, 8, normalize=True)
    back_e = []
    for e in U:
        spectral_angle = (e * p).sum() / np.linalg.norm(e, 2) / np.linalg.norm(p,2)
        if spectral_angle < 0.99:
            back_e.append(torch.Tensor(e[np.newaxis]))
        print(spectral_angle)   
    print(str(atgp))
    print('  End members indexes:', atgp.get_idx())
    return back_e