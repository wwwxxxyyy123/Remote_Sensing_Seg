import torch
import numpy as np

def miou(conf_matrix):
    """conf_matrix: torch.Tensor [C, C] 或 numpy array"""
    if isinstance(conf_matrix, torch.Tensor):
        conf = conf_matrix.cpu().numpy()
    else:
        conf = conf_matrix
    # 处理全0行的情况（该类别无像素），IoU 设为 NaN
    ious = []
    for i in range(conf.shape[0]):
        tp = conf[i, i]
        fn = conf[i, :].sum() - tp
        fp = conf[:, i].sum() - tp
        denom = tp + fp + fn
        if denom == 0:
            ious.append(float('nan'))
        else:
            ious.append(tp / denom)
    return np.nanmean(ious)

def mean_pixel_acc(conf_matrix):
    if isinstance(conf_matrix, torch.Tensor):
        conf = conf_matrix.cpu().numpy()
    else:
        conf = conf_matrix
    accs = []
    for i in range(conf.shape[0]):
        total = conf[i, :].sum()
        if total == 0:
            accs.append(float('nan'))
        else:
            accs.append(conf[i, i] / total)
    return np.nanmean(accs)

def mean_dice(conf_matrix):
    if isinstance(conf_matrix, torch.Tensor):
        conf = conf_matrix.cpu().numpy()
    else:
        conf = conf_matrix
    dices = []
    for i in range(conf.shape[0]):
        tp = conf[i, i]
        fn = conf[i, :].sum() - tp
        fp = conf[:, i].sum() - tp
        denom = 2 * tp + fp + fn
        if denom == 0:
            dices.append(float('nan'))
        else:
            dices.append(2 * tp / denom)
    return np.nanmean(dices)