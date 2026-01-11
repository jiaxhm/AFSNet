import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from train_util import *

'''
Loss function
author:Fengting Yang 
Mar.1st 2019

We only use "compute_semantic_pos_loss" func. in our final version, best result achieved with weight = 3e-3
'''

def compute_semantic_pos_loss(prob_in, labxy_feat,  pos_weight = 0.003,  kernel_size=16):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*50+2*H*W，其中b表示输入的数量，c表示特征图的通道数,h、w分布表示特征图的高宽
    # output : B*9*H*w
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()#.clone返回一个和源张量同shape、dtype和device的张量，与源张量不共享数据内存，但提供梯度的回溯。

    b, c, h, w = labxy_feat.shape#其中b表示输入的数量，c表示特征图的通道数,h、w分布表示特征图的高宽
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]

    # self def cross entropy  -- the official one combined softmax
    logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)#torch.log是以自然数e为底的指数函数
    loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b
    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S#对输入数据求范数，p=2指定L2范数，dim=1指定在维度1

    # empirically we find timing 0.005 tend to better performance
    loss_sum =  0.005 * (loss_sem + loss_pos)
    loss_sem_sum =  0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos

    return loss_sum, loss_sem_sum,  loss_pos_sum


def cross_entropy_loss_edge(prediction, label):
    label = label.long()
    mask = label.float()

    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    cost = nn.BCEWithLogitsLoss(weight=mask)(prediction.float(), label.float())
    #    print(cost.shape)
    return cost * 100.

# def dice(logits, labels):
#     logits = logits.view(-1)
#     labels = labels.view(-1)
#     eps = 1e-6
#     dice = ((logits * labels).sum() * 2 + eps) / (logits.sum() + labels.sum() + eps)
#     dice_loss = dice.pow(-1)
#     return dice_loss

def dice(predict, target, epsilon=1e-6):
    """
    Dice loss for binary classification.
    """
    assert predict.size() == target.size()
    pred = torch.sigmoid(predict)

    pred = pred.view(pred.size(0), pred.size(1), -1)
    targ = target.view(target.size(0), target.size(1), -1)

    intersection = (pred * targ).sum(dim=2)
    union = (pred + targ).sum(dim=2)

    dice_score = (2 * intersection + epsilon) / (union + epsilon)
    dice_loss = 1 - dice_score

    return dice_loss.mean()