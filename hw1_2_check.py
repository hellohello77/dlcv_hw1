import torch
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import tqdm
import math
import argparse
import random
import imageio

# source of read_masks and mean_iou_score: TA's code
def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):
        mask = imageio.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 

    return masks

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        # print('class #%d : %1.5f'%(i, iou))
    # print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou
 
mask_truth = read_masks('../Downloads/hw1_data/p2_data/validation')
mask_model = read_masks('./out_img')
mean_iou = mean_iou_score(mask_model, mask_truth)
print(mean_iou)