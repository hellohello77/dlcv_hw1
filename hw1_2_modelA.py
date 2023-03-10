# -*- coding: utf-8 -*-
"""2022dlcv_1_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O5-5WQqQx4VI5W9Gos2fo1oKnm_YnymD
"""

!gdown --id 1012Oi7Sp2aLSJK2QtH9QI8zFfcT7xTdz --output "./hw1_data.zip"
!unzip -q "./hw1_data.zip" -d "./"
!rm hw1_data.zip

!nvidia-smi

# from google.colab import drive
# drive.mount('/content/drive')

# !pip install cloud-tpu-client==0.10 torch==1.12.0 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-1.12-cp37-cp37m-linux_x86_64.whl

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
# print(torch.backends.mps.is_available())
# import torch_xla
# import torch_xla.core.xla_model as xm

path_to_datafile = '/content/hw1_data/p2_data'

# img=Image.open(os.path.join(path_to_datafile, 'p1_data/train_50/0_0.png'))
# plt.imshow(img)
# plt.show()

"""### Tools"""

import scipy.misc
import imageio
import os

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

    # for i in range(6):
    #   pred = torch.add(pred, -1)
    #   labels = torch.add(labels, -1)
    #   tp_fp = torch.numel(pred) - torch.count_nonzero(pred)
    #   tp_fn = torch.numel(labels) - torch.count_nonzero(labels)
    #   bit_or = torch.bitwise_or(pred, labels)
    #   tp = torch.numel(bit_or) - torch.count_nonzero(bit_or)
    #   iou = tp / (tp_fp + tp_fn - tp + 1e-7)
    #   mean_iou += iou / 6
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp + 1e-7)
        mean_iou += iou / 6
    #     print('class #%d : %1.5f'%(i, iou))
    # print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

"""# Data"""

import random
class hw1_1_dataset:
    def __init__(self, filepath, transform, train):
        self.transform = transform
        self.train = train
        self.filepath = filepath
        self.masks = torch.from_numpy(np.int64(read_masks(filepath)))
        self.data_list = [file for file in os.listdir(filepath) if file.endswith('.jpg')]
        self.data_list.sort()
        n_data = len(self.data_list)
        if(n_data != self.masks.shape[0]):
            print('data do not match')
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        img = Image.open(os.path.join(self.filepath, img_path))
        label = self.masks[idx]
        if self.train :
          random_flip = [0, 1]
          random_rotate = [0, 90, 180, 270]
          flip = random.sample(random_flip, 1)[0]
          rotate = random.sample(random_rotate, 1)[0]
          if random_flip:
            img = T.functional.hflip(img)
          img = T.functional.rotate(img, angle = rotate)
          # print(label.size())
          if random_flip:
            label = T.functional.hflip(label)
          label = T.functional.rotate(torch.unsqueeze(label, 0), angle = rotate).squeeze()
        img = T.Compose([
            T.RandomAutocontrast(p = 1),
            T.RandomAdjustSharpness(2, p = 1)
        ])(img)
        transformed_img = self.transform(img)
        return (transformed_img, label)

img_transform_1 = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225])
])

p2_data_train = hw1_1_dataset(os.path.join(path_to_datafile, 'train'), img_transform_1, train = True)
p2_data_test = hw1_1_dataset(os.path.join(path_to_datafile, 'validation'), img_transform_1, train = False)
BATCH_SIZE = 16
EPOCH = 60

train_loader = DataLoader(p2_data_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(p2_data_test, batch_size=BATCH_SIZE, shuffle=False)

"""### Model"""

'''
source: https://blog.csdn.net/gbz3300255/article/details/105582572
'''
class hw1_model_fcn32(nn.Module):
    def __init__(self):
        super().__init__()
        self.feats = models.vgg16(weights = 'DEFAULT').features
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score = nn.Conv2d(4096, 7, 1)

    def forward(self, x):
        feats = self.feats(x)
        fconn = self.fconn(feats)
        score = self.score(fconn)
        upsample = nn.Upsample(size = 512, mode = 'bilinear')(score)
        return upsample

device = torch.device("cuda")
# device = xm.xla_device()
pretrained_model = hw1_model_fcn32()
# pretrained_model = hw1_model_fcn16()
# pretrained_model = hw1_model_fcn8()
ck = 0
# pretrained_model.load_state_dict(torch.load(f'/content/drive/MyDrive/second_{ck}.ckpt'))
if(torch.cuda.is_available()):
    pretrained_model = pretrained_model.to(device)
else:
    print('WARNING!!!!!!!!!!!!!! MPS CAN\'T BE USED')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=0.00005)

"""### Train"""

from tqdm.notebook import tqdm
import math

so_far_best = 0
best_epoch = 0
for epoch in range(EPOCH):
    pretrained_model.train()
    # running_mIoU = 0
    corrects = 0
    print(epoch+ck+1)
    progress = tqdm(total = math.ceil(len(p2_data_train)/BATCH_SIZE))
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        labels = labels.to(device)
        temp_img = imgs.to(device)
        optimizer.zero_grad()
        outputs = pretrained_model(temp_img)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print(outputs.size())
        # Model prediction & Compute classification accuracy
        preds = torch.max(outputs, dim=1)[1]
        corrects += torch.sum(preds == labels).item()
        progress.update(1)
    print('Correct rate: ', corrects/(512*512*2000))
    

    # running_eval = 0
    pretrained_model.eval()
    tp = [0,0,0,0,0,0]
    assembly = [0,0,0,0,0,0]
    for batch_idx, (imgs, labels) in enumerate(test_loader):
        
        labels = labels.to(device)
        imgs = imgs.to(device)
        outputs = pretrained_model(imgs)
        preds = torch.max(outputs, dim=1)[1]
        for i in range(6):
          tp[i] += torch.sum((preds == i)*(labels == i)).item()
          assembly[i] += (torch.sum(preds == i).item() + torch.sum(labels == i).item())
    mean_iou = 0
    # 55 63 68 *75 **76 ***88
    for i in range(6):
      mean_iou += (tp[i]/(assembly[i]-tp[i]))/6
    print(mean_iou)
    if(so_far_best < mean_iou):
      so_far_best = mean_iou
      best_epoch = epoch
      # torch.save(pretrained_model.state_dict(), f'/content/drive/MyDrive/DV3_R{epoch+ck+1}.ckpt') 
    elif(epoch-best_epoch > 5):
      print('best_miou: ', so_far_best)
      break