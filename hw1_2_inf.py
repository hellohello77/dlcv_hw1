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
from tqdm import tqdm
import math
import argparse
import random
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("input_path")
parser.add_argument("output_path")
args = parser.parse_args()

def mask_transfer(outputs):
    mask = np.empty((outputs.size()[0], 512, 512, 3))
    # print(outputs.size())
    mask[outputs == 0, :] = np.array([0, 255, 255])
    mask[outputs == 1, :] = np.array([255, 255, 0])
    mask[outputs == 2, :] = np.array([255, 0, 255])
    mask[outputs == 3, :] = np.array([0, 255, 0])
    mask[outputs == 4, :] = np.array([0, 0, 255])
    mask[outputs == 5, :] = np.array([255, 255, 255])
    mask[outputs == 6, :] = np.array([0, 0, 0])
    # print(mask.shape)
    return mask

class hw1_1_dataset:
    def __init__(self, data_list, transform):
        self.transform = transform
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        img = Image.open(os.path.join(args.input_path, img_path))
        transformed_img = self.transform(img)
        return (transformed_img, img_path)

img_transform_1 = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225])
])

class DLV3(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self):
        super().__init__()
        self.dlv3 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        self.classifier = nn.Conv2d(21, 7, kernel_size = 1)
    def forward(self, x):
        out = self.dlv3(x)['out']
        out = self.classifier(out)
        return out

pretrained_model = DLV3()
pretrained_model.load_state_dict(torch.load('./DV3_M45.ckpt'))
pretrained_model = pretrained_model.to('cuda')
pretrained_model.eval()

data_list = [file for file in os.listdir(args.input_path) if file.endswith('.jpg')]
data_list.sort()
p2_data_test = hw1_1_dataset(data_list, img_transform_1)
BATCH_SIZE = 3
test_loader = DataLoader(p2_data_test, batch_size=BATCH_SIZE, shuffle=False)

progress = tqdm(total = len(data_list))

for batch_idx, (test_imgs, data_path) in enumerate(test_loader):
    test_imgs = test_imgs.to('cuda')
    outputs = pretrained_model(test_imgs)
    preds = torch.max(outputs, dim=1)[1]
    preds = preds.cpu()
    mask = mask_transfer(preds)
    for i in range(len(data_path)):
        im = Image.fromarray(np.uint8(mask[i]))
        name = os.path.splitext(data_path[i])[0]
        im.save(os.path.join(args.output_path, f"{name}.png"))
        progress.update(1)