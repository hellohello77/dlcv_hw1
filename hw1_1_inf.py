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
import csv

parser = argparse.ArgumentParser()
parser.add_argument("input_path")
parser.add_argument("output_path")
args = parser.parse_args()

class hw1_1_dataset:
    def __init__(self, filepath, transform_2):
        self.transform_2 = transform_2
        self.filepath = filepath
        file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
        file_list.sort()
        self.datapath = file_list
    
    def __len__(self):
        return len(self.datapath)
    
    def __getitem__(self, idx):
        img_path = self.datapath[idx]
        img = Image.open(os.path.join(self.filepath, img_path))
        transformed_img = self.transform_2(img)
        img.close()
        return transformed_img, img_path

img_transform_2 = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225])
])

p1_data_test = hw1_1_dataset(args.input_path, img_transform_2)
BATCH_SIZE = 10
test_loader = DataLoader(p1_data_test, batch_size=BATCH_SIZE, shuffle=False)

class hw1_model_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = models.resnet50()
        self.classifier = nn.Linear(1000, 50)
        
    def forward(self, x):
        x = self.core(x)
        x = self.classifier(x)
        return x


pretrained_model = hw1_model_1()
pretrained_model.load_state_dict(torch.load('./first87_28.ckpt', map_location=torch.device('cpu')))
pretrained_model.eval()

con = 0
with open(args.output_path, 'w', newline='') as csvfile:

    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'label'])
    for batch_idx, (imgs, img_path) in enumerate(test_loader):
        outputs = pretrained_model(imgs)
        preds = torch.max(outputs, dim=1)[1]
        for i in range(imgs.size()[0]):
            writer.writerow([img_path[i], preds[i].item()])
            print(img_path[i], preds[i].item(), con)
            con+=1