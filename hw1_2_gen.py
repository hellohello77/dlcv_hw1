import torch
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

model_list = ['DV3_R0.ckpt', 'DV3_M15.ckpt', 'DV3_M45.ckpt']

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
        img = Image.open(os.path.join('../Downloads/hw1_data/p2_data/validation', img_path))
        img = T.Compose([
            T.RandomAutocontrast(p = 1),
            T.RandomAdjustSharpness(2, p = 1)
        ])(img)
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

data_list = ['0013_sat.jpg', '0062_sat.jpg', '0104_sat.jpg']
p2_data_test = hw1_1_dataset(data_list, img_transform_1)
BATCH_SIZE = 3
test_loader = DataLoader(p2_data_test, batch_size=BATCH_SIZE, shuffle=False)
pretrained_model = DLV3()
for indx, model_dict in enumerate(model_list):
    pretrained_model.load_state_dict(torch.load(model_dict, map_location=torch.device('cpu')))
    pretrained_model.eval()

    for (test_imgs, data_path) in test_loader:
        outputs = pretrained_model(test_imgs)
        preds = torch.max(outputs, dim=1)[1]
        mask = mask_transfer(preds)
        # print(mask.shape)
        # print(type(data_path))
        for i in range(len(data_path)):
            im = Image.fromarray(np.uint8(mask[i]))
            name = os.path.splitext(data_path[i])[0]
            im.save(f"{indx}_{name}.png")