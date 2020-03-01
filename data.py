import os
import json
import torch
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class DATA(Dataset):
    def __init__(self, args, mode='train'): # mode: 'train', 'val' or 'test'

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir, '102flowers', 'flowers_data', 'jpg')

        ''' read the data list '''
        txt_path = os.path.join(self.data_dir, mode + 'file.txt')
        self.data = []
        f = open(txt_path, 'r')
        for line in f:
            image_name,label = line.strip().split(" ")
            self.data.append((image_name,int(label)))
        f.close()
        
        ''' set up image trainsform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                               transforms.RandomHorizontalFlip(0.5),
                               transforms.RandomResizedCrop(224),
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])

        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        img_path, cls = self.data[idx]
        img_path = os.path.join(self.img_dir, img_path)
        
        # https://www.oreilly.com/library/view/programming-computer-vision/9781449341916/ch01.html
        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        return self.transform(img), cls
