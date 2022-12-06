import cv2
import matplotlib.pyplot as plt
from os.path import isfile
import torch.nn.init as init
import torch
import torch.nn as nn
import numpy as np
import pandas as pd 
import os
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam, SGD, RMSprop
import time
from torch.autograd import Variable
import torch.functional as F
from tqdm import tqdm
from sklearn import metrics
import urllib
import pickle
import torch.nn.functional as F
from torchvision import models
import random
import sys

os.environ["CUDA_VISIBLE_DEVICES"]= "0"

package_path = './EfficientNet-PyTorch/'
sys.path.append(package_path)

from efficientnet_pytorch import EfficientNet

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

num_classes = 1
seed_everything(1234)
lr          = 1e-3
IMG_SIZE    = 380
coef = [0.5, 1.5, 2.5, 3.5]

train      = '/media/asilla/data123/quangpn/APTOS2019/train/'
test       = '/media/asilla/data123/quangpn/APTOS2019/test/'

train_csv  = pd.read_csv('data/train.csv')
submission_csv = pd.read_csv('data/sample_submission.csv')

train_df, val_df = train_test_split(train_csv, test_size=0.1, random_state=2018, stratify=train_csv.diagnosis)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

def expand_path(p):
    p = str(p)
    if isfile(train + p + ".png"):
        return train + (p + ".png")
    # if isfile(train_2015 + p + '.png'):
    #     return train_2015 + (p + ".png")
    if isfile(test + p + ".png"):
        return test + (p + ".png")
    return p


def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img


class MyDataset(Dataset):
    
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        label = self.df.diagnosis.values[idx]
        label = np.expand_dims(label, -1)
        
        p = self.df.id_code.values[idx]
        p_path = expand_path(p)
        image = cv2.imread(p_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 30) ,-4 ,128)
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-120, 120)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

trainset     = MyDataset(train_df, transform =train_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
valset       = MyDataset(val_df, transform   =train_transform)
val_loader   = torch.utils.data.DataLoader(valset, batch_size=16, shuffle=False, num_workers=4)

testset       = MyDataset(submission_csv, transform = train_transform)
test_loader   = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)

model = EfficientNet.from_name('efficientnet-b4')
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load('./saved_model/20190905_data3k_effib4_aug_adjust_clean_testzoom.pt'))
model.cuda()

model.eval()

preds = []

with torch.no_grad():
    for idx, (imgs, labels) in enumerate(test_loader):
        imgs_vaild, labels_vaild = imgs.cuda(), labels.float().cuda()
        output_test = model(imgs_vaild)
        for i in range(len(output_test)):
            if output_test[i][0] < coef[0]:
                preds.append(0)
            elif output_test[i][0] >= coef[0] and output_test[i][0] < coef[1]:
                preds.append(1)
            elif output_test[i][0] >= coef[1] and output_test[i][0] < coef[2]:
                preds.append(2)
            elif output_test[i][0] >= coef[2] and output_test[i][0] < coef[3]:
                preds.append(3)
            else:
                preds.append(4)

submission_csv.diagnosis = np.array(preds).astype(int)
submission_csv.to_csv('20190905_effib4_aug_clean_testzoom.csv', index=False)
print('Done')
