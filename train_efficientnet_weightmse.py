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

os.environ["CUDA_VISIBLE_DEVICES"]= "1"


# Add them duong dan den Efficientnet Pytorch
package_path = './EfficientNet-PyTorch/'
NVIDIA_apex_path = './repository/NVIDIA-apex-39e153a/'
sys.path.append(package_path)
sys.path.append(NVIDIA_apex_path)

from efficientnet_pytorch import EfficientNet
from apex import amp

# Seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Config
num_classes = 1
seed_everything(1234)
lr          = 1e-3
IMG_SIZE    = 380
coef = [0.5, 1.5, 2.5, 3.5]


# Data Path
train      = '/media/asilla/data123/quangpn/APTOS2019/train/'
test       = '/media/asilla/data123/quangpn/APTOS2019/test/'
train_2015 = '/media/asilla/data123/quangpn/APTOS2019/train2015/'

train_csv = pd.read_csv('./data/train.csv')
submission_csv = pd.read_csv('./data/sample_submission.csv')

# train_2019_csv = pd.read_csv('./data/train.csv')
# train_2015_csv = pd.read_csv('./data/train2015.csv')

# data_df = {
#     'id_code': list(train_2019_csv['id_code'].append(train_2015_csv['image'], ignore_index=True)),
#     'diagnosis': list(train_2019_csv['diagnosis'].append(train_2015_csv['level'], ignore_index=True))
# }
# train_csv = pd.DataFrame(data=data_df)

train_df, val_df = train_test_split(train_csv, test_size=0.1, random_state=2018, stratify=train_csv.diagnosis)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)


def expand_path(p):
    p = str(p)
    if isfile(train + p + ".png"):
        return train + (p + ".png")
    if isfile(train_2015 + p + '.png'):
        return train_2015 + (p + ".png")
    if isfile(test + p + ".png"):
        return test + (p + ".png")
    return p

def p_show(imgs, label_name=None, per_row=3):
    n = len(imgs)
    rows = (n + per_row - 1)//per_row
    cols = min(per_row, n)
    fig, axes = plt.subplots(rows,cols, figsize=(15,15))
    for ax in axes.flatten(): ax.axis('off')
    for i,(p, ax) in enumerate(zip(imgs, axes.flatten())): 
        img = Image.open(expand_path(p))
        ax.imshow(img)
        ax.set_title(train_df[train_df.id_code == p].diagnosis.values)


# Crop Image
def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim == 2:
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


# Data Transformation
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


# Model Architecture
model = EfficientNet.from_name('efficientnet-b4')
model.load_state_dict(torch.load('./pretrained_effiientnet/efficientnet-b4-e116e8b3.pth'))
# model.load_state_dict(torch.load('weight_best.pt'))
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, num_classes)
model.cuda()


# Class weight
from sklearn.utils import class_weight
class_weight_ = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_df['diagnosis']),
                                                 train_df['diagnosis'])
class_weight_ = class_weight_ / np.sum(class_weight_)


# Weighted MSE Loss
def weighted_mse_loss(input,target):
    #alpha of 0.5 means half weight goes to first, remaining half split by remaining 15
    # weights = Variable(torch.Tensor([0.5,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15])).cuda()
    weights = torch.from_numpy(np.array([class_weight_[int(one_target)] for one_target in target]).reshape(len(target),1)).float().cuda()
    pct_var = (input-target)**2
    out = pct_var * weights
    loss = out.mean()
    return loss

# class MSELoss(_Loss):
#     __constants__ = ['reduction']

#     def __init__(self, size_average=None, reduce=None, reduction='mean'):
#         super(MSELoss, self).__init__(size_average, reduce, reduction)

#     def forward(self, input, target):
#         return F.mse_loss(input, target, reduction=self.reduction)

# Training Config
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
# criterion = nn.MSELoss() # Dung MSE vi la bai toan regression
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)


# Get Label
def get_label(output_test):
    label = 0
    if output_test < coef[0]:
        label = 0
    elif output_test >= coef[0] and output_test < coef[1]:
        label = 1
    elif output_test >= coef[1] and output_test < coef[2]:
        label = 2
    elif output_test >= coef[2] and output_test < coef[3]:
        label = 3
    else:
        label = 4

    return label


# Training
def train_model(epoch):
    model.train() 
        
    avg_loss = 0.
    optimizer.zero_grad()
    for idx, (imgs, labels) in enumerate(train_loader):
        imgs_train, labels_train = imgs.cuda(), labels.float().cuda()
        output_train = model(imgs_train)
        # loss = criterion(output_train,labels_train)
        loss = weighted_mse_loss(output_train,labels_train)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avg_loss += loss.item() / len(train_loader)
        
    return avg_loss

def test_model():
    correct = 0
    total = 0
    
    avg_val_loss = 0.
    model.eval()
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(val_loader):
            imgs_vaild, labels_vaild = imgs.cuda(), labels.float().cuda()
            output_test = model(imgs_vaild)
            # avg_val_loss += criterion(output_test, labels_vaild).item() / len(val_loader)
            avg_val_loss += weighted_mse_loss(output_test, labels_vaild).item() / len(val_loader)

            for i in range(len(output_test)):
                pred_label = get_label(output_test[i][0])
                correct += (int(pred_label) == int(labels[i][0]))
                total += 1
            
        val_acc = correct * 100.0 / total
        
    return avg_val_loss, val_acc


best_avg_loss = 100.0
best_val_acc = 0.0
n_epochs      = 20

for epoch in range(n_epochs):
    
    print('lr:', scheduler.get_lr()[0]) 
    start_time   = time.time()
    avg_loss     = train_model(epoch)
    avg_val_loss, val_acc = test_model()
    elapsed_time = time.time() - start_time 
    print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t val_acc={:.4f}% \t time={:.2f}s'.format(
        epoch + 1, n_epochs, avg_loss, avg_val_loss, val_acc, elapsed_time))
    
    # if avg_val_loss < best_avg_loss:
    #     best_avg_loss = avg_val_loss
    #     torch.save(model.state_dict(), './saved_model/20190809_data3k_effi_b4.pt')
    #     print('Better val_loss. Model saved!')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), './saved_model/20190810_weight_data3k_effi_b4.pt')
        print('Better val_accuracy. Model saved!')
    
    scheduler.step()

print('Finish Training!')