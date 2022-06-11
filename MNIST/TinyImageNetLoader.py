"""
https://github.com/pranavphoenix/TinyImageNetLoader/blob/main/tinyimagenetloader.py
"""

# !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
# !unzip -q tiny-imagenet-200.zip

from requests import head
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os, glob
from torchvision.io import read_image, ImageReadMode
import pandas as pd
import numpy as np

batch_size = 64

base_path = "/home/whq/Downloads/"
id_dict = {}
for i, line in enumerate(open(base_path + 'tiny-imagenet-200/wnids.txt', 'r')):
  id_dict[line.replace('\n', '')] = i

class TrainTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob(base_path + "tiny-imagenet-200/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.id_dict[img_path.split('/')[6]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob(base_path + "tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open(base_path + 'tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

transform = transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))

trainset = TrainTinyImageNetDataset(id=id_dict)

train_data = []
test_data = []
for feature, label in trainset:
    img = feature.flatten().numpy()
    label = np.array(label).reshape(-1)
    merge_sample = np.concatenate((img, label))
    train_data.append(merge_sample)

train_data = np.array(train_data)
print(train_data.shape)
print(train_data[0])
df = pd.DataFrame(train_data)
df.to_csv(base_path + "imagenet_train.csv", index=False, header=True)

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = TestTinyImageNetDataset(id=id_dict)
for feature, label in testset:
    img = feature.flatten().numpy()
    label = np.array(label).reshape(-1)
    merge_sample = np.concatenate((img, label))
    test_data.append(merge_sample)

test_data = np.array(test_data)
print(test_data.shape)
df = pd.DataFrame(test_data)
df.to_csv(base_path + "imagenet_test.csv", index=False, header=True)

# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)