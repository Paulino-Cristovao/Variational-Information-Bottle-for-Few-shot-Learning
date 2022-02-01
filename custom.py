import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



# read the data
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# get the image pixel values and labels
train_labels = df_train.iloc[:, 0]
train_images = df_train.iloc[:, 1:]

test_labels = df_test.iloc[:, 0]
test_images = df_test.iloc[:, 1:]


# define transforms
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))
])



num_classes=10


# custom dataset
class MNISTDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None, num_classes=num_classes, num_train_sample=1, novel_only=False, aug=False):
        self.data = images
        self.y = labels
        self.transforms = transforms

        # split dataset
        data = self.data[self.data['label'] < num_classes]
        base_data = data[data['label'] < 5]
        novel_data = data[data['label'] >= 5]
        #
        
        # sampling from novel classes
        if num_train_sample != 0:
            #print(len(novel_data[:num_train_sample]))
            novel_data = novel_data.groupby('label', group_keys=False).apply(lambda x: x.iloc[:num_train_sample])

            #print(len(novel_data))
            #print()
            #print()

        # whether only return data of novel classes
        if novel_only:
            data = novel_data
            
        else:
            data = pd.concat([base_data, novel_data])
            

        # repeat 5 times for data augmentation
        if aug:
            tmp_data = pd.DataFrame()
            for i in range(5):
                tmp_data = pd.concat([tmp_data, data])
            data = tmp_data
        imgs = data.reset_index(drop=True)
         
    
    def __len__(self):
        return (len(self.data))
    
    def __getitem__(self, i):
        data = self.data.iloc[i, :]
        data = np.asarray(data).astype(np.uint8).reshape(28, 28, 1)
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data


train_data = MNISTDataset(train_images, train_labels, transform)
test_data = MNISTDataset(test_images, test_labels, transform)

# dataloaders
trainloader = DataLoader(train_data, batch_size=64,shuffle=True)
testloader = DataLoader(test_data, batch_size=64, shuffle=True)