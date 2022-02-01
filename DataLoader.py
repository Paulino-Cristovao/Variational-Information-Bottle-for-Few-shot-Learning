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
import PIL
import cv2

batch_size = 100

def gaussian_blur(img):
    image = np.array(img)
    image_blur = cv2.GaussianBlur(image,(5,5),1)
    new_image = image_blur
    return new_image

# define transforms
transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])

# custom dataset
class Data(Dataset):
    def __init__(self, data, labels=None, transforms=None):
        self.images = data
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image0 = self.images.iloc[index]
        image0 = np.asarray(image0).astype('uint8').reshape((28, 28, 1))
        label0 = self.labels.iloc[index]

        if self.transforms is not None:
            image0 = self.transforms(image0)
        return image0, label0
  
# dataloaders
def load_torch_data(k_shot, base_class):
    # mnist
    df_train_mnist = pd.read_csv('data/train.csv')
    df_train_mnist = df_train_mnist.iloc[np.random.permutation(len(df_train_mnist))]

    #Split data into train and test
    limit = 5000
    test = df_train_mnist[:limit]# train
    novel = df_train_mnist[limit:10000]# Test
    dataset_ft = df_train_mnist[10000:]
    
    # test data
    X_test = test.iloc[:,:-1]
    y_test = test.iloc[:,-1] + 30
    
    test_data = Data(X_test, y_test, transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Obtain single data per novel data
    if k_shot != 0:
        novel = novel.groupby('label', group_keys=False).apply(lambda x: x.iloc[:k_shot]) # k_shot[number of samples]

    # Novel data
    X_novel = novel.iloc[:,:-1]
    y_novel = novel.iloc[:,-1] + 30
  
    
    novel = Data(X_novel, y_novel, transform)
    novel_loader = torch.utils.data.DataLoader(novel, batch_size=batch_size, shuffle=True)


    # Fine Tune Imprinting
    X_ft = dataset_ft.iloc[:,:-1]
    y_ft = dataset_ft.iloc[:,-1] + 30

    test_ft = Data(X_ft, y_ft, transform)
    test_loader_ft = torch.utils.data.DataLoader(test_ft, batch_size=batch_size, shuffle=True)

    return test_loader, novel_loader, test_loader_ft
