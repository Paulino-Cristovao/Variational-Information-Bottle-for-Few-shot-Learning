# helper functions
## dependecies 
import os
import shutil
import torch
import torch.nn as nn
import models
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision
#from tqdm import tqdm_notebook as tqdm
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import gzip, pickle
import matplotlib.cm as cm
import random
import math


#data_save = 'savefiles_imprinting'

def imshow(img, title=''):
  #Plot image batch
  plt.figure(figsize=(28, 28))
  plt.title(title)
  plt.imshow(np.transpose(img.numpy(),(1, 2, 0)), cmap='gray') 
  #plt.show()
  plt.clf()



# plot confusion matrix
def plot_confusion_matric(data,path,name,test_avg_acc):
    plt.subplots(figsize=(10, 9))
    ax = sns.heatmap(data, annot=None, cmap='BuPu', yticklabels=True)
    ax.set_xlabel('predicted')
    ax.set_ylabel('True')
    #plt.title('Confision Matrix')
    plt.savefig(path +'/confusion_matrix'+'_'+name+'_'+str(test_avg_acc)+'.png', bbox_inches='tight')
    #plt.show()
    plt.clf()



def im_convert(tensor):
  image = tensor.clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image

def gaussian_blur(img):
    image = np.array(img)
    image_blur = cv2.GaussianBlur(image,(5,5),2)
    new_image = image_blur
    return new_image



# For further ploting the loss and test data
class TrainingHistory:
  def __init__(self):
    # vib
    self.train_acc_history_vib = []
    self.test_acc_history_vib = []
    self.train_loss_history_vib = []
    self.test_loss_history_vib = []

    

  def plot(self,path):
    plt.plot(self.train_acc_history_vib, label='Training Accuracy')
    plt.plot(self.test_acc_history_vib, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.grid(True)
    plt.savefig(path + '/Accuracy.png', bbox_inches='tight')
    #plt.show()
    plt.clf()

    plt.plot(self.train_loss_history_vib, label='Training Loss')
    plt.plot(self.test_loss_history_vib, label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.grid(True)
    plt.savefig(path + '/Loss.png', bbox_inches='tight')
    #plt.show()
    plt.clf()

  def append(self, losses_train, losses_test, top1_train, top1_test):
    self.train_acc_history_vib.append(top1_train)
    self.test_acc_history_vib.append(top1_test)
    self.train_loss_history_vib.append(losses_train)
    self.test_loss_history_vib.append(losses_test)


####
#### projection
class Proj_Latent:
  def __init__(self):
      self.out_target = []
      self.feat_ = []
  
  def append(self,feature,label):
      # Projection LR
      self.feat_np = feature.data.cpu().numpy()
      self.target_np = label.data.cpu().numpy()

      self.feat_.append(self.feat_np)
      self.out_target.append(self.target_np[:, np.newaxis])

      self.feat_array = np.concatenate(self.feat_, axis=0)
      self.target_array = np.concatenate(self.out_target, axis=0)

      return self.feat_array, self.target_array