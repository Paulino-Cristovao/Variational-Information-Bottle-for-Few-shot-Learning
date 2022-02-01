# all imprinting
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
from tqdm import tqdm_notebook as tqdm
from DataLoader import load_torch_data
from helper_functions import*
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import gzip, pickle
import matplotlib.cm as cm
import random
import math
from tsne import bh_sne
from prints import*


# Devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_features = 512
classes = 40
base_class = 30
novel_class = 10
alpha = 1

def imprint(novel_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    bar = Bar('Imprinting', max=len(novel_loader))
    with torch.no_grad():
        for batch_idx, data in enumerate(novel_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = data[0].to(device)
            target = data[1].long().to(device)

            # compute output
            # compute output
            _,_,_,feature = model.helper_extract(input)
            output = F.normalize(feature,p=2,dim=feature.dim()-1,eps=1e-12)
            
            
            if batch_idx == 0:
                output_stack = output
                target_stack = target
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, target), 0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        batch=batch_idx + 1,
                        size=len(novel_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td
                        )
            bar.next()
        bar.finish()
      
    
    new_weight = torch.zeros(base_class, num_features)

    for i in range(novel_class):
        tmp = output_stack[target_stack == (i + base_class)].mean(0) if not args.random else torch.randn(10)
        new_weight[i] = tmp / tmp.norm(p=2)
        #new_weight[i] = F.normalize(tmp, p=2, dim=tmp.dim()-1, eps=1e-12)
 


    weight = torch.cat((model.classifier.fc.weight.data[:base_class], new_weight.to(device)))
    model.classifier.fc = nn.Linear(num_features, classes, bias=False)
    model.classifier.fc.weight.data = weight
    
def validate(val_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    out_target = []
    out_data = []
    out_output = []
    feat_ = []

    class_correct = [0]*10
    class_total = [0]*10
    conf_matrix = np.zeros((10, 10))

    latent_fc2 = Proj_Latent()
    latent_fc1 = Proj_Latent()

    # switch to evaluate mode
    model.eval()
    bar = Bar('Testing   ', max=len(val_loader))
    with torch.no_grad():
        end = time.time()
        for batch_idx, data in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = data[0].to(device)
            label = data[1].long().to(device)

            # compute output
            output,feature2, feature1 = model.forward_wi_fc1(input)

            #predict --> convert output probabilities to predicted class
            pred = output.argmax(1)
           

            # measure accuracy
            prec1, prec5 = accuracy(output, label, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))


            # compare predictions to true label
            correct_tensor = pred.eq(label.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if device == "cpu" else np.squeeze(correct_tensor.cpu().numpy())

            #conf_matrix = c_matrix(label, pred, correct)
            for label_, pred_, correct_ in zip(label, pred, correct):
                label_ -=30
                class_correct[label_] += correct_
                class_total[label_] += 1
                if pred_.data >=30:
                  # update confusion matrix
                  conf_matrix[label_][pred_-30] += 1
           
            
            # project latent representations
            data_feature_extractor_fc2 = latent_fc2.append(feature2,label)
            data_feature_extractor_fc1= latent_fc1.append(feature1,label)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

             # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return top1.avg, conf_matrix, data_feature_extractor_fc2, data_feature_extractor_fc1


