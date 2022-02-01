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
import numpy as np
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from DataLoader import load_torch_data
from helper_functions import*
import seaborn as sns
import matplotlib.pyplot as plt
import gzip, pickle
import matplotlib.cm as cm
import random
import math
from tsne import bh_sne



parser = argparse.ArgumentParser(description='PyTorch Omniglot Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-c', '--checkpoint', default='imprint_ft_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: imprint_ft_checkpoint)')
parser.add_argument('-d', '--save', default='savefiles_Imprinting_FT', type=str, metavar='PATH',
                    help='path to save files (default: save_files)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to model (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--random', action='store_true', help='whether use random novel weights')
parser.add_argument('--numsample', default=1, type=int,
                    metavar='N', help='number of novel sample (default: 1)')
parser.add_argument('--test-novel-only', action='store_true', help='whether only test on novel classes')
best_prec1 = 0


# Devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_features = 512
classes = 40
base_class = 30
novel_class = 10
#alpha = 1


def main():
    global args, best_prec1
    args = parser.parse_args()
    history = TrainingHistory()

    accuracy_save = []

    if not os.path.isdir(args.checkpoint):
        mkdir_p(str(args.batch_size)+'/'+str(args.epochs)+'/'+args.save)

    if not os.path.isdir(args.save):
        mkdir_p(str(args.batch_size)+'/'+str(args.epochs)+'/'+args.checkpoint)

    model = models.Net().to(device)


    print('==> Reading from model checkpoint..')
    assert os.path.isfile(args.model), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model checkpoint '{}' (epoch {})"
            .format(args.model, checkpoint['epoch']))
    cudnn.benchmark = True

    # Data loading code
    test_loader,novel_loader,train_loader = load_torch_data(args.numsample, base_class)

    # imprint weights first
    print('Imprinting ...')
    print()
    imprint(novel_loader, model)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.94)

    title = 'Impriningt + FT'
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        logger = Logger(os.path.join(str(args.batch_size)+'/'+str(args.epochs)+'/'+args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(str(args.batch_size)+'/'+str(args.epochs)+'/'+args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        #lr = optimizer.param_groups[0]['lr']
        lr = args.lr
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        # train for one epoch
        print('Training ...')
        print()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        print('validation ...')
        print()
        test_loss, test_acc, conf_matrix, feat_fc2, feat_fc1 = validate(test_loader, model, criterion)

        # append logger file
        logger.append([lr, train_loss, test_loss, train_acc, test_acc])

        history.append(train_loss, test_loss, train_acc, test_acc)

        # remember best prec@1 and save checkpoint
        is_best = test_acc > best_prec1
        best_prec1 = max(test_acc, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=str(args.batch_size)+'/'+str(args.epochs)+'/'+args.checkpoint)

    logger.close()
    #logger.plot()
    #savefig(os.path.join(str(args.batch_size)+'/'+str(args.epochs)+'/'+args.checkpoint, 'log.eps'))

    # Plot the representations
    feature_fc2 = feat_fc2[0].astype(np.float64)
    target = feat_fc2[1]

    # feature fc2
    feature_fc2 = bh_sne(feature_fc2[-3856:])
    target = target[-3856:]

    plt.rcParams['figure.figsize'] = 20, 20
    plt.scatter(np.reshape(feature_fc2[:, 0], -1), np.reshape(feature_fc2[:, 1], -1), c=np.reshape(target,-1), cmap=plt.cm.get_cmap("jet", 30))
    plt.colorbar(ticks=range(30))
    plt.title('Validation: Latent Projection-fc2')
    plt.savefig(str(args.batch_size)+'/'+str(args.epochs)+'/'+args.save+'/fc2.png', bbox_inches='tight')
    #plt.show()
    plt.clf()


    # from fc1
    feature_fc1 = feat_fc1[0].astype(np.float64)
    target_val = feat_fc1[1]

    # feature
    feature_fc1 = bh_sne(feature_fc1[-3856:])
    target_val = target_val[-3856:]

    plt.rcParams['figure.figsize'] = 20, 20
    plt.scatter(np.reshape(feature_fc1[:, 0], -1), np.reshape(feature_fc1[:, 1], -1), c=np.reshape(target_val,-1), cmap=plt.cm.get_cmap("jet", 30))
    plt.colorbar(ticks=range(30))
    plt.title('Validation: Latent Projection-fc1')
    plt.savefig(str(args.batch_size)+'/'+str(args.epochs)+'/'+args.save+'/fc1.png', bbox_inches='tight')
    #plt.show()
    plt.clf()

    # plot confusion matrix
    test_acc = str(round(test_acc, 2))
    plot_confusion_matric(conf_matrix,str(args.batch_size)+'/'+str(args.epochs)+'/'+args.save,'fc2_after_relu',test_acc)

    # plot train and accuracy loss
    history.plot(str(args.batch_size)+'/'+str(args.epochs)+'/'+args.save)

    accuracy_save.extend([args.batch_size,test_acc])
    np.save(str(args.batch_size)+'/'+str(args.epochs)+'/'+args.save+'_'+str(args.batch_size)+'_'+'acc_ft.npy',accuracy_save)

    print('Best acc:')
    print(best_prec1)


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

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    bar = Bar('Training', max=len(train_loader))

    for batch_idx, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #data = mixup(data[0], data[1].long(), alpha, base_class)
        input = data[0].to(device)
        label = data[1].to(device)
        #label = torch.max(label, 1)[1]

        # output 
        logit, feature,_= model(input)
        
        # loss     
        loss = criterion(logit, label)

        # prediction vib
        pred = logit.argmax(1)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logit, label, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        model.weight_norm()
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)
    
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    out_target = []
    out_data = []
    out_output = []
    feat_ = []

    class_loss = 0
    info_loss = 0
    loss_vib = 0
    total_num = 0

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
            output, feature2, feature1 = model(input)
            loss = criterion(output, label)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, label, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            #predict --> convert output probabilities to predicted class
            pred = output.argmax(1)


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
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg, conf_matrix, data_feature_extractor_fc2, data_feature_extractor_fc1)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()