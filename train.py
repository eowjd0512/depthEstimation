import sys
import os
import io
import requests
from optparse import OptionParser
import numpy as np

from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import time 
import shutil
from eval import eval_net
from depthnet import depthnet_model as model
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch


device1 = torch.device("cuda:0")
device2 = torch.device("cuda:1")
device3 = torch.device("cuda:2")
device4 = torch.device("cuda:3")
device5 = torch.device("cuda:4")
device6 = torch.device("cuda:5")
device7 = torch.device("cuda:6")
device8 = torch.device("cuda:7")

def train_net(net,
              epochs=20,
              batch_size=32,
              lr=0.0001,
              val_percent=0.05,
              save_cp=True,
              gpu=True,
              img_scale=0.5):

    dir_img = 'data/image_train/'
    dir_depth = 'data/depth_train/'
    dir_checkpoint = 'checkpoints/'
    print(torch.cuda.memory_allocated())
    print(torch.cuda.empty_cache())
    best_prec1=0
    ids = get_ids(dir_depth)
    #ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)
    print("iddataset size: ", len(iddataset['train']))

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])
    w = int(1242/2)
    h = int(375/2)
    #optimizer = optim.SGD(net.parameters(),
    #                      lr=lr,
    #                      momentum=0.9,
    #                      weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters(),
                          lr=lr)
    criterion = model.ScaleInvariantLoss(batch_size).cuda()
    preprocess = transforms.Compose([
        transforms.Resize((h,w)),
        transforms.ToTensor(),
    ])
    best_epoch=0
    for epoch in range(epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        #adjust_learning_rate(optimizer, epoch)
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        print("hi there")
        start = time.time()
        # reset the generators
        #train = get_imgs_and_masks(iddataset['train'], dir_img, dir_depth, img_scale)
        #val = get_imgs_and_masks(iddataset['val'], dir_img, dir_depth, img_scale)
        #print("hi there2")
        #print("batch size: ",batch_size)
        #print ("imgs_t2 shape: ", list(zip(train)))
        epoch_loss = 0
<<<<<<< HEAD
        num=0

        for i in range(N_train-batch_size+1):
            start2 = time.time()
            t2_id = int(iddataset['train'][i])
            t1_id=t2_id+1
            t0_id=t2_id+2
            img_t2 = Image.open(dir_img+str(t2_id)+'.png')
            if t1_id in ids:
                img_t1 = Image.open(dir_img+str(t1_id)+'.png')
                if t0_id in ids:
                    img_t0 = Image.open(dir_img+str(t0_id)+'.png')
                else:
                    img_t0 = Image.open(dir_img+str(t1_id)+'.png')
            else:
                img_t1 = Image.open(dir_img+str(t2_id)+'.png')
                img_t0 = Image.open(dir_img+str(t2_id)+'.png')

            depth_t2 = Image.open(dir_depth+str(t2_id)+'.png')
            if t1_id in ids:
                depth_t1 = Image.open(dir_depth+str(t1_id)+'.png')
                if t0_id in ids:
                    depth_t0 = Image.open(dir_depth+str(t0_id)+'.png')
                else:
                    depth_t0 = Image.open(dir_depth+str(t1_id)+'.png')
            else:
                depth_t1 = Image.open(dir_depth+str(t2_id)+'.png')
                depth_t0 = Image.open(dir_depth+str(t2_id)+'.png')

            img_t2=preprocess(img_t2)
            img_t2.unsqueeze_(0)
            img_t1=preprocess(img_t1)
            img_t1.unsqueeze_(0)
            img_t0=preprocess(img_t0)
            img_t0.unsqueeze_(0)
            depth_t2=preprocess(depth_t2)
            depth_t2.unsqueeze_(0)
            depth_t1=preprocess(depth_t1)
            depth_t1.unsqueeze_(0)
            depth_t0=preprocess(depth_t0)
            depth_t0.unsqueeze_(0)

            for b in range(1,batch_size):
                t2_idb = int(iddataset['train'][i+b])
                t1_idb=t2_id+1
                t0_idb=t2_id+2
                img_t2b = Image.open(dir_img+str(t2_idb)+'.png')
                if t1_idb in ids:
                    img_t1b = Image.open(dir_img+str(t1_idb)+'.png')
                    if t0_idb in ids:
                        img_t0b = Image.open(dir_img+str(t0_idb)+'.png')
                    else:
                        img_t0b = Image.open(dir_img+str(t1_idb)+'.png')
                else:
                    img_t1b = Image.open(dir_img+str(t2_idb)+'.png')
                    img_t0b = Image.open(dir_img+str(t2_idb)+'.png')

                depth_t2b = Image.open(dir_depth+str(t2_idb)+'.png')
                if t1_idb in ids:
                    depth_t1b = Image.open(dir_depth+str(t1_idb)+'.png')
                    if t0_idb in ids:
                        depth_t0b = Image.open(dir_depth+str(t0_idb)+'.png')
                    else:
                        depth_t0b = Image.open(dir_depth+str(t1_idb)+'.png')
                else:
                    depth_t1b = Image.open(dir_depth+str(t2_idb)+'.png')
                    depth_t0b = Image.open(dir_depth+str(t2_idb)+'.png')


                img_t2b=preprocess(img_t2b)
                img_t2b.unsqueeze_(0)
                img_t1b=preprocess(img_t1b)
                img_t1b.unsqueeze_(0)
                img_t0b=preprocess(img_t0b)
                img_t0b.unsqueeze_(0)
                depth_t2b=preprocess(depth_t2b)
                depth_t2b.unsqueeze_(0)
                depth_t1b=preprocess(depth_t1b)
                depth_t1b.unsqueeze_(0)
                depth_t0b=preprocess(depth_t0b)
                depth_t0b.unsqueeze_(0)
                
                img_t2=torch.cat((img_t2,img_t2b),0)
                img_t1=torch.cat((img_t1,img_t1b),0)
                img_t0=torch.cat((img_t0,img_t0b),0)
                depth_t2=torch.cat((depth_t2,depth_t2b),0)
                depth_t1=torch.cat((depth_t1,depth_t1b),0)
                depth_t0=torch.cat((depth_t0,depth_t0b),0)

            


        #for i, b in enumerate(batch(train, batch_size)):
        #    print("hi ", num)
        #    data_time.update(time.time() - end)
        #    imgs_t2 = np.array([i[0] for i in b]).astype(np.float32)
         #   imgs_t1 = np.array([i[1] for i in b]).astype(np.float32)
          #  imgs_t0 = np.array([i[2] for i in b]).astype(np.float32)
           # dpths_t2 = np.array([i[3] for i in b]).astype(np.float32)
            #dpths_t1 = np.array([i[4] for i in b]).astype(np.float32)
            #dpths_t0 = np.array([i[5] for i in b]).astype(np.float32)
            #true_depths = np.array([i[1] for i in b])
            #for i in b:
            ##    print ("imgs_t2 shape: ", i[0].shape)
            #imgs_t2 = torch.from_numpy(imgs_t2)
            #imgs_t1 = torch.from_numpy(imgs_t1)
            #imgs_t0 = torch.from_numpy(imgs_t0)
            #dpths_t2 = torch.from_numpy(dpths_t2)
            ##dpths_t1 = torch.from_numpy(dpths_t1)
            #dpths_t0 = torch.from_numpy(dpths_t0)
            print("hello, timg_t2 shape? : ",img_t2.shape)
            #if gpu:
            
            img_t2 = torch.autograd.Variable(img_t2).cuda(0)
            img_t1 = torch.autograd.Variable(img_t1).cuda(0)
            img_t0 = torch.autograd.Variable(img_t0).cuda(0)
            dpth_t2 = torch.autograd.Variable(depth_t2).cuda(0)
            dpth_t1 = torch.autograd.Variable(depth_t1).cuda(0)
            dpth_t0 = torch.autograd.Variable(depth_t0).cuda(0)
            
            pred_t2,pred_t1,pred_t0= net(img_t2, img_t1, img_t0)
            #masks_probs = F.sigmoid(masks_pred)
            #masks_probs_flat = masks_probs.view(-1)

            #true_masks_flat = true_masks.view(-1)

            pred=np.array([pred_t2,pred_t1,pred_t0])
            truth=np.array([dpth_t2,dpth_t1,dpth_t0])

            #print("deptht2 shape: ", dpth_t2.shape)
            #print("deptht1 shape: ", dpth_t1.shape)
            #print("deptht0 shape: ", dpth_t0.shape)

            loss = criterion(pred, truth,w,h)
=======

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
>>>>>>> 7dd7c8b6346033ed1b15a2c1ab54847016a826db
            epoch_loss += loss.item()

            end2 = time.time()
            print(i+1, "th item time:  ", end2- start2, "sec")
            print('{0:.4f} --- loss: {1:.6f}'.format((i+1) * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #batch_time.update(time.time() - end)

            print(" ")
        


            num =num+1

        end = time.time()
        print(epoch, "th epoch time:  ", end - start, "sec")
        print('Epoch finished ! Loss: {}'.format(epoch_loss / (i+1)))


        #validation
        print("Validation")
        prec1 = eval_net(net, iddataset, ids,dir_img,dir_depth,batch_size, criterion, preprocess, w,h, gpu)
        

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        if prec1 > best_prec1:
            best_epoch=epoch
        
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)

    print("best epoch in valid: ", best_epoch+1)
        #if save_cp:
        #    torch.save(net.state_dict(),
        #               dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
        #    print('Checkpoint {} saved !'.format(epoch + 1))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=20, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=32,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.0001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    
    hidden_sizes=[16,32,64,128,256]
    kernel_sizes=[3,3,3,3,3]
    n_layer=5
    strides=[1,1,1,1,1]
    n_channels=3
    net = model.DepthNet(n_channels,hidden_sizes,kernel_sizes,n_layer,strides)

    


    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda(0)
        #net=nn.DataParallel(net)
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
