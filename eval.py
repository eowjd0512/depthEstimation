import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from dice_loss import dice_coeff
from depthnet import depthnet_model as model
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
import time

def eval_net(net, iddataset, ids,dir_img,dir_depth,batch_size, criterion, preprocess, w,h, gpu=True):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    epoch_loss=0
    N_train = len(iddataset['val'])
    with torch.no_grad():
        for i in range(N_train-batch_size+1):
            start = time.time()
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
            end = time.time()
            print("pred time :" , end-start, "sec")
            #true_masks_flat = true_masks.view(-1)

            pred=np.array([pred_t2,pred_t1,pred_t0])
            truth=np.array([dpth_t2,dpth_t1,dpth_t0])

<<<<<<< HEAD
            #print("deptht2 shape: ", dpth_t2.shape)
            #print("deptht1 shape: ", dpth_t1.shape)
            #print("deptht0 shape: ", dpth_t0.shape)
=======
        mask_pred = net(img)[0]
        mask_pred = (mask_pred > 0.5).float()
>>>>>>> 7dd7c8b6346033ed1b15a2c1ab54847016a826db

            loss = criterion(pred, truth,w,h)
            epoch_loss += loss.item()
            
            print('eval::: {0:.4f} --- loss: {1:.6f}'.format((i+1)  * batch_size / N_train, loss.item()))
    return epoch_loss / (i+1)
