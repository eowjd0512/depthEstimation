# full assembly of the sub-parts to form the complete net

from .unet_parts import *
from depthnet import convgru as cg
from depthnet import decoder as dc
import numpy as np

class DepthNet(nn.Module):
    def __init__(self, n_channels,hidden_sizes,kernel_sizes,n_layer,strides):
        super(DepthNet, self).__init__()


        self.model1=cg.ConvGRU(n_channels,hidden_sizes,kernel_sizes,n_layer,strides)
        #self.model2=cg.ConvGRU(n_channels,hidden_sizes,kernel_sizes,n_layer,strides)
        #self.model3=cg.ConvGRU(n_channels,hidden_sizes,kernel_sizes,n_layer,strides)

        self.decoder_t2=dc.decoder(hidden_sizes)
        self.decoder_t1=dc.decoder(hidden_sizes)
        self.decoder_t0=dc.decoder(hidden_sizes)
        #self.inc = inconv(n_channels, 64)
        #self.down1 = down(64, 128)
        #self.down2 = down(128, 256)
        #self.down3 = down(256, 512)
        #self.down4 = down(512, 512)
        #self.up1 = up(1024, 256)
        #self.up2 = up(512, 128)
        #self.up3 = up(256, 64)
        #self.up4 = up(128, 64)
        #self.outc = outconv(64, n_classes)

    def forward(self, x_t2,x_t1,x_t0):
        
        h_t2=self.model1(x_t2)
        print("model1 done")
        h_t1=self.model1(x_t1,h_t2)
        print("model2 done")
        h_t0=self.model1(x_t0,h_t1)
        print("model3 done")

        #decoder for each time
        pred_t2=self.decoder_t2(h_t2[-1],h_t2)
        pred_t1=self.decoder_t1(h_t1[-1],h_t1)
        pred_t0=self.decoder_t0(h_t0[-1],h_t0)

        #return pred_t0

        return pred_t2,pred_t1,pred_t0

class ScaleInvariantLoss(nn.Module):
    def __init__(self, batchsize):
        super(ScaleInvariantLoss, self).__init__()
        self.bsize=batchsize

    def forward(self,preds,truths,w,h):
        totalloss=0
        depth_cost=0
        num=3
        for i in range(num):
            npix = int(h*w)
            #truth[i]=torch.mul(truth[i],1/255)
            print("npix : ", npix)

            truth=truths[i].reshape((self.bsize,npix)).type('torch.cuda.FloatTensor')
            pred=preds[i].reshape((self.bsize,npix))
            mask=torch.ne(truth,torch.min(truth)).type('torch.cuda.FloatTensor')


            log_truth=torch.log(truth)

            log_truth=torch.mul(mask,log_truth)
            log_truth[log_truth.ne(log_truth)]=0


            log_pred=torch.log(pred)
            log_pred=torch.mul(mask,log_pred)
            log_pred[log_pred.ne(log_pred)]=0

            d=torch.sub(log_pred,log_truth)

           
            n_nonzero=torch.sum(mask,1)
            sum_d_2=torch.sum(d**2,1)
            sum_d=torch.sum(d,1)
            pow_sum_d = sum_d**2
            lambda_=0.5	



            a = torch.sum(n_nonzero*sum_d_2)
            b = torch.sum(pow_sum_d)
            c = torch.sum(n_nonzero)**2


            depth_cost=depth_cost+(a-lambda_*b)/c

        '''
		bsize = self.bsize
		npix = int(np.prod(test_shape(y0)[1:]))
		y0_target = y0.reshape((self.bsize, npix))
		y0_mask = m0.reshape((self.bsize, npix))
		pred = pred.reshape((self.bsize, npix))

		p = pred * y0_mask
		t = y0_target * y0_mask

		d = (p - t)

		nvalid_pix = T.sum(y0_mask, axis=1)
		depth_cost = (T.sum(nvalid_pix * T.sum(d**2, axis=1))
		                 - 0.5*T.sum(T.sum(d, axis=1)**2)) \
		             / T.maximum(T.sum(nvalid_pix**2), 1)

        '''
        return depth_cost/num



