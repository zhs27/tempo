import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from util.pcview import PCViews

from . import wa_module

from model.backbone.cartoonx import CartoonX

'''
In this model,
I replace the maxpooing among the frame dimension with my own 
aggregation method.
'''



class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)


class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)
            #wablock replacement
            #self.pool2d = wa_module.wa_module()
    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1,c,h,w))
        if self.pooling:
            x = self.pool2d(x)
            #x = x[0]
        _, c, h, w = x.size()
        return x.view(n, s, c, h ,w)




class view_pooling(nn.Module):
    def __init__(self,inchannel=32,out_channel=32):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(4*inchannel,2*out_channel,kernel_size=3,padding=1),
                               nn.ReLU(),
                               nn.Conv2d(2*inchannel,1*out_channel,kernel_size=3,padding=1),
                               nn.ReLU())
    
    
    def forward(self,x):
        '''
        x's shape is (bs,6,32,64,64)
        '''
        lr=torch.max(x[:,[0,2],:,:],1)[0] # left and right
        fb=torch.max(x[:,[1,3],:,:],1)[0] # front and back
        tb=torch.max(x[:,[4,5],:,:],1)[0] # top and bottom

        # lft=torch.max(x[:,[0,1,4],:,:],1)[0] # left front and top
        # rbb=torch.max(x[:,[2,3,5],:,:],1)[0] # right back and bottom

        al=torch.max(x,1)[0]

        feat=torch.cat([al,lr,fb,tb],1)
        feat=self.net(feat)

        return feat


class view_pooling_attention(nn.Module):
    def __init__(self,inchannel=32,out_channel=32):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(6*inchannel,out_channel,kernel_size=3,padding=1),
                               nn.ReLU())


        self.score_mat=nn.Sequential(nn.Conv2d(6*inchannel,inchannel,kernel_size=1),
                                    nn.ReLU(),
                                    nn.Conv2d(inchannel,6,kernel_size=1),
                                    nn.Sigmoid())
        self.inchannel=inchannel
    
    def forward(self,x):
        '''
        x's shape is (bs,6,32,64,64)
        '''
        lr=torch.max(x[:,[0,2],:,:],1)[0] # left and right
        fb=torch.max(x[:,[1,3],:,:],1)[0] # front and back
        tb=torch.max(x[:,[4,5],:,:],1)[0] # top and bottom
        
        lft=torch.max(x[:,[0,1,4],:,:],1)[0] # left front and top
        rbb=torch.max(x[:,[2,3,5],:,:],1)[0] # right back and bottom
        
        al=torch.max(x,1)[0]
        
        feat=torch.cat([al,lr,fb,tb,lft,rbb],1)
        sm=self.score_mat(feat)
        sm=torch.repeat_interleave(sm,self.inchannel,dim=1)
        feat=feat*sm

        feat=self.net(feat)
        
        return feat






class ViewNetpt(nn.Module):
    def __init__(self):
        super().__init__()
        self.pcview=PCViews()
        self.hidden_dim=256
        
        self.vp1=view_pooling(inchannel=32,out_channel=32)
        self.vp2=view_pooling(inchannel=64,out_channel=64)
        self.vp3=view_pooling(inchannel=128,out_channel=128)
        
        _set_in_channels = 1
        _set_channels = [32, 64, 128]
        self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels, _set_channels[0], 5, padding=2))
        self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 3, padding=1), True)
        self.set_layer3 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[1], 3, padding=1))
        self.set_layer4 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[1], 3, padding=1), True)
        self.set_layer5 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[2], 3, padding=1))
        self.set_layer6 = SetBlock(BasicConv2d(_set_channels[2], _set_channels[2], 3, padding=1))


        _gl_in_channels = 32
        _gl_channels = [64, 128]
        
        self.gl_layer1 = BasicConv2d(_gl_in_channels, _gl_channels[0], 3, padding=1)
        self.gl_layer2 = BasicConv2d(_gl_channels[0], _gl_channels[0], 3, padding=1)
        self.gl_layer3 = BasicConv2d(_gl_channels[0], _gl_channels[1], 3, padding=1)
        self.gl_layer4 = BasicConv2d(_gl_channels[1], _gl_channels[1], 3, padding=1)
        self.gl_layer5 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.gl_pooling = nn.MaxPool2d(2)
        # Replace max pooling by wavelet block
        self.gl_pooling2 = wa_module.wa_module()


        # ===== bin number ======
        self.bin_num = [1, 2, 4, 8, 16]
        # self.bin_num = [1, 2, 4]
        # =======================
        
        self.final=nn.Linear(128,self.hidden_dim)

        self.compress=nn.Sequential(nn.Linear(62,16),
                                nn.ReLU(),
                                nn.Linear(16,1))



    def frame_max(self, x):
        return torch.max(x, 1)




    def get_img(self,inpt):
        bs=inpt.shape[0]
        imgs=self.pcview.get_img(inpt.permute(0,2,1))
        
        _,h,w=imgs.shape
        
        imgs=imgs.reshape(bs,6,-1)
        max=torch.max(imgs,-1,keepdim=True)[0]
        min=torch.min(imgs,-1,keepdim=True)[0]
        
        nor_img=(imgs-min)/(max-min+0.0001)
        nor_img=nor_img.reshape(bs,6,h,w)
        return nor_img



    def forward(self,inpt, xtocartoonx= None, modelQh = None, mode = "train", mixup=False, target=None,lam=0.4):
        '''
        norm_img shape is (20,6,128,128)
        20 is the batch_size
        6 is the view number
        128 is the image size
        '''
        CARTOONX_HPARAMS = {
        "l1lambda": 0.01, "lr": 1e-1, 'obfuscation': 'gaussian',
        "maximize_label": False, "optim_steps": 30,  
        "noise_bs": 1, 'mask_init': 'ones'
        }
        if modelQh != None :
            cartoonx_method = CartoonX(model=modelQh, device='cuda', **CARTOONX_HPARAMS)
            with torch.no_grad():
                pred,loss=modelQh(inpt)
            if(xtocartoonx != None):
                cartoonx = cartoonx_method(inpt,torch.argmax(pred, dim = 1).detach(),xtocartoonx)
                inpt = torch.stack(cartoonx)
            else:
                '''
                get masked img with modelQh and cartoonX
                '''
                cartoonx = cartoonx_method(inpt,torch.argmax(pred, dim = 1).detach())
                inpt = torch.stack(cartoonx)
        
        if mode == "eval":
            torch.set_grad_enabled(False)


        
        x=self.set_layer1(inpt)
        x=self.set_layer2(x)
        
        gl = self.gl_layer1(self.vp1(x))
        # gl = self.gl_layer1(self.frame_max(x)[0]) # x's shape is (bs,64,32,32)
        
        #gl = self.gl_layer2(gl) # just normal convolutional network
        gl = self.gl_pooling2(gl)[0] 
        gl = self.gl_layer2(gl) 
        #gl2 = self.gl_pooling2(gl)
        #gl = self.gl_pooling(gl) # the shape is (40,64,16,16)
        
        #print("\n")
        #print("test size:")    
        #print(len(gl), "," , len(gl2[1]))
        #print("\n")

        x = self.set_layer3(x) # 40,64,16,16
        x = self.set_layer4(x)
        
        # gl = self.gl_layer3(gl + self.frame_max(x)[0])
        gl = gl + self.vp2(x)
        gl = self.gl_pooling2(gl)[0]
        gl = self.gl_layer5(gl)
        gl = self.gl_layer3(gl)


        x = self.set_layer5(x)
        x = self.set_layer6(x)
        x=self.vp3(x)
        # x = self.frame_max(x)[0]
        
        gl = gl + x

        feature=[]
        n, c, h, w = gl.size()
        for num_bin in self.bin_num:
            z = x.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
            z = gl.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
        feature=self.final(feature)

        torch.set_grad_enabled(True)
        # a=self.compress(feature.permute(1,2,0)).squeeze()
        target_a = target_b = target
        if mixup == True and target != None:
             x, target_a, target_b, lam = self.mixup_data4(x, target, lam=lam)


        return feature, target_a, target_b
    
    
    def uniform_mixup(self, x1, x2, lam):
        '''
        point cloud uniform sampling: sampling lambda*npoints from x1, and
        sampling (1-lambda)*npoints from x2, then concatenate them to get
        the mixed_x
        Args: 
            x1: (batch_size, feature_dimentionality, num_points)
            x2: (batch_size, feature_dimentionality, num_points)
            lam: uniformly sampled from U[0,1]
        Returns:
            mixed_x: (batch_size, feature_dimentionality, num_points)
        '''
        device = x1.device
        bs, fd, npoints = x1.shape
        # x1 = x1.permute(0, 2, 1)
        # x2 = x2.permute(0, 2, 1)
        
        npoints_x1 = int(lam * npoints)
        npoints_x2 = npoints - npoints_x1
        
        # rand_id1 = torch.randperm(npoints).to(device)
        # rand_id2 = torch.randperm(npoints).to(device)

        new_x2 = x2[:, :, :npoints_x2]
        new_x1 = x1[:, :, :npoints_x1]
        
        mixed_x = torch.cat((new_x1, new_x2), dim=-1)
        # mixed_x = mixed_x.permute(0, 2, 1)
        
        return mixed_x
        
    def mixup_data4(self, x, y, lam):

        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        if torch.cuda.is_available():
            index = index.cuda()
        mixed_x = self.uniform_mixup(x, x[index], lam)#lam * x + (1 - lam) * x[index,:]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

if __name__=='__main__':
    '''
    5 way
    1 shot
    3 query
    '''

    inpt=torch.randn((20,3,1024))
    network=ViewNet()
    out=network(inpt)
    a=1