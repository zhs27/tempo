import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======= backbone ============
from model.backbone.DGCNN import DGCNN_fs
from model.backbone.multiview import mutiview_net
from model.backbone.Gaitset_net import Gateset_net
from model.backbone.mymodel_moreview import ViewNet
from model.backbone.model_imginpt import ViewNetimginpt
from model.backbone.mymodel_pointview import pointview
from model.backbone.PointTransformer import PointTransformerCls
from model.backbone.newmodel_ViewnetQ import ViewNetpt
# =============================


#======== fs algorithm =========
from model.fs_module.protonet import protonet
from model.fs_module.cia import CIA 
from model.fs_module.trip import trip
from model.fs_module.pointview_trip import pointview_trip
from model.fs_module.contrastive_loss_bin import Trip_CIA
from model.fs_module.MetaOp import Class_head_MetaOpt
from model.fs_module.RelationNet import RelationNet

#===============================

class fs_network(nn.Module):
    def __init__(self,k_way,n_shot,query,backbone='ViewNet',fs='Trip_CIA'):
        super().__init__()
        self.k=k_way
        self.n=n_shot
        self.query=query
        self.s_label=torch.arange(k_way).repeat_interleave(n_shot)
        self.q_label=torch.arange(k_way).repeat_interleave(query)

        self.s_label=torch.arange(k_way).repeat_interleave(n_shot)
        self.q_label=torch.arange(k_way).repeat_interleave(query)
        self.label = torch.cat((self.s_label, self.q_label))
        
        self.backbone=self.get_backbone(backbone)
        self.fs_head=self.get_fs_head(fs)
    


    def get_backbone(self,backbone):
        if backbone=='dgcnn':
            print("DGCNN is loaded")
            return DGCNN_fs()

        elif backbone=='mv':
            print("multiview is loaded")
            return mutiview_net()
        
        elif backbone=='gaitset':
            print("gaitset is loaded")
            return Gateset_net()
        
        
        elif backbone=='ViewNet':
            print('ViewNet is loaded')
            return ViewNet()
        
        elif backbone=='pointview':
            print('pointview is loaded')
            return pointview()
        
        
        elif backbone=='Point_Trans':
            print('point transformer is loaded')
            return PointTransformerCls()
        
        elif backbone=='ViewNetimg':
            print("ViewNetimg is loaded")
            return ViewNetimginpt()
        
        elif backbone=='ViewNetpt':
            print("ViewNetpt is loaded")
            return ViewNetpt()
        
        else:
            raise ValueError('Illegal Backbone')



    
    def get_fs_head(self,fs):
        if fs=='protonet':
            print("protonet is loaded")
            return protonet(self.k,self.n,self.query)

        elif fs=='cia':
            print("CIA is loaded")
            return CIA(k_way=self.k,n_shot=self.n,query=self.query)
        
        elif fs=='trip':
            print("trip is loaded")
            return trip(k_way=self.k,n_shot=self.n,query=self.query)
    
        elif fs=='pv_trip':
            print('point view trip is loaded')
            return pointview_trip(k_way=self.k,n_shot=self.n,query=self.query)
        

        elif fs=='Trip_CIA':
            print('Trip_CIA is loaded')
            return Trip_CIA(k_way=self.k,n_shot=self.n,query=self.query)

        elif fs=='MetaOp':
            print('MetaOp is loaded')
            return Class_head_MetaOpt(way=self.k,shot=self.n,query=self.query)
        
        elif fs=='Relation':
            print('RelationNet is loaded')
            return RelationNet(k_way=self.k,n_shot=self.n,query=self.query)

        else:
            raise ValueError('Illegal fs_head')
             
     
    
    def forward(self,x,model = None,xtocartoonx = None, mode = "train", mixup=False, target_a = None, target_b = None, lam = 0.4):
        '''
        If backbone is the gait related network
        the embeding shape is (bin,sample_num,feat_dim), like (62,20,256)
        '''

            
        if model != None:
            embeding = self.backbone(x,  xtocartoonx,model, mode)
        else:
            embeding = self.backbone(x)

        pred,loss=self.fs_head(embeding,[self.s_label,self.q_label])

        if mixup == True and target_a != None and target_b != None:
            qry_target_a, qry_target_b = target_a[self.k*self.n:], target_b[self.k*self.n:]
            qry_target_a = qry_target_a.to(device='cuda')
            qry_target_b = qry_target_b.to(device='cuda')
            loss_0, loss_1 = F.cross_entropy(pred, qry_target_a), F.cross_entropy(pred, qry_target_b)
            # loss = lam*loss_0 + (1-lam)*loss_1
            loss = loss_0 + loss_1
        
        if torch.isnan(loss):
            save_dict={}
            save_dict['inpt']=x
            torch.save(save_dict,'nan_x')
        
        assert not torch.isnan(loss)

        return pred,loss


if __name__=='__main__':
    fs_net=fs_network(k_way=5,n_shot=1,query=3,backbone='ViewNet',fs='Trip_CIA')
    sample_inpt=torch.randn((20,3,1024))
    pred,loss=fs_net(sample_inpt)
    a=1
    
        
        
        