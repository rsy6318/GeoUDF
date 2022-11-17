from cmath import isnan
import os
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.autograd import Variable, grad
import torch.nn.functional as F
#import poisson_disc as pd
import pointnet2_ops.pointnet2_utils as utils
import pytorch3d.ops
import pytorch3d
import torch.autograd as ag
  

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # (B,C,N,K)

    return feature



class PUGeo(nn.Module):
    def __init__(self, knn=20, fd=128, train_up_ratio=16):
        super(PUGeo, self).__init__()
        self.knn = knn
        
        self.dgcnn_conv1 = nn.Sequential(nn.Conv2d(6, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv2 = nn.Sequential(nn.Conv2d(fd, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv3 = nn.Sequential(nn.Conv1d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm1d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv4 = nn.Sequential(nn.Conv2d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv5 = nn.Sequential(nn.Conv1d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm1d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv6 = nn.Sequential(nn.Conv2d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv7 = nn.Sequential(nn.Conv1d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm1d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))

        self.uv_2order_coefficient_conv = nn.Sequential(
            nn.Conv1d(fd * 9 , 256, kernel_size=1),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 128, kernel_size=1),nn.BatchNorm1d(128),nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, 64, kernel_size=1),nn.BatchNorm1d(64),nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 6*3, kernel_size=1))
        
        for m in self.uv_2order_coefficient_conv.modules():
            if isinstance(m,(torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Linear)):
                #torch.nn.init.xavier_uniform_(m.weight)
                m.weight.data=m.weight.data*1e-5
                #print(m.weight.data)

        
        grid_size=train_up_ratio

        u=torch.from_numpy(np.arange(0,grid_size,dtype = float).reshape(grid_size,1,1)).repeat(1,grid_size,1)/(grid_size-1)
        v=torch.from_numpy(np.arange(0,grid_size,dtype = float).reshape(1,grid_size,1)).repeat(grid_size,1,1)/(grid_size-1)
        uv=torch.cat((u,v,torch.zeros_like(u)),dim=2).reshape(-1,3)         #(grid_size*grid_size,3)
        
        uv=uv.unsqueeze(0).repeat(grid_size*grid_size,1,1)  #(grid_size*grid_size,grid_size*grid_size,3)  ,  regard as (B,N,3)

        first_p_index=torch.arange(0,uv.size(1)).unsqueeze(-1).unsqueeze(-1).repeat(1,1,3)

        first_uv=torch.gather(uv,dim=1,index=first_p_index)

        grid=torch.cat((first_uv,uv),dim=1).cuda().float()   #(B,N+1,3)

        index= utils.furthest_point_sample(grid.contiguous(),train_up_ratio) 
        uv=utils.gather_operation(grid.transpose(1,2).contiguous(),index).transpose(1,2)[:,:,0:2]   #(grid_size*grid_size,up_ratio,2)
        
        self.uv_set=(0.1*(uv*2-1)).cpu()




    def forward(self, x, up_ratio=16, poisson=True ):
        # x:(B,3,N)
        
        batch_size = x.size(0)
        num_point = x.size(2)
        
        edge_feature = get_graph_feature(x, k=self.knn)     #(B,6,N,20)
        out1 = self.dgcnn_conv1(edge_feature)               #(B,128,N,20)    
        out2 = self.dgcnn_conv2(out1)                       #(B,128,N,20)  
        net_max_1 = out2.max(dim=-1, keepdim=False)[0]      #(B,128,N)
        net_mean_1 = out2.mean(dim=-1, keepdim=False)       #(B,128,N)

        out3 = self.dgcnn_conv3(torch.cat((net_max_1, net_mean_1), 1))      #(B,128,N)

        edge_feature = get_graph_feature(out3, k=self.knn)      #(B,256,N,20)
        out4 = self.dgcnn_conv4(edge_feature)                   #(B,128,N,20)

        net_max_2 = out4.max(dim=-1, keepdim=False)[0]          #(B,128,N)
        net_mean_2 = out4.mean(dim=-1, keepdim=False)           #(B,128,N)

        out5 = self.dgcnn_conv5(torch.cat((net_max_2, net_mean_2), 1))  #(B,128,N)

        edge_feature = get_graph_feature(out5,k=self.knn)       #(B,256,N,20)
        out6 = self.dgcnn_conv6(edge_feature)                   #(B,128,N,20)

        net_max_3 = out6.max(dim=-1, keepdim=False)[0]          #(B,128,N)
        net_mean_3 = out6.mean(dim=-1, keepdim=False)           #(B,128,N)

        out7 = self.dgcnn_conv7(torch.cat((net_max_3, net_mean_3), dim=1))

        concat = torch.cat((net_max_1,      # 128
                            net_mean_1,     # 128
                            out3,           # 128
                            net_max_2,      # 128
                            net_mean_2,     # 128
                            out5,           # 128
                            net_max_3,      # 128
                            net_mean_3,     # 128
                            out7,           # 128
                            ), dim=1)  # (B,C,N)

        sel_uv_index=torch.randint(0,int(up_ratio*up_ratio),size=(int(batch_size*num_point),1,1)).repeat(1,self.uv_set.size(1),2).to(concat)

        uv=torch.gather(self.uv_set.to(concat),dim=0,index=sel_uv_index.type(torch.int64)).reshape(batch_size,num_point,-1,2).to(concat)

        u=uv[:,:,:,0:1]
        v=uv[:,:,:,1:]          #(B,N,U,1)


        # [1, u, v , u*u, u*v, v*v]     
        uv_vector=torch.concat((torch.ones_like(u),u,v,u*u,u*v,v*v),dim=-1)        #(B,N,U,6)  

        # [0, 1, 0, 2*u, v, 0]      grad u
        uv_vector_grad_u=torch.concat((torch.zeros_like(u),torch.ones_like(u),torch.zeros_like(u),2*u,v,torch.zeros_like(u)),dim=-1)

        # [0, 0, 1, 0, u, 2*v]      grad v
        uv_vector_grad_v=torch.concat((torch.zeros_like(u),torch.zeros_like(u),torch.ones_like(u),torch.zeros_like(u),u,2*v),dim=-1)

        # [0, 0, 0, 2, 0, 0]        grad uu
        #uv_vector_grad_uu=torch.concat((torch.zeros_like(u),torch.zeros_like(u),torch.zeros_like(u),2*torch.ones_like(u),torch.zeros_like(u),torch.zeros_like(u)),dim=-1)
        # [0, 0, 0, 0, 1, 0]        grad uv
        #uv_vector_grad_uv=torch.concat((torch.zeros_like(u),torch.zeros_like(u),torch.zeros_like(u),torch.zeros_like(u),torch.ones_like(u),torch.zeros_like(u)),dim=-1)
        # [0,0,0,0,0,2]
        #uv_vector_grad_vv=torch.concat((torch.zeros_like(u),torch.zeros_like(u),torch.zeros_like(u),torch.zeros_like(u),torch.zeros_like(u),2*torch.ones_like(u)),dim=-1)

        coefficient=self.uv_2order_coefficient_conv(concat).transpose(2,1).reshape(batch_size,num_point,6,3)        #(B,6*3,N)->(B,N,6*3)->(B,N,6,3)

        xyz_offset=torch.matmul(uv_vector,coefficient)          #(B,N,U,6)@(B,N,6,3) -> (B,N,U,3)

        xyz=x.transpose(2,1).unsqueeze(2)+xyz_offset            #(B,N,U,3)

        xyz_grad_u=torch.matmul(uv_vector_grad_u,coefficient)   #(B,N,U,3)
        xyz_grad_v=torch.matmul(uv_vector_grad_v,coefficient)


        normal=torch.cross(xyz_grad_u,xyz_grad_v)
        normal=F.normalize(normal,dim=-1)                       #(B,N,U,3)

        return {'dense_xyz':xyz.reshape(batch_size,-1,3),                #(B,N,U,3)
                'dense_normal':normal.reshape(batch_size,-1,3),          #(B,N,U,3)            #'sparse_normal':normal_sparse,
                }
        
class UDF(nn.Module):
    def __init__(self, K=10):
        super(UDF, self).__init__()
        
        self.attention_net=nn.Sequential(   nn.Conv2d(6+1+128+128,256,1),   nn.BatchNorm2d(256),    nn.LeakyReLU(negative_slope=0.2),
                                            nn.Conv2d(256,128,1),           nn.BatchNorm2d(128),    nn.LeakyReLU(negative_slope=0.2),
                                            nn.Conv2d(128,32,1),            nn.BatchNorm2d(32),     nn.LeakyReLU(negative_slope=0.2),
                                            nn.Conv2d(32,1,1))

        self.grad_attention_net=nn.Sequential(  nn.Conv2d(6+1+128+128,256,1),   nn.BatchNorm2d(256),    nn.LeakyReLU(negative_slope=0.2),
                                                nn.Conv2d(256,128,1),           nn.BatchNorm2d(128),    nn.LeakyReLU(negative_slope=0.2),
                                                nn.Conv2d(128,32,1),            nn.BatchNorm2d(32),     nn.LeakyReLU(negative_slope=0.2),
                                                nn.Conv2d(32,1,1))
        self.K=K

        self.patch_feature_net=nn.Sequential(   nn.Conv2d(6,64,1),      nn.BatchNorm2d(64),     nn.LeakyReLU(negative_slope=0.2),
                                                nn.Conv2d(64,128,1),    nn.BatchNorm2d(128),    nn.LeakyReLU(negative_slope=0.2),
                                                nn.Conv2d(128,128,1),   nn.BatchNorm2d(128),    nn.LeakyReLU(negative_slope=0.2),
                                                nn.Conv2d(128,128,1), )

    def forward(self,input_dict,query):
        '''
        dense_pc:               (B,NU,3)
        dense_normal:           (B,NU,3)
        query:                  (B,3,M)
        '''
        
        dense_pc=input_dict['dense_xyz']            #(B,NU,3)
        dense_normal=input_dict['dense_normal']     #(B,NU,3)
        


        B=dense_pc.size(0)
        #N=sparse_pc.size(1)
        M=query.size(2)

        query=query.transpose(1,2)                      #(B,M,3)


        _,idx,query_knn_pc=pytorch3d.ops.knn_points(query,dense_pc,K=self.K,return_nn=True,return_sorted=False)             #(B,M,K)    (B,M,K,3)
        query_knn_normal=pytorch3d.ops.knn_gather(dense_normal,idx=idx)                                                     #(B,M,K,3)

        query_knn_pc_local=query.unsqueeze(2)-query_knn_pc      #(B,M,K,3)
        signed_dist=torch.sum(query_knn_pc_local*query_knn_normal,dim=3,keepdim=True)    #(B,M,K,1)

        dist=torch.abs(signed_dist) #(B,M,K,1)

        oriented_normal=torch.sgn(signed_dist)*query_knn_normal         #(B,M,K,3)

        concat_vector1=torch.concat((query_knn_pc_local,oriented_normal),dim=3).permute(0,3,1,2)     #(B,M,K,6)->(B,6,M,K)

        feature=self.patch_feature_net(concat_vector1)          #(B,128,M,K)

        patch_feature=torch.max(feature,dim=3,keepdim=True)[0]     #(B,128,M,1)

        concat_vector2=torch.concat((concat_vector1,dist.permute(0,3,1,2),feature,patch_feature.repeat(1,1,1,self.K)),dim=1)   #(B,6+1+128+128,M,K)

        weight1=self.attention_net(concat_vector2).squeeze(1)      #(B,1,M,K)->(B,M,K)
        weight1=F.softmax(weight1,dim=2)

        weight2=self.grad_attention_net(concat_vector2)  #(B,1,M,K)
        weight2=F.softmax(weight2,dim=3)

        udf=torch.sum(weight1*dist.squeeze(3),dim=2)    #(B,M)

        udf_grad=torch.sum(weight2.permute(0,2,3,1)*oriented_normal,dim=2)  #(B,M,3)
        udf_grad=F.normalize(udf_grad,dim=2)    #(B,M,3)


        return udf,udf_grad
        
