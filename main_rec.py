import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapenet_small_dataset import mesh_pc_dataset
import numpy as np
import argparse
from model import PUGeo,UDF
from glob import glob
from datetime import datetime
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch3d.loss import chamfer_distance
from tensorboardX import SummaryWriter 



def log_string(out_str):
    global LOG_FOUT
    LOG_FOUT.write(out_str)
    LOG_FOUT.flush()

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='/gpfs1/scratch/siyuren2/shapenet_xu2_points/', help='train or test')
    parser.add_argument('--phase', default='train', help='train or test')
    parser.add_argument('--gpu', default='0', help='which gpu to use')
     
    parser.add_argument('--model', default='model_pugeo', help='Model for upsampling')
    parser.add_argument('--min_num_point', type=int, default=3000, help='Point Number')
    parser.add_argument('--max_num_point', type=int, default=48000, help='Point Number')

    # for phase train
    #parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training')
    parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--min_lr', type=float, default=0.00001)
    
    parser.add_argument('--num_sample', type=int, default=2048)
    parser.add_argument('--udf_K', type=int, default=10)

    # for phase test
    parser.add_argument('--pretrained', default='', help='Model stored')
    parser.add_argument('--resume', type=bool, default=False, help='Number of points covered by patch')
    
    parser.add_argument('--lambda1',default=100, type=float,)
    parser.add_argument('--lambda2',default=1, type=float,)
    parser.add_argument('--lambda3',default=0.1, type=float,)

    parser.add_argument('--eval_sample',default=10000, type=int,)



    arg = parser.parse_args()
    
    arg.up_ratio=int(arg.max_num_point//arg.min_num_point)

    arg.log_dir='log_reconstruction_%0.2f_%0.3f_%0.3f'%(arg.lambda1,arg.lambda2,arg.lambda3)
    try:
        os.mkdir(arg.log_dir)
    except:
        pass

    writer = SummaryWriter(arg.log_dir)

    global LOG_FOUT
    LOG_FOUT = open(os.path.join(arg.log_dir, 'log.txt'), 'w')
    LOG_FOUT.write(str(datetime.now()) + '\n')
    LOG_FOUT.write(os.path.abspath(__file__) + '\n')
    LOG_FOUT.write(str(arg) + '\n')

    dataset = mesh_pc_dataset(arg.data_path,mode='train',min_num_point=arg.min_num_point,max_num_point=arg.max_num_point,num_sample=arg.num_sample)
    dataset_test = mesh_pc_dataset(arg.data_path,mode='test',min_num_point=arg.min_num_point,max_num_point=arg.max_num_point,num_sample=arg.eval_sample)

    dataloader=torch.utils.data.DataLoader(dataset,batch_size=arg.batch_size,shuffle=True,drop_last=True,num_workers=16)
    dataloader_test=torch.utils.data.DataLoader(dataset_test,batch_size=1,shuffle=False,drop_last=False,num_workers=16)

    pu_model=PUGeo(knn=20)
    pu_model = nn.DataParallel(pu_model)
    
    pumodel=pu_model.cuda()

    pumodel.load_state_dict(torch.load('log_x16/model_best.t7'))

    udf_model=UDF()
    udf_model=nn.DataParallel(udf_model)

    udf_model=udf_model.cuda()

    current_lr=arg.learning_rate

    optimizer=torch.optim.Adam(list(pu_model.parameters())+list(udf_model.parameters()),lr=current_lr)

    #scheduler = CosineAnnealingLR(optimizer, arg.max_epoch, eta_min=arg.min_lr)


    if arg.resume:
        pumodel.load_state_dict(torch.load(os.path.join(arg.log_dir,'pu_model_last.t7')))
        udf_model.load_state_dict(torch.load(os.path.join(arg.log_dir,'udf_model_last.t7')))

    loss_sum_dense_l2_test_best=1e10

    global_step=0

    for epoch in range(arg.max_epoch):
        #scheduler.step()
        
        loss_sum_all=[]
        loss_sum_dense_cd = []
        loss_sum_dense_normal = []
        loss_sum_sparse_normal = []
        
        loss_sum_l2_dist=[]
        loss_sum_udf_grad=[]
        pu_model.train()
        udf_model.train()
        for data in tqdm(dataloader,desc='epoch %d train'%epoch):
            global_step=global_step+1
            input_sparse_xyz=data['sparse_pc']
            gt_dense_xyz=data['dense_pc']
            sample_points=data['sample_pc']
            closest_points=data['closest_points']       #(B,3,M)
            gt_udf=data['df']
            
            input_sparse_xyz=input_sparse_xyz.cuda()
            gt_dense_xyz=gt_dense_xyz.cuda()
            sample_points=sample_points.cuda()
            sample_points.requires_grad=True
            closest_points=closest_points.cuda()
            gt_udf=gt_udf.cuda()

            batch_size,_,num_point=input_sparse_xyz.size()

            optimizer.zero_grad()

            pu_model.train()
            udf_model.train()
            output_dict=pu_model(input_sparse_xyz)
            
            dense_xyz=output_dict['dense_xyz']
            dense_normal=output_dict['dense_normal']

            output_dict['dense_xyz']=dense_xyz.detach()
            output_dict['dense_normal']=dense_normal.detach()
            

            pred_udf,pred_udf_grad=udf_model(output_dict,sample_points)
            
            gt_udf_grad=F.normalize(sample_points-closest_points,dim=1).transpose(1,2)  #(B,M,3)

            #pred_udf_grad=diff(pred_udf,sample_points.transpose(1,2))   #(B,M,3)
            #print(pred_udf_grad.size())

            cd_loss=chamfer_distance(dense_xyz.reshape(batch_size,-1,3),gt_dense_xyz.transpose(1,2))[0]

            l1_dist=torch.mean(torch.abs(pred_udf-gt_udf))


            udf_grad_loss=torch.mean(1-torch.sum(gt_udf_grad*pred_udf_grad,dim=2))

            loss_all=arg.lambda1*cd_loss+arg.lambda2*l1_dist+arg.lambda3*udf_grad_loss

            loss_all.backward()
            optimizer.step()

            loss_sum_all.append(loss_all.detach().cpu().numpy())
            loss_sum_dense_cd.append(cd_loss.detach().cpu().numpy())
            loss_sum_l2_dist.append(l1_dist.detach().cpu().numpy())
            loss_sum_udf_grad.append(udf_grad_loss.detach().cpu().numpy())
            #break

            if global_step%50==0:
                writer.add_scalar('cd_loss', cd_loss.detach().cpu().numpy().mean(), global_step)
                writer.add_scalar('L1_udf_loss', l1_dist.detach().cpu().numpy().mean(), global_step)
                writer.add_scalar('udf_grad_loss', udf_grad_loss.detach().cpu().numpy().mean(), global_step)
        
        loss_sum_all=np.array(loss_sum_all)
        loss_sum_dense_cd=np.array(loss_sum_dense_cd)
        loss_sum_l2_dist=np.array(loss_sum_l2_dist)
        loss_sum_udf_grad=np.array(loss_sum_udf_grad)
        

        log_string('epoch: %03d total loss: %0.7f, cd: %0.7f, l2 dist: %0.7f, udf_grad: %0.7f\n' % (
                    epoch, loss_sum_all.mean(),  loss_sum_dense_cd.mean(),loss_sum_l2_dist.mean(),loss_sum_udf_grad.mean()
                   ))
        torch.cuda.empty_cache()
        
        loss_sum_dense_l2_test=[]
        
        
        pu_model.eval()
        udf_model.eval()
        if 1:
            for data in tqdm(dataloader_test,desc='epoch %d test'%epoch):

                input_sparse_xyz=data['sparse_pc']
                gt_dense_xyz=data['dense_pc']
                sample_points=data['sample_pc']
                #closest_points=data['closest_points']
                gt_udf=data['df']
                
                input_sparse_xyz=input_sparse_xyz.cuda()
                gt_dense_xyz=gt_dense_xyz.cuda()
                sample_points=sample_points.cuda()
                sample_points.requires_grad=True
                #closest_points=closest_points.cuda()
                gt_udf=gt_udf.cuda()

                batch_size,_,num_point=input_sparse_xyz.size()

                
                pu_model.eval()
                udf_model.eval()
                output_dict=pu_model(input_sparse_xyz)
                
                dense_xyz=output_dict['dense_xyz']
                dense_normal=output_dict['dense_normal']

                pred_udf,_=udf_model(output_dict,sample_points)
                
                cd_loss=chamfer_distance(dense_xyz.reshape(batch_size,-1,3),gt_dense_xyz.transpose(1,2))[0]

                l2_dist=torch.mean(torch.abs(pred_udf-gt_udf))

                loss_all=100*cd_loss+l2_dist

                loss_sum_dense_l2_test.append(l2_dist.detach().cpu().numpy())
            
            
        loss_sum_dense_l2_test = np.asarray(loss_sum_dense_l2_test).mean()
        
        if loss_sum_dense_l2_test_best>loss_sum_dense_l2_test:
            torch.save(pu_model.state_dict(), os.path.join(arg.log_dir, 'pu_model_best.t7'))
            torch.save(udf_model.state_dict(), os.path.join(arg.log_dir, 'udf_model_best.t7'))
            loss_sum_dense_l2_test_best=loss_sum_dense_l2_test
        
        '''if epoch%100==0:
            torch.save(model.state_dict(),os.path.join(arg.log_dir,'model_%d.t7'%epoch))'''

        '''if epoch%100==0 and epoch>200:
            current_lr = current_lr * 0.25
            if current_lr < arg.min_lr:
                current_lr = arg.min_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr'''
        
        torch.save(pu_model.state_dict(), os.path.join(arg.log_dir, 'pu_model_last.t7'))
        torch.save(udf_model.state_dict(), os.path.join(arg.log_dir, 'udf_model_last.t7'))
        torch.cuda.empty_cache()