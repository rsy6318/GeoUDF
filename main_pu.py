import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import torch.nn as nn
from shapenet_small_dataset import mesh_pc_dataset
import numpy as np
import argparse
import model as model
#import loss
import logging
from glob import glob
#from pc_util import  normalize_point_cloud, farthest_point_sample, group_points
from datetime import datetime
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch3d.loss import chamfer_distance
from tensorboardX import SummaryWriter 

#print(torch.cuda.is_available())

def log_string(out_str):
    global LOG_FOUT
    LOG_FOUT.write(out_str)
    LOG_FOUT.flush()


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', required=True, help='train or test')
    parser.add_argument('--phase', default='train', help='train or test')
    parser.add_argument('--gpu', default='0', help='which gpu to use')
     
    parser.add_argument('--model', default='model_pugeo', help='Model for upsampling')
    parser.add_argument('--min_num_point', type=int, default=3000, help='min Point Number')
    parser.add_argument('--max_num_point', type=int, default=48000, help='max Point Number')
    
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training')
    parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=0.00001)
    

    parser.add_argument('--pretrained', default='', help='Model stored')
    parser.add_argument('--resume', type=bool, default=False, help='Number of points covered by patch')
    
    arg = parser.parse_args()
    
    arg.up_ratio=int(arg.max_num_point//arg.min_num_point)

    arg.log_dir='log_x%d'%arg.up_ratio
    try:
        os.mkdir(arg.log_dir)
    except:
        pass
    global LOG_FOUT
    LOG_FOUT = open(os.path.join(arg.log_dir, 'log.txt'), 'w')
    LOG_FOUT.write(str(datetime.now()) + '\n')
    LOG_FOUT.write(os.path.abspath(__file__) + '\n')
    LOG_FOUT.write(str(arg) + '\n')

    dataset = mesh_pc_dataset(arg.data_path,mode='train',min_num_point=arg.min_num_point,max_num_point=arg.max_num_point,rot=False)
    dataset_test = mesh_pc_dataset(arg.data_path,mode='val',min_num_point=arg.min_num_point,max_num_point=arg.max_num_point)
    
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=arg.batch_size,shuffle=True,drop_last=True,num_workers=16)
    dataloader_test=torch.utils.data.DataLoader(dataset_test,batch_size=arg.batch_size//10,shuffle=False,drop_last=False,num_workers=16)
    
    model=model.PUGeo(knn=20)
    model = nn.DataParallel(model)
    #model.load_state_dict(torch.load('log_x16_old/model_best.t7'))
    model=model.cuda()

    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    current_lr=arg.learning_rate

    optimizer=torch.optim.Adam(learnable_params,lr=current_lr)

    scheduler = CosineAnnealingLR(optimizer, arg.max_epoch, eta_min=arg.min_lr)

    loss_sum_dense_cd_test_best=1e10

    writer = SummaryWriter(arg.log_dir)
    
    global_step=0

    for epoch in range(arg.max_epoch):
        #scheduler.step()

        loss_sum_all=[]
        loss_sum_dense_cd = []
        loss_sum_dense_normal = []
        loss_sum_sparse_normal = []

        model.train()
        for data in tqdm(dataloader,desc='epoch %d train'%epoch):
            
            global_step=global_step+1

            input_sparse_xyz=data['sparse_pc']
            gt_dense_xyz=data['dense_pc']
            
            input_sparse_xyz = input_sparse_xyz.cuda()
            
            gt_dense_xyz = gt_dense_xyz.cuda()
            batch_size=gt_dense_xyz.size(0)
            optimizer.zero_grad()

            model.train()
            output_dict=model(input_sparse_xyz,poisson=True,up_ratio=arg.up_ratio)
            
            gen_dense_xyz=output_dict['dense_xyz']
            
            loss_dense_cd=chamfer_distance(gen_dense_xyz.reshape(batch_size,-1,3),gt_dense_xyz.transpose(1,2))[0]

            #loss_dense_cd,cd_idx1,cd_idx2=loss.cd_loss(gen_dense_xyz,gt_dense_xyz)
            
            loss_all=100*loss_dense_cd

            loss_all.backward()
            optimizer.step()

            loss_sum_all.append(loss_all.detach().cpu().numpy())
            loss_sum_dense_cd.append(loss_dense_cd.detach().cpu().numpy())
            
            if global_step%10==0:
            
                writer.add_scalar('cd_loss', loss_dense_cd.detach().cpu().numpy().mean(), global_step)


        loss_sum_all=np.array(loss_sum_all)
        loss_sum_dense_cd=np.array(loss_sum_dense_cd)
        

        log_string('epoch: %03d total loss: %0.7f, cd: %0.7f\n' % (
                    epoch, loss_sum_all.mean(),  loss_sum_dense_cd.mean()
                   ))

        writer.add_scalar('Epoch_train_cd_loss', loss_sum_dense_cd.mean(), epoch)
        
        torch.cuda.empty_cache()
        
        loss_sum_dense_cd_test=[]
        
        count_test=0
        model.eval()
        with torch.no_grad():
            for data in tqdm(dataloader_test,desc='epoch %d test'%epoch):

                count_test=count_test+input_sparse_xyz.size(0)

                input_sparse_xyz=data['sparse_pc']
                gt_dense_xyz=data['dense_pc']

                input_sparse_xyz = input_sparse_xyz.cuda()
                
                gt_dense_xyz = gt_dense_xyz.cuda()
               
                batch_size=gt_dense_xyz.size(0)

                output_dict=model(input_sparse_xyz,up_ratio=arg.up_ratio,poisson=True)
                gen_dense_xyz=output_dict['dense_xyz']

                #loss_dense_cd_test,cd_idx1,cd_idx2=loss.cd_loss(gen_dense_xyz,gt_dense_xyz)
                
                loss_dense_cd_test=chamfer_distance(gen_dense_xyz.reshape(batch_size,-1,3),gt_dense_xyz.transpose(1,2))[0]

                loss_sum_dense_cd_test.append(loss_dense_cd_test.detach().cpu().numpy())
            
            

        loss_sum_dense_cd_test = np.asarray(loss_sum_dense_cd_test).mean()

        writer.add_scalar('Epoch_test_cd_loss', loss_sum_dense_cd_test, epoch)
        
        if loss_sum_dense_cd_test_best>loss_sum_dense_cd_test:
            torch.save(model.state_dict(), os.path.join(arg.log_dir, 'model_best.t7'))
            loss_sum_dense_cd_test_best=loss_sum_dense_cd_test
        
        torch.save(model.state_dict(), os.path.join(arg.log_dir, 'model_last.t7'))
        torch.cuda.empty_cache()