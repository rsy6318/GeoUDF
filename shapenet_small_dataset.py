import numpy as np
import os
import torch
import torch.utils.data as data

class mesh_pc_dataset(data.Dataset):
    def __init__(self,data_path,mode='train',min_num_point=3000,max_num_point=48000,num_sample=2048,rot=False):
        super(mesh_pc_dataset, self).__init__()
        self.data_path=data_path
        self.mode=mode
        assert mode in ['train','test','val']
        self.min_num_point=min_num_point
        self.max_num_point=max_num_point
        self.num_sample=num_sample
        #self.split=np.load(os.path.join(data_path,'split.npz'))[mode]
        self.rot=rot

        self.data_path=[]

        cat_list=os.listdir(data_path)
        for cat in cat_list:
            with open(os.path.join(data_path,cat,mode+'.lst')) as f:
                models_c=f.read().split('\n')
            if '' in models_c:
                models_c.remove('')
            self.data_path=self.data_path+[os.path.join(data_path,cat,model) for model in models_c if os.path.exists(os.path.join(data_path,cat,model))]
        

    def rotate_point_cloud_and_gt(self):
        #angles = np.random.uniform(0, 1) * np.pi * 2

        angles = np.random.choice([0,np.pi/2,np.pi,np.pi/2*3],1)

        Ry = np.array([[np.cos(angles), 0,-np.sin(angles)],
                       [0, 1, 0],
                       [np.sin(angles), 0,np.cos(angles)]
                       ])

        return Ry

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):

        data_path=self.data_path[idx]
        lr_pc=np.load(os.path.join(data_path,'points.npy'))
        hr_pc=np.load(os.path.join(data_path,'%d.npy'%self.max_num_point))
        npz_data=np.load(os.path.join(data_path,'sample_gauss.npz'))

        index=np.random.permutation(lr_pc.shape[0])[0:self.min_num_point]
        lr_pc=lr_pc[index,:]

        if hr_pc.shape[0]<self.max_num_point:
            add_num=self.max_num_point-hr_pc.shape[0]
            add_index=np.random.permutation(hr_pc.shape[0])[:add_num]
            add_points=hr_pc[add_index,:]
            hr_pc=np.concatenate((hr_pc,add_points),axis=0)
        if hr_pc.shape[0]>self.max_num_point:
            index=np.random.permutation(hr_pc.shape[0])[:self.max_num_point]
            hr_pc=hr_pc[index,:]


        sample_pc=npz_data['points']
        df=npz_data['df']
        closest_points=npz_data['closest_points']

        idx1 = (np.isnan(closest_points[:, 0]) == False) & (np.isnan(closest_points[:, 1]) == False) & (np.isnan(closest_points[:, 2]) == False) & (
                np.isinf(closest_points[:, 0]) == False) & (np.isinf(closest_points[:, 1]) == False) & (np.isinf(closest_points[:, 2]) == False)

        sample_pc = sample_pc[idx1, :]
        df = df[idx1]
        closest_points = closest_points[idx1, :]

        idx2=np.random.choice(np.arange(closest_points.shape[0]),self.num_sample)

        sample_pc = sample_pc[idx2, :]
        df = df[idx2]
        closest_points = closest_points[idx2, :]

        '''import open3d as o3d

        pointcloud=o3d.geometry.PointCloud()
        pointcloud.points=o3d.utility.Vector3dVector(lr_pc)
        pointcloud.paint_uniform_color([0,0,1])

        pointcloud2=o3d.geometry.PointCloud()
        pointcloud2.points=o3d.utility.Vector3dVector(hr_pc)
        pointcloud2.paint_uniform_color([0,1,1])

        o3d.visualization.draw_geometries([pointcloud2,pointcloud])'''

        if self.mode=='train' and self.rot:
            Rot=self.rotate_point_cloud_and_gt()
            lr_pc=np.dot(lr_pc,Rot)
            hr_pc=np.dot(hr_pc,Rot)
            sample_pc=np.dot(sample_pc,Rot)
            closest_points=np.dot(closest_points,Rot)

        return {
            'sparse_pc': torch.from_numpy(lr_pc).transpose(0, 1).float(),       #(3,N)
            'dense_pc': torch.from_numpy(hr_pc).transpose(0, 1).float(),        #(3,RN)
            'sample_pc': torch.from_numpy(sample_pc).transpose(0, 1).float(),   #(3,M)
            'df': torch.from_numpy(df).float(),  # (M,)
            'closest_points':torch.from_numpy(closest_points).transpose(0,1).float(),
            'file_name': data_path
        }


if __name__=='__main__':
    dataset=mesh_pc_dataset('/home/siyu_ren/shapenet_xu2_points')
    out_dict=dataset[0]

    np.savetxt('sparse.xyz',out_dict['sparse_pc'].numpy().T)
    np.savetxt('dense.xyz',out_dict['dense_pc'].numpy().T)