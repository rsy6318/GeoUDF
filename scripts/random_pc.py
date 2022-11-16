import os
import sys
import numpy as np
import pymeshlab
from multiprocessing import Pool
import trimesh
import point_cloud_utils as pcu

num_sample = 100000

base_path='data'
save_path='data'

cat_list=os.listdir(base_path) #['03001627']

path_list = []
save_path_list=[]

for cat in cat_list:
    mesh_list = os.listdir(os.path.join(base_path, cat))
    for mesh_name in mesh_list:
        path_list.append(os.path.join(base_path, cat, mesh_name))
        save_path_list.append(os.path.join(save_path, cat, mesh_name))


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


idx_list = list(range(len(path_list)))


def multiprocess(func):
    p = Pool(20)
    p.map(func, idx_list)
    p.close()
    p.join()

ratio_list=[0.1,0.4,0.5]
std_list=[0.05,0.02,0.003]

def sample(idx):

    mesh=trimesh.load(os.path.join(path_list[idx],'scaled_model.off'))
    points=mesh.sample(int(num_sample))
            
    if not os.path.exists(save_path_list[idx]):
        os.makedirs(os.path.join(save_path_list[idx]))
    np.save(os.path.join(save_path_list[idx],'points.npy'), points.astype(np.float32))
        
    print(os.path.join(save_path_list[idx],'points.npy'))


if __name__ == '__main__':
    multiprocess(sample)

