import os
import sys
import numpy as np
import pymeshlab
from multiprocessing import Pool
#print(pymeshlab.__version__)
#assert False
num_point=48000

base_path='data'
save_path='data'

cat_list=os.listdir(base_path) #['03001627']

path_list=[]
save_path_list=[]

for cat in cat_list[:13]:
    mesh_list=os.listdir(os.path.join(base_path,cat))
    for mesh_name in mesh_list:
        path_list.append(os.path.join(base_path,cat,mesh_name))
        save_path_list.append(os.path.join(save_path,cat,mesh_name))

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


idx_list=list(range(len(path_list)))

def multiprocess(func):
    p = Pool(80)
    p.map(func, idx_list)
    p.close()
    p.join()

def pds(idx):
    if 1:
    #if not os.path.exists(os.path.join(save_path_list[idx],'%d.npy'%num_point)):
        #with HiddenPrints():
        print(os.path.join(save_path_list[idx], '%d.npy' % num_point),' is preparing.')

        ms_set = pymeshlab.MeshSet()
        ms_set.load_new_mesh(os.path.join(os.path.join(path_list[idx], 'scaled_model.off')))
        ms_set.generate_sampling_poisson_disk(samplenum=int(num_point / (1 - 0.006)), exactnumflag=True)
        pc = np.array(ms_set.current_mesh().vertex_matrix())
        #index = np.random.choice(np.arange(pc.shape[0]), num_point)
        #pc = pc[index, :]
        if not os.path.exists(save_path_list[idx]):
            os.makedirs(save_path_list[idx])
        np.save(os.path.join(save_path_list[idx],'%d.npy'%num_point),pc.astype(np.float32))
        print(os.path.join(save_path_list[idx],'%d.npy'%num_point),' is prepared.')
        del pc,ms_set #,index 
    else:
        print('skip ',os.path.join(save_path_list[idx],'%d.npy'%num_point))

if __name__=='__main__':
    multiprocess(pds)

