import numpy as np
import os
import trimesh
from tqdm import tqdm
from multiprocessing import Pool
from glob import glob

base_path='your downloaded data'
save_path='data'

cat_list=os.listdir(base_path)

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    Suggested by https://github.com/mikedh/trimesh/issues/507
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh



base_path_1=[]
save_path_1=[]


for cat in cat_list:
    mesh_list=os.listdir(os.path.join(base_path,cat))
    mesh_list.sort()
    for mesh_name in (mesh_list):
        base_path_1.append(os.path.join(base_path,cat,mesh_name))
        save_path_1.append(os.path.join(save_path,cat,mesh_name))


def scalled_off(idx):
    input = trimesh.load(os.path.join(base_path_1[idx], 'isosurf.obj'))
    mesh = as_mesh(input)
    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
    mesh.apply_translation(-centers)
    mesh.apply_scale(1 / total_size)

    if not os.path.exists(save_path_1[idx]):
        os.makedirs(save_path_1[idx])
    mesh.export(os.path.join(save_path_1[idx], 'scaled_model.off'))
    print(os.path.join(save_path_1[idx], 'scaled_model.off'))

idx_list=list(range(len(save_path_1)))

def multiprocess(func):
    p = Pool(16)
    p.map(func, idx_list)
    p.close()
    p.join()
if __name__=='__main__':
    multiprocess(scalled_off)