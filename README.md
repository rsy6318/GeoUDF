# GeoUDF  
[arxiv](https://arxiv.org/abs/2211.16762)  

<img src='demo/input.gif' width=24%> 
<img src='demo/pu.gif' width=24%> 
<img src='demo/result.gif' width=24%>
<img src='demo/gt.gif' width=24%>
<center><div>Left to Right: Input, Upsampled, Ours, and GT.</div> </center>
  
## Requirement
```
pytorch             #1.10.0+cu111
pytorch3d           #0.6.2
open3d
trimesh
point-cloud-utils
```
### Install **pointnet2_ops**
```
cd pointnet2_ops_lib   
python setup.py --install
```
## Data Preparation
Download the data from [Google Drive](https://drive.google.com/drive/folders/1QGhDW335L7ra31uw5U-0V7hB-viA0JXr) (These shapes are processed by [DISN](https://github.com/Xharlie/DISN), remove the interior and non-manifold structures.)   
Then use the codes in [**scripts**](scripts) to get the dataset.
```
python scale_off.py   #data_path need to be changed   
python pds_pc.py   
python random_pc.py   
python sample.py
```

## Training
```
python main_pu.py  --data_path=your_data_path
python main_rec.py --data_path=your_data_path
```
Note: You need to change the data path.
## Evaluation
We provide the pretrained model in [**log_reconstruction_100.00_1.000_0.100**](log_reconstruction_100.00_1.000_0.100) and demo data in [**test_data**](test_data), and you can use them to generate meshes
```
python eval_rec.py --res=128 --input='test_data/shapenet.ply' --output='test_data/shapenet_mesh.ply'   
python eval_rec.py --res=128 --input='test_data/MGN.ply' --output='test_data/MGN_mesh.ply'   
python eval_rec.py --res=192 --input='test_data/scene.ply' --output='test_data/scene_mesh.ply' --scale=True
```

## Citation  
```bibtex
@misc{ren2022geoudf,
      title={GeoUDF: Surface Reconstruction from 3D Point Clouds via Geometry-guided Distance Representation}, 
      author={Siyu Ren and Junhui Hou and Xiaodong Chen and Ying He and Wenping Wang},
      year={2022},
      eprint={2211.16762},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
