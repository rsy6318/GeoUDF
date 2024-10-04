# GeoUDF (ICCV 2023)
## GeoUDF: Surface Reconstruction from 3D Point Clouds via Geometry-guided Distance Representation [[**Project Page**]](https://rsy6318.github.io/GeoUDF.html)  [[**arxiv**]](https://arxiv.org/abs/2211.16762)  

<div class="container">
<div class="row">
<div class="col-12 text-center" id="pipeline">
<img src='demo/input.gif' width=20%> 
<img src='demo/pu.gif' width=20%> 
<img src='demo/result.gif' width=20%>
<img src='demo/gt.gif' width=20%>
<center><div>Left to Right: Input, Upsampled, Ours, and GT.</div> </center>
</div>
</div>
</div>  
      
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
python setup.py install
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
If the input point cloud is dense enough and it does not need to be upsampled, you gan run the following code
```
python eval_rec_dense.py --res=128 --input=<path to input mesh> --output=<path to output mesh>
```

## Citation  
```bibtex
@inproceedings{ren2023geoudf,
title={Geoudf: Surface reconstruction from 3d point clouds via geometry-guided distance representation},
author={Ren, Siyu and Hou, Junhui and Chen, Xiaodong and He, Ying and Wang, Wenping},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
pages={14214--14224},
year={2023}}
```
