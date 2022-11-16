# GeoUDF
## Requirement
```
pytorch
pytorch3d==0.6.2
open3d
trimesh
point-cloud-utils
```
## Data Preparation
Download the data from [Google Drive](https://drive.google.com/drive/folders/1QGhDW335L7ra31uw5U-0V7hB-viA0JXr) (These shapes are processed by [DISN](https://github.com/Xharlie/DISN), remove the interior and non-manifold structures.)   
Then use the codes in **scripts** to get the dataset.
```
python scale_off.py   
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
We provide the pretrained model in **log_reconstruction_100.00_1.000_0.100** and demo data in **test_data**, and you can use them to generate meshes
```
python eval_rec.py --res=128 --input='test_data/shapenet.ply' --output='test_data/shapenet_mesh.ply'   
python eval_rec.py --res=128 --input='test_data/MGN.ply' --output='test_data/MGN.ply'   
python eval_rec.py --res=192 --input='test_data/scene.ply' --output='test_data/scene.ply' 
```