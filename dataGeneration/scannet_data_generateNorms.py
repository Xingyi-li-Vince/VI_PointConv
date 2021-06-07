import open3d as o3d
import sys
import os

import torch, numpy as np, glob
from sklearn.neighbors import KDTree

from pathlib import Path
from tqdm import tqdm

# paths for the raw data without surface normals
train_files = glob.glob('./ScanNet/train/*.pth') 
val_files = glob.glob('./ScanNet/val/*.pth')
test_files = glob.glob('./ScanNet/test/*.pth')

# paths for the new saved data with surface normals
save_train_path = Path('./ScanNet_withNormal/train')
save_val_path =  Path('./ScanNet_withNormal/val')
save_test_path =  Path('./ScanNet_withNormal/test')

save_train_path.mkdir(exist_ok=True)
for idx in tqdm(range(len(train_files))):
    coord, color, label = torch.load(train_files[idx])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30), fast_normal_computation=False)
    pcd.orient_normals_towards_camera_location(camera_location=([0., 0., 0.]))
    norms = np.asarray(pcd.normals)
    fileName = str(save_train_path)+train_files[idx][len(str(save_train_path))-1:]
    torch.save([coord, color, label, norms], fileName)


save_val_path.mkdir(exist_ok=True)
for idx in tqdm(range(len(val_files))):
    coord, color, label = torch.load(val_files[idx])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30), fast_normal_computation=False)
    pcd.orient_normals_towards_camera_location(camera_location=([0., 0., 0.]))
    norms = np.asarray(pcd.normals)
    fileName = str(save_val_path)+val_files[idx][len(str(save_val_path))-1:]
    torch.save([coord, color, label, norms], fileName)

save_test_path.mkdir(exist_ok=True)
for idx in tqdm(range(len(test_files))):
    coord, color = torch.load(test_files[idx])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30), fast_normal_computation=False)
    pcd.orient_normals_towards_camera_location(camera_location=([0., 0., 0.]))
    norms = np.asarray(pcd.normals)
    fileName = str(save_test_path)+test_files[idx][len(str(save_test_path))-1:]
    torch.save([coord, color, norms], fileName)

