import open3d as o3d
import sys
import os

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import pickle
from sklearn.neighbors import KDTree
import random
from easydict import EasyDict as edict
import yaml


def compute_weight(train_data, num_class = 20):
    weights = np.array([0.0 for i in range(num_class)])

    num_rooms = len(train_data)
    for i in range(num_rooms):
        _, _, labels,_ = train_data[i]
        #rm invalid labels
        labels = labels[labels >= 0]
        for j in range(num_class):
            weights[j] += np.sum(labels == j)

    ratio = weights / float(sum(weights))
    ce_label_weight = 1 / (np.power(ratio, 1/3))
    return list(ce_label_weight)

def compute_knn(ref_points, query_points, K, dialated_rate = 1):
    num_ref_points = ref_points.shape[0]

    if num_ref_points < K or num_ref_points < dialated_rate * K:
        num_query_points = query_points.shape[0]
        inds = np.random.choice(num_ref_points, (num_query_points, K)).astype(np.int32)

        return inds

    kdt = KDTree(ref_points)
    neighbors_idx = kdt.query(query_points, k = K * dialated_rate, return_distance=False)
    neighbors_idx = neighbors_idx[:, ::dialated_rate]

    return neighbors_idx

class data_loader_ScanNet(object):
    def __init__(self, cfg):
        self.train = []
        self.val = []
        self.cfg = cfg
        
        train_files = glob.glob(self.cfg.train_data_path)
        val_files = glob.glob(self.cfg.val_data_path)

        for x in torch.utils.data.DataLoader(
                train_files,
                collate_fn=lambda x: torch.load(x[0]), num_workers=5):
            self.train.append(x)
        for x in torch.utils.data.DataLoader(
                val_files,
                collate_fn=lambda x: torch.load(x[0]), num_workers=5):
            self.val.append(x)
        print('Training examples:', len(self.train))
        print('Validation examples:', len(self.val))
        
        if self.cfg.USE_WEIGHT:
            weights = compute_weight(self.train)
        else:
            weights = [1.0] * 20
        print("label weights", weights)
        self.cfg.weights = weights
        
    def trainSet(self, tbl):
        locs=[]
        feats=[]
        labels=[]
        norms = []
        nei_forward = []
        nei_propagate = []
        nei_self = []
        for idx,i in enumerate(tbl):
            coord, color, label, norm = self.train[i]
            m=np.eye(3)# + np.random.randn(3,3)*0.1
            m[0][0]*=np.random.randint(0,2)*2-1
            theta=np.random.rand()*2*math.pi
            m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
            coord=np.matmul(coord,m)
            norm=np.matmul(norm,m)
            num_points = coord.shape[0]
            if num_points > self.cfg.MAX_POINTS_NUM:
                sample_idx = np.random.choice(num_points, self.cfg.MAX_POINTS_NUM, replace = False)
                coord = coord[sample_idx, ...]
                color = color[sample_idx, ...]
                label = label[sample_idx, ...]
                norm  = norm[sample_idx, ...]
            point_list, color_list, label_list, norm_List = [coord.astype(np.float32)], \
                                                 [(color + np.random.randn(3)*0.1).astype(np.float32)], \
                                                 [label.astype(np.int32)], \
                                                 [norm.astype(np.float32)]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_list[0])
            pcd.colors = o3d.utility.Vector3dVector(color)
            pcd.normals = o3d.utility.Vector3dVector(norm)
            nself = compute_knn(point_list[0], point_list[0], self.cfg.K_self)
            nei_forward_list, nei_propagate_list, nei_self_list = [], [], [nself]

            for j, grid_s in enumerate(self.cfg.grid_size):
                pcd = pcd.voxel_down_sample(voxel_size=self.cfg.grid_size[j])
                sub_point = np.asarray(pcd.points)
                sub_color = np.asarray(pcd.colors)
                sub_norm  = np.asarray(pcd.normals)

                nforward = compute_knn(point_list[j], sub_point, self.cfg.K_forward)
                npropagate = compute_knn(sub_point, point_list[j], self.cfg.K_propagate)

                if self.cfg.use_ASPP and j == len(self.cfg.grid_size) - 1:
                    nself = compute_knn(sub_point, sub_point, 8 * self.cfg.K_self)
                else:
                    nself = compute_knn(sub_point, sub_point, self.cfg.K_self)

                nei_forward_list.append(nforward)
                nei_propagate_list.append(npropagate)
                nei_self_list.append(nself)

                point_list.append(sub_point)
                color_list.append(sub_color)
                norm_List.append(sub_norm)

            locs.append(point_list)
            feats.append(color_list)
            labels.append(label_list)
            norms.append(norm_List)

            nei_forward.append(nei_forward_list)
            nei_propagate.append(nei_propagate_list)
            nei_self.append(nei_self_list)

        return {'locs': locs, 'feats': feats, \
                'nei_forward': nei_forward, 'nei_propagate': nei_propagate, 'nei_self': nei_self, \
                'labels': labels, 'id': tbl, 'norms': norms}

    def valSet(self, tbl):
        locs=[]
        feats=[]
        labels=[]
        norms = []
        nei_forward = []
        nei_propagate = []
        nei_self = []
        for idx,i in enumerate(tbl):
            coord, color, label, norm = self.val[i]
            m=np.eye(3)
            m[0][0]*=np.random.randint(0,2)*2-1
            theta=np.random.rand()*2*math.pi
            m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
            coord=np.matmul(coord,m)
            norm=np.matmul(norm,m)

            num_points = coord.shape[0]
            if num_points > self.cfg.MAX_POINTS_NUM:
                sample_idx = np.random.choice(num_points, self.cfg.MAX_POINTS_NUM, replace = False)
                coord = coord[sample_idx, ...]
                color = color[sample_idx, ...]
                label = label[sample_idx, ...]
                norm  = norm[sample_idx, ...]

            point_list, color_list, label_list, norm_List = [coord.astype(np.float32)], \
                                                 [color.astype(np.float32)], \
                                                 [label.astype(np.int32)], \
                                                 [norm.astype(np.float32)]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_list[0])
            pcd.colors = o3d.utility.Vector3dVector(color)
            pcd.normals = o3d.utility.Vector3dVector(norm)

            nself = compute_knn(point_list[0], point_list[0], self.cfg.K_self)
            nei_forward_list, nei_propagate_list, nei_self_list = [], [], [nself]

            for j, grid_s in enumerate(self.cfg.grid_size):
                pcd = pcd.voxel_down_sample(voxel_size=self.cfg.grid_size[j])
                sub_point = np.asarray(pcd.points)
                sub_color = np.asarray(pcd.colors)
                sub_norm  = np.asarray(pcd.normals)

                nforward = compute_knn(point_list[j], sub_point, self.cfg.K_forward)
                npropagate = compute_knn(sub_point, point_list[j], self.cfg.K_propagate)

                if self.cfg.use_ASPP and j == len(self.cfg.grid_size) - 1:
                    nself = compute_knn(sub_point, sub_point, 8 * self.cfg.K_self)
                else:
                    nself = compute_knn(sub_point, sub_point, self.cfg.K_self)

                nei_forward_list.append(nforward)
                nei_propagate_list.append(npropagate)
                nei_self_list.append(nself)

                point_list.append(sub_point)
                color_list.append(sub_color)
                #label_list.append(sub_label)
                norm_List.append(sub_norm)


            locs.append(point_list)
            feats.append(color_list)
            labels.append(label_list)
            norms.append(norm_List)

            nei_forward.append(nei_forward_list)
            nei_propagate.append(nei_propagate_list)
            nei_self.append(nei_self_list)

        return {'locs': locs, 'feats': feats, \
                'nei_forward': nei_forward, 'nei_propagate': nei_propagate, 'nei_self': nei_self, \
                'labels': labels, 'id': tbl, 'norms': norms}

    def getdataLoaders(self):
        train_data_loader = torch.utils.data.DataLoader(
            list(range(len(self.train))),batch_size=self.cfg.BATCH_SIZE, collate_fn=self.trainSet, num_workers=5, shuffle=True)
        val_data_loader = torch.utils.data.DataLoader(
            list(range(len(self.val))),batch_size=1, collate_fn=self.valSet, num_workers=5, shuffle=True)
        return train_data_loader, val_data_loader


