import open3d as o3d
import argparse
import sys
import os
import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import datetime
import logging
from sklearn.neighbors import KDTree


from model_architecture import VI_PointConv
from pathlib import Path


from easydict import EasyDict as edict
import yaml

def get_default_configs(cfg):
    cfg.num_level = len(cfg.grid_size)
    cfg.feat_dim  = [cfg.base_dim * (i + 1) for i in range(cfg.num_level)]
    return cfg

def compute_knn(ref_points, query_points, K, dialated_rate = 1):
    num_ref_points = ref_points.shape[0]

    if num_ref_points < K or num_ref_points < dialated_rate * K:
        num_query_points = query_points.shape[0]
        inds = np.random.choice(num_ref_points, (num_query_points, K)).astype(np.int32)

        return inds

    #start_t = time.time()
    kdt = KDTree(ref_points)
    neighbors_idx = kdt.query(query_points, k = K * dialated_rate, return_distance=False)
    #print("num_points:", num_ref_points, "time:", time.time() - start_t)
    neighbors_idx = neighbors_idx[:, ::dialated_rate]

    return neighbors_idx

def parse_args():
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--test_reps', type=int, default=3, help='number of data loading workers')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--model_name', type=str, default='pointconv_res', help='Name of model')

    return parser.parse_args()

def tensorlizeList(nplist, cfg, is_index = False):
    ret_list = []
    for i in range(len(nplist)):
        if is_index:
            if nplist[i] is None:
                ret_list.append(None)
            else:
                ret_list.append(torch.from_numpy(nplist[i]).long().to(cfg.device).unsqueeze(0))
        else:
            ret_list.append(torch.from_numpy(nplist[i]).float().to(cfg.device).unsqueeze(0))

    return ret_list

def tensorlize(pointclouds, edges_self, edges_forward, edges_propagate, feat, norms, cfg):
    pointclouds = tensorlizeList(pointclouds, cfg)
    norms = tensorlizeList(norms, cfg)
    edges_self = tensorlizeList(edges_self, cfg, True)
    edges_forward = tensorlizeList(edges_forward, cfg, True)
    edges_propagate = tensorlizeList(edges_propagate, cfg, True)

    feat = torch.from_numpy(feat).to(cfg.device).unsqueeze(0)

    return pointclouds, edges_self, edges_forward, edges_propagate, feat, norms

def processData(points, feat, norm, cfg):
    point_list, color_list, norm_List = [points.astype(np.float32)], [feat.astype(np.float32)], [norm.astype(np.float32)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_list[0])
    pcd.colors = o3d.utility.Vector3dVector(color_list[0])
    pcd.normals = o3d.utility.Vector3dVector(norm_List[0])

    nself = compute_knn(point_list[0], point_list[0], cfg.K_self)
    nei_forward_list, nei_propagate_list, nei_self_list = [], [], [nself]

    for j, grid_s in enumerate(cfg.grid_size):
        #sub_point, sub_color = grid_subsampling(point_list[-1], color_list[-1], sampleDl = grid_s)
        pcd = pcd.voxel_down_sample(voxel_size=cfg.grid_size[j])
        sub_point = np.asarray(pcd.points)
        sub_color = np.asarray(pcd.colors)
        sub_norm  = np.asarray(pcd.normals)

        nforward = compute_knn(point_list[j], sub_point, cfg.K_forward)
        npropagate = compute_knn(sub_point, point_list[j], cfg.K_propagate)

        if cfg.use_ASPP and j == len(cfg.grid_size) - 1:
            nself = compute_knn(sub_point, sub_point, 8 * cfg.K_self)
        else:
            nself = compute_knn(sub_point, sub_point, cfg.K_self)

        nei_forward_list.append(nforward)
        nei_propagate_list.append(npropagate)
        nei_self_list.append(nself)

        point_list.append(sub_point)
        color_list.append(sub_color)
        norm_List.append(sub_norm)

    return point_list, color_list, norm_List, nei_forward_list, nei_propagate_list, nei_self_list

def get_test_point_clouds(path):
    test_instances = []
    test_files = glob.glob(path)
    for x in torch.utils.data.DataLoader(
            test_files,
            collate_fn=lambda x: torch.load(x[0]), num_workers=5):
        test_instances.append(x)
    print('test instances:', len(test_instances))
    return test_instances, test_files




def main():
    config_file_path = './config.yaml'
    cfg = edict(yaml.safe_load(open('./config.yaml', 'r')))
    cfg = get_default_configs(cfg)
    numOfRounds = 3
    device = cfg.device

    test_instances, test_files = get_test_point_clouds(cfg.test_data_path)
    
    class2label = {cls: i for i,cls in enumerate(cfg.classes)}
    mapper = np.array([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
    seg_classes = class2label
    seg_label_to_cat = {}
    for i,cat in enumerate(seg_classes.keys()):
        catlabel = seg_classes[cat]
        seg_label_to_cat[catlabel] = cat

    '''CREATE DIR'''
    experiment_dir = Path(cfg.test_dir)
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%sSemSeg-'%cfg.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    #os.system('cp %s %s' % (args.pretrain, checkpoints_dir))
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    ply_dir = file_dir.joinpath('plys/')
    ply_dir.mkdir(exist_ok=True)
    txt_dir = file_dir.joinpath('txts/')
    txt_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % (config_file_path, log_dir))
    os.system('cp %s %s' % (os.path.basename(__file__), log_dir))

    '''LOG'''
    logger = logging.getLogger(cfg.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'eval_%s_ScanNet.txt'%cfg.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------EVALUATING-----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(cfg)
    print('Load data...')

    model = VI_PointConv(cfg)
    cfg.pretrain = './weights.pth'

    if cfg.pretrain is not None:
        model.load_state_dict(torch.load(cfg.pretrain))
        print('load model %s'%cfg.pretrain)
        logger.info('load model %s'%cfg.pretrain)
    else:
        print('To EVAL, Please set pretrain.')
        logger.info('To EVAL, Please set pretrain.')
        sys.exit(0)

    colormap = np.array(create_color_palette40()) / 255.0

    model = model.to(device)
    model.eval()

    for scene_id, testPointCloud in enumerate(test_instances):
        coord, color, norms = testPointCloud
        scene_name = test_files[scene_id][-15-12:-15]
        num_points = coord.shape[0]
        pred_class = torch.zeros((num_points, cfg.num_classes))

        for _ in range(numOfRounds):

            point_idxs = []
            if num_points > cfg.MAX_POINTS_NUM:
                point_idx = np.arange(num_points)
                np.random.shuffle(point_idx)
                left_points = num_points
                while left_points > 0:
                    if left_points < cfg.MAX_POINTS_NUM:
                        point_idxs.append(point_idx[-cfg.MAX_POINTS_NUM:])
                    else:
                        start_idx = num_points - left_points
                        point_idxs.append(point_idx[start_idx:start_idx + cfg.MAX_POINTS_NUM])
                    left_points -= cfg.MAX_POINTS_NUM
            else:
                point_idxs.append(np.arange(num_points))

            m=np.eye(3)
            m[0][0]*=np.random.randint(0,2)*2-1
            theta=np.random.rand()*2*math.pi
            m=np.matmul(m,[[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
            coord=np.matmul(coord,m)
            norms=np.matmul(norms,m)
            for point_idx in point_idxs:
                points = coord[point_idx, ...]
                feat = color[point_idx, ...]
                norm = norms[point_idx, ...]

                point_list, color_list, norm_List, nei_forward_list, nei_propagate_list, nei_self_list = processData(points, feat, norm, cfg)
                pointclouds, edges_self, edges_forward, edges_propagate, feat, surnorm = tensorlize(point_list, nei_self_list, nei_forward_list, nei_propagate_list, color_list[0], norm_List, cfg)

                with torch.no_grad():
                    pred = model(feat, pointclouds, edges_self, edges_forward, edges_propagate, surnorm)
                    pred_prob = F.softmax(pred.squeeze(0), dim = -1)

                pred_class[point_idx] += pred_prob.cpu()

        pred_class = pred_class.contiguous().view(-1, cfg.num_classes)
        pred_class = pred_class/numOfRounds
        pred_choice = pred_class.data.max(1)[1].cpu().data.numpy()

        pred_choice = pred_choice.squeeze()

        
        pred_choice = mapper[pred_choice.astype(np.int32)]

        #import ipdb; ipdb.set_trace()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord)
        pcd.colors = o3d.utility.Vector3dVector(colormap[pred_choice])
        pcd.normals = o3d.utility.Vector3dVector(norms)
        o3d.io.write_point_cloud(str(ply_dir) + '/' + scene_name + ".ply" , pcd)

        fp = open(str(txt_dir) + '/' + scene_name + ".txt", "w")
        for l in pred_choice:
            fp.write(str(int(l)) + '\n')
        fp.close()

        str_out = 'Scene Name: %s'%(scene_name)
        print(str_out)
        logger.info(str_out)




def create_color_palette40():
    return [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),
       (247, 182, 210),		# desk
       (66, 188, 102),
       (219, 219, 141),		# curtain
       (140, 57, 197),
       (202, 185, 52),
       (51, 176, 203),
       (200, 54, 131),
       (92, 193, 61),
       (78, 71, 183),
       (172, 114, 82),
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),
       (153, 98, 156),
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),
       (227, 119, 194),		# bathtub
       (213, 92, 176),
       (94, 106, 211),
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]


if __name__ == '__main__':
    main()
