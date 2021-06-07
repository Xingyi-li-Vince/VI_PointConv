import open3d as o3d
import argparse
import sys
import os

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import pickle
import datetime
import logging
import pandas as pd

from tqdm import tqdm
from model_architecture import VI_PointConv
from pathlib import Path
from collections import defaultdict
from easydict import EasyDict as edict
import yaml

from scannet_data_loader import data_loader_ScanNet

def get_default_configs(cfg):
    cfg.num_level = len(cfg.grid_size)
    cfg.feat_dim  = [cfg.base_dim * (i + 1) for i in range(cfg.num_level)]
    return cfg

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

def tensorlize(pointclouds, edges_self, edges_forward, edges_propagate, feat, target, norms, cfg):
    pointclouds = tensorlizeList(pointclouds, cfg)
    norms = tensorlizeList(norms, cfg)
    edges_self = tensorlizeList(edges_self, cfg, True)
    edges_forward = tensorlizeList(edges_forward, cfg, True)
    edges_propagate = tensorlizeList(edges_propagate, cfg, True)

    feat = torch.from_numpy(feat).to(cfg.device).unsqueeze(0)
    target = torch.from_numpy(target).long().to(cfg.device).unsqueeze(0)

    return pointclouds, edges_self, edges_forward, edges_propagate, feat, target, norms

def listToBatch(pointclouds, edges_self, edges_forward, edges_propagate, feat, target, norms):
    num_sample = len(pointclouds)

    #process sample 0
    pointcloudsBatch = pointclouds[0]
    pointcloudsNormsBatch = norms[0]
    featBatch = feat[0][0]
    targetBatch = target[0][0]

    edgesSelfBatch = edges_self[0]
    edgesForwardBatch = edges_forward[0]
    edgesPropagateBatch = edges_propagate[0]

    points_stored = [val.shape[0] for val in pointcloudsBatch]

    for i in range(1, num_sample):
        featBatch = np.concatenate([featBatch, feat[i][0]], 0)
        targetBatch = np.concatenate([targetBatch, target[i][0]], 0)

        for j in range(len(edges_forward[i])):
            tempMask = edges_forward[i][j] == -1
            edges_forwardAdd = edges_forward[i][j] + points_stored[j]
            edges_forwardAdd[tempMask] = -1
            edgesForwardBatch[j] = np.concatenate([edgesForwardBatch[j], \
                                   edges_forwardAdd], 0)

            tempMask2 = edges_propagate[i][j] == -1
            edges_propagateAdd = edges_propagate[i][j] + points_stored[j + 1]
            edges_propagateAdd[tempMask2] = -1
            edgesPropagateBatch[j] = np.concatenate([edgesPropagateBatch[j], \
                                   edges_propagateAdd], 0)

        for j in range(len(pointclouds[i])):
            tempMask3 = edges_self[i][j] == -1
            edges_selfAdd = edges_self[i][j] + points_stored[j]
            edges_selfAdd[tempMask3] = -1
            edgesSelfBatch[j] = np.concatenate([edgesSelfBatch[j], \
                                    edges_selfAdd], 0)

            pointcloudsBatch[j] = np.concatenate([pointcloudsBatch[j], pointclouds[i][j]], 0)
            pointcloudsNormsBatch[j] = np.concatenate([pointcloudsNormsBatch[j], norms[i][j]], 0)

            points_stored[j] += pointclouds[i][j].shape[0]

    return pointcloudsBatch, edgesSelfBatch, edgesForwardBatch, edgesPropagateBatch, \
            featBatch, targetBatch, pointcloudsNormsBatch


def main():
    config_file_path = './config.yaml'
    cfg = edict(yaml.safe_load(open('./config.yaml', 'r')))
    cfg = get_default_configs(cfg)
    
    scannet_data = data_loader_ScanNet(cfg)
    train_data_loader, val_data_loader = scannet_data.getdataLoaders()


    torch.cuda.set_device(0)
    device = cfg.device
    
    class2label = {cls: i for i,cls in enumerate(cfg.classes)}
    
    seg_classes = class2label
    seg_label_to_cat = {}
    for i,cat in enumerate(seg_classes.keys()):
        catlabel = seg_classes[cat]
        seg_label_to_cat[catlabel] = cat


    '''CREATE DIR'''
    experiment_dir = Path(cfg.experiment_dir)
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%sSemSeg-'%cfg.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % (config_file_path, log_dir))
    os.system('cp %s %s' % (os.path.basename(__file__), log_dir))

    '''LOG'''
    logger = logging.getLogger(cfg.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_semseg.txt'%cfg.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------TRAINING-----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(cfg)
    print('Load data...')

    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = VI_PointConv(cfg)
    class_weights = torch.tensor(cfg.weights).float().to(device)

    if cfg.pretrain is not None:
        model.load_state_dict(torch.load(cfg.pretrain))
        print('load model %s'%cfg.pretrain)
        logger.info('load model %s'%cfg.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')

    init_epoch = int(cfg.pretrain[-14:-11]) if cfg.pretrain is not None else 0

    if cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)
    elif cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.decay_rate)
    optimizer.param_groups[0]['initial_lr'] = cfg.learning_rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5, last_epoch = init_epoch - 1)
    LEARNING_RATE_CLIP = 1e-5

    '''GPU selection and multi-GPU'''
    if cfg.multi_gpu is not None:
        device_ids = [int(x) for x in cfg.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids = device_ids)
    else:
        model.to(device)

    history = defaultdict(lambda: list())
    best_acc = 0
    best_meaniou = 0

    for epoch in range(init_epoch, cfg.total_epoches):
        model.train()
        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        print('Learning rate:%f'%lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        for i, data in tqdm(enumerate(train_data_loader, 0), total=len(train_data_loader), smoothing=0.9):
            #import ipdb; ipdb.set_trace()

            pointclouds = data['locs']
            colors = data['feats']
            target = data['labels']
            norms = data['norms']
            edges_self = data['nei_self']
            edges_forward = data['nei_forward']
            edges_propagate = data['nei_propagate']

            pointclouds, edges_self, edges_forward, edges_propagate, feat, target, norms = \
                listToBatch(pointclouds, edges_self, edges_forward, edges_propagate, colors, target, norms)

            pointclouds, edges_self, edges_forward, edges_propagate, feat, target, norms = \
                tensorlize(pointclouds, edges_self, edges_forward, edges_propagate, feat, target, norms, cfg)

            pred = model(feat, pointclouds, edges_self, edges_forward, edges_propagate, norms)
            pred = pred.contiguous().view(-1, cfg.num_classes)
            target = target.view(-1, 1)[:, 0]
            loss = F.cross_entropy(pred, target, class_weights)

            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()

            if (i + 1) % cfg.repeat_time == 0:
                optimizer.step()
                optimizer.zero_grad()


        test_metrics, test_hist_acc, cat_mean_iou = test_semseg(model.eval(), val_data_loader,
                    seg_label_to_cat, cfg, num_classes=cfg.num_classes)
        mean_iou = np.mean(cat_mean_iou)
        str_out = 'Epoch %d %s accuracy: %f meanIOU: %f'%(epoch, blue('test'),
                test_metrics['accuracy'], mean_iou)
        print(str_out)
        logger.info(str_out)

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if mean_iou > best_meaniou:
            best_meaniou = mean_iou
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, cfg.model_name, epoch, best_meaniou))
            torch.save({'opt' : optimizer.state_dict(), 'sch' : scheduler.state_dict()}, '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, cfg.model_name+'Opt', epoch, best_meaniou))
            logger.info(cat_mean_iou)
            logger.info('Save model ...')
            print('Save model..')
            print(cat_mean_iou)
        print('Best accuracy is: %.5f'%best_acc)
        logger.info('Best accuracy is: %.5f'%best_acc)
        print('Best meanIOU is: %.5f'%best_meaniou)
        logger.info('Best meanIOU is: %.5f'%best_meaniou)



def test_semseg(model, loader, catdict, cfg, num_classes = 13):
    #import pdb; pdb.set_trace()
    iou_tabel = np.zeros((len(catdict),3))
    metrics = defaultdict(lambda:list())
    hist_acc = []
    for batch_id, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        pointclouds = data['locs']
        colors = data['feats']
        target = data['labels']
        norms = data['norms']
        edges_self = data['nei_self']
        edges_forward = data['nei_forward']
        edges_propagate = data['nei_propagate']

        pointclouds, edges_self, edges_forward, edges_propagate, feat, target, norms = \
            listToBatch(pointclouds, edges_self, edges_forward, edges_propagate, colors, target, norms)

        pointclouds, edges_self, edges_forward, edges_propagate, feat, target, norms = \
            tensorlize(pointclouds, edges_self, edges_forward, edges_propagate, feat, target, norms, cfg)

        with torch.no_grad():
            pred = model(feat, pointclouds, edges_self, edges_forward, edges_propagate, norms)

        #print(pred.size())
        iou_tabel, iou_list = compute_cat_iou(pred,target,iou_tabel)
        # shape_ious += compute_overall_iou(pred, target, num_classes)
        pred = pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = pred.data.max(1)[1]
        valid_mask = target >= 0
        pred_choice = pred_choice[valid_mask]
        target = target[valid_mask]
        correct = pred_choice.eq(target.data).cpu().sum()
        num_point = target.shape[0]
        metrics['accuracy'].append(correct.item()/num_point)
    iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(metrics['accuracy'])
    metrics['iou'] = np.mean(iou_tabel[:, 2])
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = [catdict[i] for i in range(len(catdict)) ]
    # print(iou_tabel)
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()

    return metrics, hist_acc, cat_iou

def compute_cat_iou(pred,target,iou_tabel):
    iou_list = []
    target = target.cpu().data.numpy()
    for j in range(pred.size(0)):
        batch_pred = pred[j]
        batch_target = target[j]
        batch_choice = batch_pred.data.max(1)[1].cpu().data.numpy()
        batch_mask = target[j] >= 0
        batch_choice = batch_choice[batch_mask]
        batch_target = batch_target[batch_mask]
        for cat in np.unique(batch_target):
            # intersection = np.sum((batch_target == cat) & (batch_choice == cat))
            # union = float(np.sum((batch_target == cat) | (batch_choice == cat)))
            # iou = intersection/union if not union ==0 else 1
            I = np.sum(np.logical_and(batch_choice == cat, batch_target == cat))
            U = np.sum(np.logical_or(batch_choice == cat, batch_target == cat))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            iou_tabel[cat,0] += iou
            iou_tabel[cat,1] += 1
            iou_list.append(iou)
    return iou_tabel,iou_list



if __name__ == '__main__':
    main()
