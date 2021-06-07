import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] / [B, S, K]
    Return:
        new_points:, indexed points data, [B, S, C] / [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class WeightNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights =  F.relu(bn(conv(weights)), inplace=True)

        return weights


class PointConv(nn.Module):
    def __init__(self, in_channel, out_channel, mlp, cfg, weightnet = [9, 16]):
        super(PointConv, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel + 3
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        self.weightnet = WeightNet(weightnet[0], weightnet[1])
        self.linear = nn.Linear(last_ch * weightnet[-1], out_channel)
        if cfg.BATCH_NORM:
            self.bn_linear = nn.BatchNorm1d(out_channel)

    def forward(self, dense_xyz, sparse_xyz, dense_feats, nei_inds, dense_xyz_norm, sparse_xyz_norm):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        """
        B, N, D = dense_xyz.shape
        _, M, _ = sparse_xyz.shape
        _, _, in_ch = dense_feats.shape
        _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0

        gathered_xyz = index_points(dense_xyz, nei_inds)
        #localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K, D]
        localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        gathered_norm = index_points(dense_xyz_norm, nei_inds)

        n_alpha = gathered_norm
        n_miu = sparse_xyz_norm
        r_miu = localized_xyz
        r_hat = F.normalize(r_miu, dim = 3)
        v_miu = n_miu.unsqueeze(dim=2) - torch.matmul(n_miu.unsqueeze(dim=2), r_hat.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) * r_hat
        v_miu = F.normalize(v_miu, dim = 3)
        w_miu = torch.cross(r_hat, v_miu, dim=3)
        w_miu = F.normalize(w_miu, dim = 3)
        theta1 = torch.matmul(n_alpha, n_miu.unsqueeze(dim=3))
        theta2 = torch.matmul(r_hat, n_miu.unsqueeze(dim=3))
        theta3 = torch.sum(r_hat * n_alpha, dim = 3, keepdim = True)
        theta4 = torch.matmul(r_miu, n_miu.unsqueeze(dim=3))
        theta5 = torch.sum(n_alpha * r_hat, dim = 3, keepdim = True)
        theta6 = torch.sum(n_alpha * v_miu, dim = 3, keepdim = True)
        theta7 = torch.sum(n_alpha * w_miu, dim = 3, keepdim = True)
        theta8 = torch.sum(r_miu*torch.cross(n_alpha, n_miu.unsqueeze(dim=2).repeat(1,1,K,1), dim = 3), dim = 3, keepdim = True)
        theta9 = torch.norm(r_miu, dim =3, keepdim = True)
        weightNetInput = torch.cat([theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz], dim = 3).contiguous()

        gathered_feat = index_points(dense_feats, nei_inds) #[B, M, K, in_ch]
        new_feat = torch.cat([gathered_feat, localized_xyz], dim = -1)

        new_feat = new_feat.permute(0, 3, 2, 1) # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.mlp_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                new_feat =  F.relu(bn(conv(new_feat)), inplace=True)
            else:
                new_feat =  F.relu(conv(new_feat), inplace=True)

        weightNetInput = weightNetInput.permute(0, 3, 2, 1)
        weights = self.weightnet(weightNetInput)

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1)
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)

        new_feat = torch.matmul(input=new_feat.permute(0, 3, 1, 2), other = weights.permute(0, 3, 2, 1)).view(B, M, -1)
        #new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = self.linear(new_feat)
        if self.cfg.BATCH_NORM:
            new_feat = self.bn_linear(new_feat.permute(0, 2, 1))
            new_feat = F.relu(new_feat, inplace=True).permute(0, 2, 1)
        else:
            new_feat = F.relu(new_feat, inplace=True)

        return new_feat

    def _gather_nd(self, inputs, nn_idx):
        """
        input: (num_points, num_dims)
        nn_idx:(num_points', k)

        output:
        output:(num_points', k)
        """
        inputs = inputs.permute(1, 0)
        nn_idx = nn_idx.permute(1, 0)

        c, n1 = inputs.size()
        k, n2 = nn_idx.size()

        # (c, k*n)
        nn_idx = nn_idx.contiguous().unsqueeze(dim=0).expand(c, -1, -1).contiguous().view(c, -1)

        inputs_gather = inputs.gather(dim=-1, index=nn_idx).contiguous().view(c, k, n2).permute(2, 1, 0).contiguous()

        return inputs_gather

class PointConvTranspose(nn.Module):
    def __init__(self, in_channel, out_channel, mlp, cfg, weightnet = [9, 16], mlp2 = None):
        super(PointConvTranspose, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel + 3
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        self.weightnet = WeightNet(weightnet[0], weightnet[1])
        self.linear = nn.Linear(last_ch * weightnet[-1], out_channel)
        if self.cfg.BATCH_NORM:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.mlp2_convs = nn.ModuleList()
        self.mlp2_bns = nn.ModuleList()
        if mlp2 is not None:
            for i in range(1, len(mlp2)):
                self.mlp2_convs.append(nn.Conv1d(mlp2[i-1], mlp2[i], 1))
                if self.cfg.BATCH_NORM:
                    self.mlp2_bns.append(nn.BatchNorm1d(mlp2[i]))

    def forward(self, sparse_xyz, dense_xyz, sparse_feats, dense_feats, nei_inds, dense_xyz_norm, sparse_xyz_norm):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3)
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        """
        B, N, D = sparse_xyz.shape
        _, M, _ = dense_xyz.shape
        _, _, in_ch = sparse_feats.shape
        _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0

        gathered_xyz = index_points(sparse_xyz, nei_inds)
        #localized_xyz = gathered_xyz - dense_xyz.view(B, M, 1, D) #[B, M, K, D]
        localized_xyz = gathered_xyz - dense_xyz.unsqueeze(dim=2)
        gathered_norm = index_points(sparse_xyz_norm, nei_inds)

        n_alpha = gathered_norm
        n_miu = dense_xyz_norm
        r_miu = localized_xyz
        r_hat = F.normalize(r_miu, dim = 3)
        v_miu = n_miu.unsqueeze(dim=2) - torch.matmul(n_miu.unsqueeze(dim=2), r_hat.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) * r_hat
        v_miu = F.normalize(v_miu, dim = 3)
        w_miu = torch.cross(r_hat, v_miu, dim=3)
        w_miu = F.normalize(w_miu, dim = 3)
        theta1 = torch.matmul(n_alpha, n_miu.unsqueeze(dim=3))
        theta2 = torch.matmul(r_hat, n_miu.unsqueeze(dim=3))
        theta3 = torch.sum(r_hat * n_alpha, dim = 3, keepdim = True)
        theta4 = torch.matmul(r_miu, n_miu.unsqueeze(dim=3))
        theta5 = torch.sum(n_alpha * r_hat, dim = 3, keepdim = True)
        theta6 = torch.sum(n_alpha * v_miu, dim = 3, keepdim = True)
        theta7 = torch.sum(n_alpha * w_miu, dim = 3, keepdim = True)
        theta8 = torch.sum(r_miu*torch.cross(n_alpha, n_miu.unsqueeze(dim=2).repeat(1,1,K,1), dim = 3), dim = 3, keepdim = True)
        theta9 = torch.norm(r_miu, dim =3, keepdim = True)
        weightNetInput = torch.cat([theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz], dim = 3).contiguous()

        gathered_feat = index_points(sparse_feats, nei_inds) #[B, M, K, in_ch]
        new_feat = torch.cat([gathered_feat, localized_xyz], dim = -1)

        new_feat = new_feat.permute(0, 3, 2, 1) # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.mlp_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                new_feat =  F.relu(bn(conv(new_feat)), inplace=True)
            else:
                new_feat = F.relu(conv(new_feat), inplace=True)

        weightNetInput = weightNetInput.permute(0, 3, 2, 1)
        weights = self.weightnet(weightNetInput)

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1) # [B, D, K, M]
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)
        # C_mid = weights.shape[1]

        new_feat = torch.matmul(input=new_feat.permute(0, 3, 1, 2), other = weights.permute(0, 3, 2, 1)).view(B, M, -1)
        #new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = self.linear(new_feat)
        if self.cfg.BATCH_NORM:
            new_feat = self.bn_linear(new_feat.permute(0, 2, 1))
            new_feat = F.relu(new_feat, inplace=True) #[B, C, M]
        else:
            new_feat = F.relu(new_feat.permute(0, 2, 1), inplace=True)

        if dense_feats is not None:
            #new_feat = torch.cat([new_feat, dense_feats.permute(0, 2, 1)], dim = 1)
            new_feat = new_feat + dense_feats.permute(0, 2, 1)

        for i, conv in enumerate(self.mlp2_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp2_bns[i]
                new_feat = F.relu(bn(conv(new_feat)), inplace=True)
            else:
                new_feat = F.relu(conv(new_feat), inplace=True)

        new_feat = new_feat.permute(0, 2, 1)

        return new_feat

    def _gather_nd(self, inputs, nn_idx):
        """
        input: (num_points, num_dims)
        nn_idx:(num_points', k)

        output:
        output:(num_points', k)
        """
        inputs = inputs.permute(1, 0)
        nn_idx = nn_idx.permute(1, 0)

        c, n1 = inputs.size()
        k, n2 = nn_idx.size()

        # (c, k*n)
        nn_idx = nn_idx.contiguous().unsqueeze(dim=0).expand(c, -1, -1).contiguous().view(c, -1)

        inputs_gather = inputs.gather(dim=-1, index=nn_idx).contiguous().view(c, k, n2).permute(2, 1, 0).contiguous()

        return inputs_gather

class PointConvResBlock(nn.Module):
    def __init__(self, in_channel, mlp, cfg, weightnet = [9, 16]):
        super(PointConvResBlock, self).__init__()
        self.cfg = cfg
        self.in_channel = in_channel
        self.out_channel = in_channel

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel + 3
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            if self.cfg.BATCH_NORM:
                self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

        self.weightnet = WeightNet(weightnet[0], weightnet[1])
        self.linear = nn.Linear(last_ch * weightnet[-1], in_channel)
        if self.cfg.BATCH_NORM:
            self.bn_linear = nn.BatchNorm1d(in_channel)

    def forward(self, xyz, feats, nei_inds, xyz_norm):
        """
        xyz: tensor (batch_size, num_points, 3)
        feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points, K)
        """
        B, N, D = xyz.shape
        M = N
        _, _, in_ch = feats.shape
        _, _, K = nei_inds.shape

        # nei_inds = nei_inds.clone().detach()
        # nei_inds_mask = (nei_inds != -1).float()
        # nn_idx_divider = nei_inds_mask.sum(dim = -1)
        # nn_idx_divider[nn_idx_divider == 0] = 1
        # nei_inds[nei_inds == -1] = 0

        gathered_xyz = index_points(xyz, nei_inds)
        #localized_xyz = gathered_xyz - xyz.view(B, M, 1, D) #[B, M, K, D]
        localized_xyz = gathered_xyz - xyz.unsqueeze(dim=2)
        gathered_norm = index_points(xyz_norm, nei_inds)

        n_alpha = gathered_norm
        n_miu = xyz_norm
        r_miu = localized_xyz
        r_hat = F.normalize(r_miu, dim = 3)
        v_miu = n_miu.unsqueeze(dim=2) - torch.matmul(n_miu.unsqueeze(dim=2), r_hat.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) * r_hat
        v_miu = F.normalize(v_miu, dim = 3)
        w_miu = torch.cross(r_hat, v_miu, dim=3)
        w_miu = F.normalize(w_miu, dim = 3)
        theta1 = torch.matmul(n_alpha, n_miu.unsqueeze(dim=3))
        theta2 = torch.matmul(r_hat, n_miu.unsqueeze(dim=3))
        theta3 = torch.sum(r_hat * n_alpha, dim = 3, keepdim = True)
        theta4 = torch.matmul(r_miu, n_miu.unsqueeze(dim=3))
        theta5 = torch.sum(n_alpha * r_hat, dim = 3, keepdim = True)
        theta6 = torch.sum(n_alpha * v_miu, dim = 3, keepdim = True)
        theta7 = torch.sum(n_alpha * w_miu, dim = 3, keepdim = True)
        theta8 = torch.sum(r_miu*torch.cross(n_alpha, n_miu.unsqueeze(dim=2).repeat(1,1,K,1), dim = 3), dim = 3, keepdim = True)
        theta9 = torch.norm(r_miu, dim =3, keepdim = True)
        weightNetInput = torch.cat([theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, localized_xyz], dim = 3).contiguous()

        gathered_feat = index_points(feats, nei_inds) #[B, M, K, in_ch]
        new_feat = torch.cat([gathered_feat, localized_xyz], dim = -1)

        new_feat = new_feat.permute(0, 3, 2, 1) # [B, in_ch+D, K, M]
        for i, conv in enumerate(self.mlp_convs):
            if self.cfg.BATCH_NORM:
                bn = self.mlp_bns[i]
                new_feat =  F.relu(bn(conv(new_feat)), inplace=True)
            else:
                new_feat = F.relu(conv(new_feat), inplace=True)

        weightNetInput = weightNetInput.permute(0, 3, 2, 1)
        weights = self.weightnet(weightNetInput)

        # localized_xyz = localized_xyz.permute(0, 3, 2, 1) # [B, D, K, M]
        # weights = self.weightnet(localized_xyz)*nei_inds_mask.permute(0,2,1).unsqueeze(dim=1)

        new_feat = torch.matmul(input=new_feat.permute(0, 3, 1, 2), other = weights.permute(0, 3, 2, 1)).view(B, M, -1)
        #new_feat = new_feat/nn_idx_divider.unsqueeze(dim=-1)
        new_feat = self.linear(new_feat)
        if self.cfg.BATCH_NORM:
            new_feat = self.bn_linear(new_feat.permute(0, 2, 1))
            new_feat = F.relu(new_feat, inplace=True).permute(0, 2, 1)
        else:
            new_feat = F.relu(new_feat, inplace=True)

        #import ipdb; ipdb.set_trace()

        new_feat = new_feat + feats

        return new_feat

class VI_PointConv(nn.Module):
    def __init__(self, cfg):
        super(VI_PointConv, self).__init__()

        self.cfg = cfg
        self.total_level = cfg.num_level

        self.relu = torch.nn.ReLU(inplace=True)

        weightnet = [cfg.point_dim+9, 16] # 2 hidden layer

        #pointwise_encode
        self.pw_conv1 = nn.Conv1d(6 if cfg.USE_XYZ else 3, cfg.base_dim, 1)
        self.pw_bn1 = nn.BatchNorm1d(cfg.base_dim)
        self.pw_conv2 = nn.Conv1d(cfg.base_dim, cfg.base_dim, 1)
        self.pw_bn2 = nn.BatchNorm1d(cfg.base_dim)
        self.pw_conv3 = nn.Conv1d(cfg.base_dim, cfg.base_dim, 1)
        self.pw_bn3 = nn.BatchNorm1d(cfg.base_dim)

        self.pointconv = nn.ModuleList()
        self.pointconv_res = nn.ModuleList()

        for i in range(1, len(cfg.feat_dim)):
            in_ch = cfg.feat_dim[i - 1]
            out_ch = cfg.feat_dim[i]

            if cfg.USE_MLP:
                mlp = [in_ch, in_ch]
            else:
                mlp = []

            self.pointconv.append(PointConv(in_ch, out_ch, mlp, cfg, weightnet))

            if self.cfg.resblocks[i] == 0:
                self.pointconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(self.cfg.resblocks[i]):
                    res_blocks.append(PointConvResBlock(out_ch, mlp, cfg, weightnet))
                self.pointconv_res.append(res_blocks)


        self.pointdeconv = nn.ModuleList()
        self.pointdeconv_res = nn.ModuleList()

        for i in range(len(cfg.feat_dim) - 2, -1, -1):
            in_ch = cfg.feat_dim[i + 1]
            out_ch = cfg.feat_dim[i]

            if cfg.USE_MLP:
                mlp = [out_ch, out_ch]
            else:
                mlp = []
            mlp2 = [out_ch, out_ch]
            self.pointdeconv.append(PointConvTranspose(in_ch, out_ch, mlp, cfg, weightnet, mlp2))

            if self.cfg.resblocks[i] == 0:
                self.pointdeconv_res.append(nn.ModuleList([]))
            else:
                res_blocks = nn.ModuleList()
                for _ in range(self.cfg.resblocks_back[i]):
                    res_blocks.append(PointConvResBlock(out_ch, mlp, cfg, weightnet))
                self.pointdeconv_res.append(res_blocks)

        #pointwise_decode
        self.fc1 = nn.Conv1d(cfg.base_dim, cfg.num_classes, 1)

    def forward(self, feat, pointclouds, edges_self, edges_forward, edges_propagate, norms):

        #import ipdb; ipdb.set_trace()

        #encode pointwise info
        feat = torch.cat([feat, pointclouds[0]], -1) if self.cfg.USE_XYZ else feat
        feat = feat.permute(0, 2, 1)
        pointwise_feat = self.relu(self.pw_bn1(self.pw_conv1(feat)))
        pointwise_feat = self.relu(self.pw_bn2(self.pw_conv2(pointwise_feat)))
        pointwise_feat = self.relu(self.pw_bn3(self.pw_conv3(pointwise_feat)))
        pointwise_feat = pointwise_feat.permute(0, 2, 1)

        feat_list = [pointwise_feat]
        for i, pointconv in enumerate(self.pointconv):
            dense_xyz = pointclouds[i]
            sparse_xyz = pointclouds[i + 1]

            dense_xyz_norm = norms[i]
            sparse_xyz_norm = norms[i + 1]

            dense_feat = feat_list[-1]
            nei_inds = edges_forward[i]
            sparse_feat = pointconv(dense_xyz, sparse_xyz, dense_feat, nei_inds, dense_xyz_norm, sparse_xyz_norm)

            for res_block in self.pointconv_res[i]:
                nei_inds = edges_self[i + 1]
                sparse_feat = res_block(sparse_xyz, sparse_feat, nei_inds, sparse_xyz_norm)

            feat_list.append(sparse_feat)

        sparse_feat = feat_list[-1]
        for i, pointdeconv in enumerate(self.pointdeconv):
            cur_level = self.total_level - 2 - i
            sparse_xyz = pointclouds[cur_level + 1]
            dense_xyz = pointclouds[cur_level]
            dense_feat = feat_list[cur_level]
            nei_inds = edges_propagate[cur_level]

            dense_xyz_norm = norms[cur_level]
            sparse_xyz_norm = norms[cur_level + 1]

            sparse_feat = pointdeconv(sparse_xyz, dense_xyz, sparse_feat, dense_feat, nei_inds, dense_xyz_norm, sparse_xyz_norm)

            # for res_block in self.pointdeconv_res[i]:
            #     nei_inds = edges_self[cur_level]
            #     sparse_feat = res_block(dense_xyz, sparse_feat, nei_inds)

            feat_list[cur_level] = sparse_feat

        fc = self.fc1(sparse_feat.permute(0, 2, 1)).permute(0, 2, 1)
        return fc
