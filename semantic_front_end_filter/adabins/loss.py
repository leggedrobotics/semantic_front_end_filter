from math import nan
import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence
from semantic_front_end_filter.adabins.pointcloudUtils import RaycastCamera
import os
class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            # input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
            input = nn.functional.interpolate(input, target.shape[-2:], align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)

class UncertaintyLoss(nn.Module):  # Add variance to loss
    def __init__(self, train_args):
        super(UncertaintyLoss, self).__init__()
        self.name = 'SILog'
        self.args = train_args
        self.depth_variance_ratio = self.args.traj_distance_variance_ratio

    def forward(self, input, target, target_variance, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        # mask = mask if mask is not None else mask.fill_(True)
        if mask is not None:
            input = input[mask]
            target = target[mask]
            target_variance = target_variance[mask]
        # g = (torch.log(input) - torch.log(target))/target_variance + target_variance
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        # Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        if(target_variance.numel() !=0):
           #  Dg = 1/(input.shape[0]) * torch.sum(0.5 * torch.pow(input - target, 2)/target_variance)
            Dg = torch.sum(0.5 * torch.pow(input - target, 2)/(1e-3 + target*self.depth_variance_ratio + target_variance))
            if(self.args.scale_loss_with_point_number):
                Dg /= input.shape[0]
        else:
            Dg = torch.tensor(0.).to('cuda')
        return Dg
class ConsistencyLoss(nn.Module):  # Add variance to loss
    def __init__(self, train_args):
        super(ConsistencyLoss, self).__init__()
        self.name = "ConsistencyLoss"
        self.args = train_args
        self.raycastCamera = RaycastCamera(
                    os.getcwd()+self.args.camera_cali_path, device = torch.device("cuda:0"))
    def cal_loss_two(self, pcA, pcB, poseA, poseB):
        # project pc_image A to 3D
        pcA[pcA==0] = nan
        pts = self.raycastCamera.project_depth_to_cloud(poseA, pcA[0, :, :].T)
        pts = pts[~torch.isnan(pts).any(axis=1), :]
        # reproject the points on the image plane of B
        # pcA_reproject = torch.zeros_like(pcA).float()
        pcA_reproject = pcB.clone()
        self.raycastCamera.project_cloud_to_depth(poseB.float(), pts.float(), pcA_reproject.float())
        loss = (pcA_reproject - pcB)[pcA_reproject!=0].mean()
        return loss
    
    def forward(self, pred, pose):
        # pred    n*1*540*720
        # pose    n*7
        loss = torch.tensor(0.).to('cuda')
        for i in range(pred.shape[0]-1):
            loss += self.cal_loss_two(pred[i], pred[i+1], pose[i], pose[i+1])
        return loss
        
class EdgeAwareLoss(nn.Module):  # Add variance to loss
    def __init__(self, train_args):
        super(EdgeAwareLoss, self).__init__()
        self.name = 'EdgeAwareLoss'
        self.args = train_args

    def forward(self, input, target, interpolate=True):
        # return torch.tensor(0)
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
        
        # input_gradient = torch.gradient(input, dim=[2, 3]) # gradient of prediction
        # target_gradient = torch.gradient(target[:, 0:1, :, :], dim=[2, 3]) # gradient of input image
        # For torch 1.10.0+cu113
        if input.shape[0]==1:
            input_gradient = [gra[None,None,:,:] for gra in torch.gradient(input[0,0], dim=[0,1])]
            target_gradient = [gra[None,None,:,:] for gra in torch.gradient(target[0,0], dim=[0,1])]
        else:
            input_gradient = [gra[:,None,:,:] for gra in torch.gradient(input[:,0], dim=[1,2])]
            target_gradient = [gra[:,None,:,:] for gra in torch.gradient(target[:,0], dim=[1,2])]
        loss = 1/torch.numel(input) * torch.sum(torch.abs(input_gradient[0]) * torch.exp(-torch.abs(target_gradient[0])) + torch.abs(input_gradient[1]) * torch.exp(-torch.abs(target_gradient[1])))
        return loss

class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss
