#!/usr/bin/env python
# coding: utf-8

# from train import *
import os
import torch
from torch import nn
import numpy as np
from semantic_front_end_filter.adabins.dataloader import DepthDataLoader
from semantic_front_end_filter.adabins import model_io, models
from semantic_front_end_filter.adabins.cfgUtils import parse_args
from simple_parsing import ArgumentParser
import yaml

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.transforms import Bbox
from ruamel.yaml import YAML
import cv2
from tqdm import tqdm

test_loader_iter = None
train_loader_iter = None
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# def full_extent(ax, pad=0.0):
#     """Get the full extent of an axes, including axes labels, tick labels, and
#     titles."""
#     # For text objects, we need to draw the figure first, otherwise the extents
#     # are undefined.
#     ax.figure.canvas.draw()
#     items = ax.get_xticklabels() + ax.get_yticklabels()
# #    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
#     items += [ax, ax.title]
#     bbox = Bbox.union([item.get_window_extent() for item in items])

#     return bbox.expanded(1.0 + pad, 1.0 + pad)

AnnotatedSupportSurfaceIoUs = np.array([])
AnnotatedObstaclesIoUs = np.array([])
SelfGenratedSSIoUs = np.array([])

@torch.no_grad()
def computeIoUs(model, args, loader = 'test', env = 'forest', depth_limit = 5, print_result=True):
    # load dataset
    if env == 'high grass':
        args.trainconfig.testing = ["Reconstruct_2022-07-19-18-16-39_0"] # 95
        args.trainconfig.training = ["Reconstruct_2022-07-19-20-46-08_0"] # 132
    elif env == 'forest':
        args.trainconfig.testing = ["Reconstruct_2022-07-19-19-02-15_0"] # 96
        args.trainconfig.training = ["Reconstruct_2022-07-21-10-47-29_0"] # 118
    elif env == 'grassland':
        args.trainconfig.testing = ["Reconstruct_2022-07-18-20-34-01_0"] # 112
        args.trainconfig.training = ["Reconstruct_2022-07-19-20-06-22_0"] #214
    else:
        print('No such dataset available!')
        return
    
    if(loader == "test"):
        data_loader = DepthDataLoader(args, 'online_eval').data
        data_loader_iter = iter(data_loader)
        test_loader_iter = data_loader_iter
    elif(loader == "train"):
        data_loader = DepthDataLoader(args, 'train').data
        data_loader_iter = iter(data_loader)
        train_loader_iter = data_loader_iter
    
    # sample_num = len(data_loader_iter)
    sample_num = 0
    ASSIoUs = 0
    AOIoUs = 0
    SGIoUs = 0
    RMSEs = 0
    RELs = 0
    RMSElogs = 0
    total_pixel_num = 0
    rawRMSEs = 0
    rawRELs = 0
    rawRMSElogs = 0
    err_list = np.array([])
    err_list_raw = np.array([])
    depth_list = np.array([])
    for i, sample in enumerate(tqdm(data_loader_iter)):
        # print(i)
        # inputimg = np.moveaxis(sample["image"][0][:3, :, :].numpy(), 0, 2)
        # inputpc = np.moveaxis(sample["image"][0][3:, :, :].numpy(), 0, 2)
        input_ = sample["image"].to(device)
        pc_image = sample["pc_image"].to(device)
        traj_label = sample['depth'].to(device)
        mask_gt = (sample["mask_gt"]==1).to(device)
        # For Ablation study
        # input_[:, 0:3, :, :] = 0
        # Predict
        if (args.modelconfig.ablation == 'onlyPC'):
            input_[:, 0:3] = 0        
        elif (args.modelconfig.ablation == 'onlyRGB'):
            input_[:, 3] = 0
        if(model.use_adabins):
            bins, images = model(input_)
        else:
            pred = model(input_)
        
        # pred[:, 2:][pred[:, 2:]<0]=0
        if args.trainconfig.sprase_traj_mask == True:
            pred[:, 2:] += (pc_image).to(device)
        mask_pred = (pred[:, 1:2] > pred[:, 0:1])
        mask_gt_self = (traj_label>0) & (traj_label<depth_limit)

        if mask_gt_self.sum()<=0:
            continue

        # Compute IoU for annotated labels
        TP = (mask_gt & mask_pred).sum()    
        TN = (mask_gt & (~mask_pred)).sum()
        FP = ((~mask_gt) & mask_pred).sum()
        FN = ((~mask_gt) & (~mask_pred)).sum()

        ASSIoU = TP / (TP + TN + FP)
        AOIoU = FN / (FP + TN + FN)
        
        ASSIoUs += ASSIoU
        AOIoUs += AOIoU

        # Compute IoU for self generated labels
        TP_self = (mask_gt_self & mask_pred).sum()
        SGIoU = TP_self/(mask_gt_self.sum())
        SGIoUs += SGIoU


        # Compute RMSE 
        mask_de = mask_gt_self & (pc_image>0) & (pc_image<depth_limit) & (sample["depth_variance"]<1).to(device)
        # if ((pred[:, 2:] - sample['depth'])[mask_de]/sample['depth'][mask_de]).sum()>0:
        #     continue
        # RMSE_list = ((traj_label - pred[:, 2:])[mask_de])**2
        err_list = np.append(err_list, (traj_label - pred[:, 2:])[mask_de].detach().cpu().numpy())
        depth_list = np.append(depth_list, traj_label[mask_de].detach().cpu().numpy())
        RMSE = (((traj_label - pred[:, 2:])[mask_de])**2).sum()
        RMSElog = (((torch.log10(traj_label[mask_de]) - torch.log10(pred[:, 2:][mask_de]))).abs()).sum()
        RMSEs += RMSE
        RMSElogs += RMSElog
        total_pixel_num += mask_de.sum()

        # REL = ((pred[:, 2:] - traj_label)[mask_de]/traj_label[mask_de]).sum()
        REL = ((pred[:, 2:] - traj_label)[mask_de]/traj_label[mask_de]).abs().sum()
        RELs += REL
        # print("pred: ", REL)

        # Raw pc
        pred[:, 2:] = sample["pc_image"]
        err_list_raw = np.append(err_list_raw, (traj_label - pred[:, 2:])[mask_de].detach().cpu().numpy())
        rawRMSE = (((traj_label - pred[:, 2:])[mask_de])**2).sum()
        rawRMSElog = (((torch.log10(traj_label[mask_de]) - torch.log10(pred[:, 2:][mask_de]))).abs()).sum()
        rawRMSEs += rawRMSE
        rawRMSElogs += rawRMSElog

        rawREL = ((pred[:, 2:] - traj_label)[mask_de]/traj_label[mask_de]).abs().sum()
        rawRELs += rawREL

        sample_num += 1
        # print("raw: ", rawREL)
        # if(i>10):
        #     break

    torch.set_printoptions(precision=3)
    if(print_result):
        # print(ASSIoUs/sample_num, AOIoUs/sample_num, SGIoUs/sample_num,
        #         "RMSE: ", torch.sqrt(RMSEs/total_pixel_num),
        #         "REL: ", RELs/total_pixel_num,
        #         "RMSELog: ", RMSElogs/total_pixel_num)
        print("{:.3f} & {:.3f} & {:.3f}".format(ASSIoUs/sample_num, AOIoUs/sample_num, SGIoUs/sample_num))
        print("{:.3f} & {:.3f} & {:.3f}".format(torch.sqrt(RMSEs/total_pixel_num), RELs/total_pixel_num, RMSElogs/total_pixel_num))

        # print(ASSIoUs/sample_num, AOIoUs/sample_num, SGIoUs/sample_num,
        #         "rawRMSE: ", torch.sqrt(rawRMSEs/total_pixel_num),
        #         "rawREL: ", rawRELs/total_pixel_num,
        #         "rawRMSELog: ", rawRMSElogs/total_pixel_num)
        print("{:.3f} & {:.3f} & {:.3f}".format(torch.sqrt(rawRMSEs/total_pixel_num), rawRELs/total_pixel_num, rawRMSElogs/total_pixel_num))
    
            # pred_origin = pred[:, 2:]
            # pred = pred[:, 2:].clone()
            # pred[~mask_weight] = sample["pc_image"][~mask_weight]
    
    return depth_list, err_list, err_list_raw            
        



def vis_network_structure():
    data_loader = DepthDataLoader(args, 'train').data
    writer = SummaryWriter('.visulization/tmpvis')
    sample = next(iter(data_loader))
    model = model_list[0]
    writer.add_graph(model, sample["image"])
    writer.close()


"""
Load the parameters from the 
"""


def load_param_from_path(data_path):
    model_cfg = YAML().load(open(os.path.join(data_path, "ModelConfig.yaml"), 'r'))
    train_cfg = YAML().load(open(os.path.join(data_path, "TrainConfig.yaml"), 'r'))
    return model_cfg, train_cfg


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--models", default="/home/anqiao/tmp/semantic_front_end_filter/adabins/checkpoints/2023-02-28-12-00-40_fixed/UnetAdaptiveBins_latest.pt")
        # "--models", default="/home/anqiao/tmp/semantic_front_end_filter/adabins/checkpoints/2023-02-28-13-58-27_onlyRGB/UnetAdaptiveBins_latest.pt")
        # "--models", default="/home/anqiao/tmp/semantic_front_end_filter/adabins/checkpoints/2023-02-28-13-59-02_onlyPC/UnetAdaptiveBins_latest.pt")
    parser.add_argument("--names", default="")
    parser.add_argument(
        "--outdir", default="/home/anqiao/tmp/semantic_front_end_filter/adabins/checkpoints/2023-03-02-18-15-59/results_best_test")
    parser.add_argument('--gpu', default=None, type=int,
                        help='Which gpu to use')
    parser.add_argument("--name", default="UnetAdaptiveBins")
    parser.add_argument("--distributed", default=False,
                        action="store_true", help="Use DDP if set")
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--resume", default='', type=str,
                        help="Resume from checkpoint")
    parser.add_argument("--tqdm", default=False,
                        action="store_true", help="show tqdm progress bar")

    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='', type=str,
                        help="Wandb tags, seperate by `,`")
    # model_path = "/home/anqiao/tmp/semantic_front_end_filter/checkpoints/2022-10-27-18-43-00"
    # model_cfg = yaml.load(open(os.path.join(os.path.dirname(model_path), "ModelConfig.yaml"), 'r'), Loader=yaml.FullLoader)
    # model_cfg["input_channel"] = 4

    args = parse_args(parser)
    # args.data_path = "/home/anqiao/catkin_ws/SA_dataset/extract_trajectories_test"
    args.data_path = "/media/anqiao/Semantic/Data/extract_trajectories_007_Italy_Anomaly_clean/extract_trajectories"
    # args.data_path = "/media/anqiao/Semantic/Data/extract_trajectories_006_Italy/extract_trajectories"

    # if not os.path.exists(args.outdir):
    #     os.makedirs(args.outdir)
    args.trainconfig.bs = 3
    args.batch_size = 5
    try:
        # checkpoint_paths = sys.argv[1:]
        checkpoint_paths = args.models.split(" ")
        print(checkpoint_paths)
    except Exception as e:
        print(e)
        print("Usage: python vismodel checkpoint_path")

    # model_cfgs = [load_param_from_path(os.path.dirname(
    #     checkpoint_path)) for checkpoint_path in checkpoint_paths]
    model_cfg, train_cfg = load_param_from_path(os.path.dirname(args.models))    

    args.modelconfig.ablation = model_cfg['ablation']
    args.trainconfig.sprase_traj_mask = train_cfg['sprase_traj_mask']
    
    model = models.UnetAdaptiveBins.build(**model_cfg) 
    model = model_io.load_checkpoint(args.models, model)[0]
    model = model.to(device)
    # model_list = [models.UnetAdaptiveBins.build(**cfg) for cfg in model_cfgs]
    # names_list = args.names.split(" ")
    # loads = [model_io.load_checkpoint(checkpoint_path, model)
    #          for checkpoint_path, model in zip(checkpoint_paths, model_list)]
    # model,opt,epoch = model_io.load_checkpoint(checkpoint_path ,model)
    # model_list = [l[0] for l in loads]
    # for model in model_list:
    #     model.transform()

    # for i in range(20):
    #     vis_one("test", figname=os.path.join(args.outdir, "%d" % i))
    #     plt.savefig(os.path.join(args.outdir, "%d.jpg" % i))
        # plt.show()
    # vis_network_structure()
    print(args.modelconfig.ablation, "skip_connection: ", args.trainconfig.sprase_traj_mask)
    dg, eg, eg_raw = computeIoUs(model, args, loader='test', env='grassland')
    dh, eh, eh_raw = computeIoUs(model, args, loader='test', env='high grass')
    df, ef, ef_raw = computeIoUs(model, args, loader='test', env='forest')
    print("{:.2f}".format(((np.abs(eg)/dg).mean() + (np.abs(eh)/dh).mean() + np.abs(ef/df).mean())/3))
    print("{:.2f}".format(((np.abs(eg_raw)/dg).mean() + (np.abs(eh_raw)/dh).mean() + np.abs(ef_raw/df).mean())/3))
    # print("{:.2f}".format((np.abs(eg_raw).mean() + np.abs(eh_raw).mean() + np.abs(ef_raw).mean())/3))
    # print("{:.2f}".format((eg_raw.abs().mean() + eh_raw.abs().mean() + ef_raw.abs().mean())/3))
