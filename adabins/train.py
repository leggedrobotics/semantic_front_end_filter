import argparse
import os
import sys
import uuid
from datetime import datetime as dt

import numpy as np
from scipy import ndimage # this has to happend before import torch on euler
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
# import wandb
from tqdm import tqdm

import model_io
import models
import utils
from dataloader import DepthDataLoader
from loss import SILogLoss, BinsChamferLoss, UncertaintyLoss
from utils import RunningAverage, colorize
from simple_parsing import ArgumentParser
from cfg import TrainConfig, ModelConfig
from experimentSaver import ConfigurationSaver
from torch.utils.tensorboard import SummaryWriter
import time

# os.environ['WANDB_MODE'] = 'dryrun'
PROJECT = "MDE-AdaBins"
logging = True


def is_rank_zero(args):
    return args.rank == 0


import matplotlib


def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img


def log_images(img, depth, pred, args, step):
    depth = colorize(depth, vmin=args.min_depth, vmax=args.max_depth)
    pred = colorize(pred, vmin=args.min_depth, vmax=args.max_depth)
    # wandb.log(
    #     {
    #         "Input": [wandb.Image(img)],
    #         "GT": [wandb.Image(depth)],
    #         "Prediction": [wandb.Image(pred)]
    #     }, step=step)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ###################################### Load model ##############################################

    model = models.UnetAdaptiveBins.build(n_bins=args.modelconfig.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                          norm=args.modelconfig.norm)

    ## Load pretrained kitti
    # model,_,_ = model_io.load_checkpoint("./pretrained/AdaBins_kitti.pt", model)

    ################################################################################################

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.batch_size = 8
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(args.gpu, args.rank, args.batch_size, args.workers)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    train(model, args, epochs=args.trainconfig.epochs, lr=args.trainconfig.lr, device=args.gpu, root=args.root,
          experiment_name=args.name, optimizer_state_dict=None)


def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None,
          optimizer_state_dict=None):
    global PROJECT
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ###################################### Logging setup #########################################
    print(f"Training {experiment_name}")

    should_write = ((not args.distributed) or args.rank == 0)
    should_log = should_write and logging

    ################################################################################################

    train_loader = DepthDataLoader(args, 'train').data
    test_loader = DepthDataLoader(args, 'online_eval').data
    
    ###################################### losses ##############################################
    # criterion_ueff = SILogLoss()
    criterion_ueff = UncertaintyLoss()
    criterion_bins = BinsChamferLoss() if args.chamfer else None
    ################################################################################################

    model.train()

    ###################################### Optimizer ################################################
    if args.trainconfig.same_lr:
        print("Using same LR")
        params = model.parameters()
    else:
        print("Using diff LR")
        m = model.module if args.multigpu else model
        params = [{"params": m.get_1x_lr_params(), "lr": lr / 10},
                  {"params": m.get_10x_lr_params(), "lr": lr}]

    optimizer = optim.AdamW(params, weight_decay=args.trainconfig.wd, lr=args.trainconfig.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    ################################################################################################
    # some globals
    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf

    ###################################### Scheduler ###############################################
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
                                              div_factor=args.trainconfig.div_factor,
                                              final_div_factor=args.trainconfig.final_div_factor)
    if args.resume != '' and scheduler is not None:
        scheduler.step(args.epoch + 1)
    ################################################################################################

    # max_iter = len(train_loader) * epochs
    for epoch in range(args.epoch, epochs):
        time_core = 0.
        time_total = -time.time()
        ################################# Train loop ##########################################################
        # if should_log: wandb.log({"Epoch": epoch}, step=step)
        for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                             total=len(train_loader)) if args.tqdm else enumerate(train_loader):

            time_core -= time.time()
            optimizer.zero_grad()

            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            depth_var = batch['depth_variance'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            bin_edges, pred = model(img)
            mask = (depth > args.min_depth) & (depth < args.max_depth)
            l_dense = criterion_ueff(pred, depth, depth_var, mask=mask.to(torch.bool), interpolate=True)
            mask0 = depth < 1e-9 # the mask of places with on label
            negativedepth = torch.zeros_like(depth)
            negativedepth[mask0] = depth[~mask0].min()/2
            l_dense += args.trainconfig.pc_min_depth_label_W * criterion_ueff(pred, negativedepth, depth_var, mask=mask0.to(torch.bool), interpolate=True)
            pc_image = batch["pc_image"].to(device)
            maskpc = mask0 & (pc_image > 1e-9) # pc image have label
            l_dense += args.trainconfig.pc_image_label_W * criterion_ueff(pred, pc_image,depth_var, mask=maskpc.to(torch.bool), interpolate=True)

            if args.trainconfig.w_chamfer > 0:
                l_chamfer = criterion_bins(bin_edges, depth)
            else:
                l_chamfer = torch.Tensor([0]).to(img.device)

            loss = l_dense + args.trainconfig.w_chamfer * l_chamfer
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()
            # if should_log and step % 5 == 0:
            #     wandb.log({f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)
            #     wandb.log({f"Train/{criterion_bins.name}": l_chamfer.item()}, step=step)

            step += 1
            scheduler.step()

            time_core += time.time()
            ########################################################################################################

            if should_write and step % args.trainconfig.validate_every == 0:

                ################################# Validation loop ##################################################
                model.eval()
                metrics, val_si = validate(args, model, test_loader, criterion_ueff, epoch, epochs, device)
                [writer.add_scalar("metrics/"+k, v, epoch*len(train_loader) + i*args.batch_size) for k,v in metrics.items()]
                # print("Validated: {}".format(metrics))
                if should_log:
                    # wandb.log({
                    #     f"Test/{criterion_ueff.name}": val_si.get_value(),
                    #     # f"Test/{criterion_bins.name}": val_bins.get_value()
                    # }, step=step)

                    # wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)
                    model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_latest.pt",
                                             root=saver.data_dir)
                                            
                    print(f"Total time spent: {time_total+time.time()}, core time spent:{time_core}")
                    time_total = -time.time()
                    time_core = 0.
                if metrics['abs_rel'] < best_loss and should_write:
                    model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_best.pt",
                                             root=saver.data_dir)
                    best_loss = metrics['abs_rel']
                model.train()
                #################################################################################################

    return model


def validate(args, model, test_loader, criterion_ueff, epoch, epochs, device='cpu'):
    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation") if is_rank_zero(
                args) else test_loader:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            depth_var = batch['depth_variance'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            depth_var = depth_var.squeeze().unsqueeze(0).unsqueeze(0)
            bins, pred = model(img)

            mask = depth > args.min_depth
            l_dense = criterion_ueff(pred, depth, depth_var, mask=mask.to(torch.bool), interpolate=True)
            val_si.append(l_dense.item())

            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            if args.trainconfig.garg_crop or args.trainconfig.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.trainconfig.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:

                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    
            valid_mask = np.logical_and(valid_mask, eval_mask)
            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_si


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)



# Arguments
parser = ArgumentParser()
parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
parser.add_argument("--name", default="UnetAdaptiveBins")
parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
parser.add_argument("--root", default=".", type=str,
                    help="Root folder to save data in")
parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")
parser.add_argument("--tqdm", default=False, action="store_true", help="show tqdm progress bar")

parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")

parser.add_arguments(TrainConfig, dest="trainconfig")
parser.add_arguments(ModelConfig, dest="modelconfig")

def parse_args():
    
    args = parser.parse_args()
    args.batch_size = args.trainconfig.bs
    args.num_threads = args.workers
    args.mode = 'train'
    args.data_path = args.trainconfig.data_path
    args.min_depth = args.modelconfig.min_depth
    args.max_depth = args.modelconfig.max_depth
    args.min_depth_eval = args.modelconfig.min_depth_eval
    args.max_depth_eval = args.modelconfig.max_depth_eval

    args.chamfer = args.trainconfig.w_chamfer > 0
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')

        args.world_size = len(nodes)
        args.rank = int(os.environ['SLURM_PROCID'])

    except KeyError as e:
        # We are NOT using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')

        print(args.rank)
        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(args.dist_url)
        args.dist_backend = 'nccl'
        args.gpu = None


    return args
if __name__ == '__main__':

    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node
    
    saver_dir = os.path.join(args.root,"checkpoints")
    saver = ConfigurationSaver(log_dir=saver_dir,
                            save_items=[os.path.realpath(__file__)],
                            args=args,
                            dataclass_configs=[TrainConfig(**vars(args.trainconfig)), 
                                ModelConfig(**vars(args.modelconfig))])
                
    writer = SummaryWriter(log_dir=saver.data_dir, flush_secs=60)

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
