import argparse
import os
import time
from datetime import datetime


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from simple_parsing import ArgumentParser
from ruamel.yaml import dump, RoundTripDumper

from semantic_front_end_filter.utils.file_util import save_checkpoint
from semantic_front_end_filter.models import UnetAdaptiveBins
from semantic_front_end_filter.cfgs import ModelConfig, TrainConfig, parse_args, dataclass, asdict
from semantic_front_end_filter.utils.dataloader import DepthDataLoader
from semantic_front_end_filter.utils.train_util import RunningAverage, colorize, RunningAverageDict, compute_errors


class UncertaintyLoss(nn.Module):  # Add variance to loss
    def __init__(self, train_args):
        super(UncertaintyLoss, self).__init__()
        self.name = 'SILog'
        self.args = train_args
        self.depth_variance_ratio = self.args.traj_distance_variance_ratio

    def forward(self, input, target, target_variance, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
            target_variance = target_variance[mask]
        if(target_variance.numel() !=0):
            Dg = torch.sum(0.5 * torch.pow(input - target, 2)/(1e-3 + target*self.depth_variance_ratio + target_variance))
            if(self.args.scale_loss_with_point_number):
                Dg /= input.shape[0]
        else:
            Dg = torch.tensor(0.).to('cuda')
        return Dg



def train_loss(args, criterion_depth, criterion_mask, pred, depth, depth_var, pc_image, image, mask_gt):
    # Only apply l_mask and l_mask_regulation
    l_mask = torch.tensor(0).to('cuda')
    l_mask_regulation = torch.tensor(0).to('cuda')

    if(args.trainconfig.sprase_traj_mask):
        masktraj = (depth > args.min_depth) & (depth < args.max_depth) & (pc_image > 1e-9)
    else:
        masktraj = (depth > args.min_depth) & (depth < args.max_depth)
    depth[~masktraj] = 0.
    # Apply traj mask

    # Apply anomaly mask    
    l_mask = criterion_mask(pred[:, 0:2], mask_gt.squeeze(dim=1).long())
    l_mask_regulation = criterion_mask(pred[:, 0:2], masktraj.squeeze(dim=1).long())
    
    mask_weight = (pred[:, 1:2] > pred[:, :1]).long()
    mask_soft = nn.functional.softmax(pred[:, 0:2], dim = 1)

    pred = pred[:, 2:]
    l_dense = args.trainconfig.traj_label_W * criterion_depth(pred, depth, depth_var, mask=masktraj.to(torch.bool), interpolate=True)
    mask0 = depth < 1e-9 # the mask of places with on label
    maskpc = mask0 & (pc_image > 1e-9) & (pc_image < args.max_pc_depth) # pc image have label
    depth_var_pc = depth_var if args.trainconfig.pc_label_uncertainty else torch.ones_like(depth_var)
    l_pc = args.trainconfig.pc_image_label_W * criterion_depth(pred, pc_image, depth_var_pc, mask=maskpc.to(torch.bool), interpolate=True)
    
    print("MASK_L: ",l_mask.item(), "SSL: ", l_dense.item())
    return l_dense, l_mask, l_mask_regulation, masktraj, maskpc


def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None,
          optimizer_state_dict=None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_loader = DepthDataLoader(args, 'train').data
    test_loader = DepthDataLoader(args, 'online_eval').data
    
    ###################################### losses ##############################################
    criterion_depth = UncertaintyLoss(args.trainconfig)
    criterion_mask = nn.CrossEntropyLoss(weight=torch.tensor([args.trainconfig.pc_image_label_W, args.trainconfig.traj_label_W_4mask]).to('cuda').float())
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
    step_count = 0
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
        print("EPOCH:", epoch)
        time_core = 0.
        time_total = -time.time()
        train_metrics = RunningAverageDict()
        ################################# Train loop ##########################################################
        for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                             total=len(train_loader)) if args.tqdm else enumerate(train_loader):

            time_core -= time.time()
            optimizer.zero_grad()

            img = batch['image'].to(device)
            # img[:, 3] = 0
            depth = batch['depth'].to(device)
            depth_var = batch['depth_variance'].to(device)
            mask_gt = batch['mask_gt'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue

            if(args.modelconfig.ablation == "onlyPC"):
                img[:, 0:3] = 0
            elif(args.modelconfig.ablation == "onlyRGB"):
                img[:, 3] = 0    
            pred = model(img)
            pc_image = batch["pc_image"].to(device)

            if(args.trainconfig.sprase_traj_mask):
                pred[:, 2:] = pred[:, 2:] + pc_image # with or withour pc_image
            else:
                pred[:, 2:] = pred[:, 2:]
            if(pred.shape != depth.shape): # need to enlarge the output prediction
                pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='nearest')
            l_dense, l_mask, l_mask_regulation, masktraj, maskpc = train_loss(args, criterion_depth, criterion_mask, pred, depth, depth_var, pc_image, img, mask_gt)
            if(pred.shape[1]==2):
                mask_weight = pred[:, 1:, :, :]
                pred = mask_weight * pred[:, :1, :, :] + (1-mask_weight)*pc_image[:, 0:, :, :]
            elif (pred.shape[1]==3):
                mask_weight = (pred[:, 1:2]>pred[:, 0:1])
                pred =  pred[:, 2:].clone()
                pred[~mask_weight] = pc_image[~mask_weight]

            loss = args.trainconfig.mask_loss_W*l_mask + l_dense

            pred[pred < args.min_depth] = args.min_depth
            max_depth_gt = max(args.max_depth, args.max_pc_depth)
            pred[pred > max_depth_gt] = max_depth_gt
            pred[torch.isinf(pred)] = max_depth_gt
            pred[torch.isnan(pred)] = args.min_depth

            pred = pred.detach()
            train_metrics.update(compute_errors(depth[masktraj].cpu(), pred[masktraj].cpu(), 'traj/'), depth[masktraj].shape[0])

            writer.add_scalar("Loss/train/l_sum", loss/args.batch_size, global_step=epoch*len(train_loader)+i)
            writer.add_scalar("Loss/train/l_dense", l_dense/args.batch_size, global_step=epoch*len(train_loader)+i)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()

            step_count += 1
            scheduler.step()

            time_core += time.time()
            ########################################################################################################

            if step_count % args.trainconfig.validate_every == 0:

                ################################# Validation loop ##################################################
                model.eval()
                metrics, val_si = validate(args, model, test_loader, criterion_depth, criterion_mask, epoch, epochs, device, step_count)
                [writer.add_scalar("test/"+k, v, step_count) for k,v in metrics.items()]
                print("Validated: {}".format(metrics))
                save_checkpoint(model, optimizer, epoch, f"{experiment_name}_latest.pt",
                                            root=data_dir)
                                        
                print(f"Total time spent: {time_total+time.time()}, core time spent:{time_core}")
                time_total = -time.time()
                time_core = 0.
                if metrics['traj/abs_rel'] < best_loss:
                    save_checkpoint(model, optimizer, epoch, f"{experiment_name}_best.pt",
                                             root=data_dir)
                    best_loss = metrics['traj/abs_rel']
                model.train()
                #################################################################################################
    return model


def validate(args, model, test_loader, criterion_depth, criterion_mask, epoch, epochs, device='cpu', step = 0):
    with torch.no_grad():
        val_si = RunningAverage()
        metrics = RunningAverageDict()
        for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation") if args.tqdm else test_loader:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            depth_var = batch['depth_variance'].to(device)
            mask_gt = batch['mask_gt'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue

            if(args.modelconfig.ablation == "onlyPC"):
                img[:, 0:3] = 0
            elif(args.modelconfig.ablation == "onlyRGB"):
                img[:, 3] = 0 
            pred = model(img)
            pc_image = batch["pc_image"].to(device)
            if(args.trainconfig.sprase_traj_mask):
                pred[:, 2:] = pred[:, 2:] + pc_image # with or withour pc_image
            else:
                pred[:, 2:] = pred[:, 2:]
            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='nearest')
            l_dense, l_mask, l_mask_regulation, masktraj, maskpc = train_loss(args, criterion_depth, criterion_mask, pred, depth, depth_var, pc_image, img, mask_gt)
            if(pred.shape[1]==2):
                mask_weight = pred[:, 1:, :, :]
                pred = mask_weight * pred[:, :1, :, :] + (1-mask_weight)*pc_image[:, 0:, :, :]
            elif (pred.shape[1]==3):
                mask_weight = (pred[:, 1:2]>pred[:, 0:1])
                pred =  pred[:, 2:].clone()
                pred[~mask_weight] = pc_image[~mask_weight]
            loss = args.trainconfig.mask_loss_W*l_mask + l_dense

            writer.add_scalar("Loss/test/l_sum", loss, global_step=epoch)
            writer.add_scalar("Loss/test/l_dense", l_dense, global_step=epoch)

            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            depth_var = depth_var.squeeze().unsqueeze(0).unsqueeze(0)
            mask = depth > args.min_depth
            val_si.append(loss.item())

            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            max_depth_gt = max(args.max_depth_eval, args.max_pc_depth)
            pred[pred > max_depth_gt] = max_depth_gt
            pred[np.isinf(pred)] = max_depth_gt
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            pc_image = pc_image.squeeze().cpu().numpy()
            masktraj = masktraj.squeeze().squeeze()
            maskpc = maskpc.squeeze().squeeze()
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
            valid_mask_traj = np.logical_and(valid_mask, masktraj.cpu().numpy())
            valid_mask_pc = np.logical_and(eval_mask, maskpc.cpu().numpy()) 
            if(not (valid_mask_traj.any() & valid_mask_pc.any())): continue
            metrics.update(compute_errors(gt_depth[valid_mask_traj], pred[valid_mask_traj], 'traj/'), gt_depth[valid_mask_traj].shape[0])

        return metrics.get_value(), val_si


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)



if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--name", default="UnetAdaptiveBins")
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")
    parser.add_argument("--tqdm", default=False, action="store_true", help="show tqdm progress bar")


    args = parse_args(parser, flatten = True)

    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    args.num_workers = args.trainconfig.workers
    
    saver_dir = os.path.join(args.root,"checkpoints")
    data_dir = os.path.join(saver_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(data_dir)
    with open(os.path.join(data_dir, "args.txt"), "w") as f:
        for key, value in args.__dict__.items():
            f.write(key + ': ' + str(value) + '\n')

    for cfg in [TrainConfig(**vars(args.trainconfig)), 
                ModelConfig(**vars(args.modelconfig))]:
        dict_data = asdict(cfg)
        name = type(cfg).__name__ + ".yaml"
        with open(os.path.join(data_dir, name), 'w') as f:
            dump(dict_data, f, Dumper=RoundTripDumper)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    writer = SummaryWriter(log_dir=data_dir, flush_secs=60)

    model = UnetAdaptiveBins.build(**asdict(args.modelconfig))
    model = model.to(device)

    args.epoch = 0
    args.last_epoch = -1
    
    train(model, args, epochs=args.trainconfig.epochs, lr=args.trainconfig.lr, device=device, root=args.root,
          experiment_name=args.name, optimizer_state_dict=None)
