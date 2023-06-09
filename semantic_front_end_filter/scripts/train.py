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

from semantic_front_end_filter.utils.file_util import save_checkpoint
from semantic_front_end_filter.models import UnetAdaptiveBins
from semantic_front_end_filter.cfgs import ModelConfig, TrainConfig, parse_args, dataclass, asdict
from semantic_front_end_filter.utils.dataloader import DepthDataLoader
from semantic_front_end_filter.utils.loss_util import EdgeAwareLoss, SILogLoss, BinsChamferLoss, UncertaintyLoss, ConsistencyLoss, MaskLoss
from semantic_front_end_filter.utils.train_util import RunningAverage, colorize, RunningAverageDict, compute_errors


DTSTRING = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
PROJECT = "semantic_front_end_filter-Anomaly"
logging = True

count_val = 0


def train_loss(args, criterion_ueff, criterion_bins, criterion_edge, criterion_consistency, criterion_mask, pred, bin_edges, depth, depth_var, pc_image, image, pose, mask_gt):
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
    l_dense = args.trainconfig.traj_label_W * criterion_ueff(pred, depth, depth_var, mask=masktraj.to(torch.bool), interpolate=True)
    mask0 = depth < 1e-9 # the mask of places with on label
    maskpc = mask0 & (pc_image > 1e-9) & (pc_image < args.max_pc_depth) # pc image have label
    depth_var_pc = depth_var if args.trainconfig.pc_label_uncertainty else torch.ones_like(depth_var)
    l_pc = args.trainconfig.pc_image_label_W * criterion_ueff(pred, pc_image, depth_var_pc, mask=maskpc.to(torch.bool), interpolate=True)
    l_edge = criterion_edge(pred, image, interpolate = True)
    if bin_edges is not None and args.trainconfig.w_chamfer > 0:
        l_chamfer = criterion_bins(bin_edges, depth)
    else:
        l_chamfer = torch.Tensor([0]).to(l_dense.device)
    
    l_consis = criterion_consistency(pred, pose) if args.trainconfig.consistency_W > 1e-3 else torch.tensor(0.).to('cuda')
    print("MASK_L: ",l_mask.item(), "Mask_R", args.trainconfig.mask_regulation_W * l_mask_regulation.item(), "SSL: ", l_dense.item())
    return l_dense, l_chamfer, l_edge, l_consis, l_mask, l_mask_regulation, masktraj, maskpc


def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None,
          optimizer_state_dict=None):
    global PROJECT
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_loader = DepthDataLoader(args, 'train').data
    test_loader = DepthDataLoader(args, 'online_eval').data
    
    ###################################### losses ##############################################
    # criterion_ueff = SILogLoss()
    criterion_ueff = UncertaintyLoss(args.trainconfig)
    criterion_bins = BinsChamferLoss() if args.chamfer else None
    criterion_edge = EdgeAwareLoss(args.trainconfig)
    criterion_consistency = ConsistencyLoss(args.trainconfig)
    # criterion_mask = MaskLoss(args.trainconfig)
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
            if(args.modelconfig.use_adabins):
                bin_edges, pred = model(img)
            else:
                if(args.modelconfig.ablation == "onlyPC"):
                    img[:, 0:3] = 0
                elif(args.modelconfig.ablation == "onlyRGB"):
                    img[:, 3] = 0    
                bin_edges, pred = None, model(img)
            pc_image = batch["pc_image"].to(device)

            if(args.trainconfig.sprase_traj_mask):
                pred[:, 2:] = pred[:, 2:] + pc_image # with or withour pc_image
            else:
                pred[:, 2:] = pred[:, 2:]
            if(pred.shape != depth.shape): # need to enlarge the output prediction
                pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='nearest')
            l_dense, l_chamfer, l_edge, l_consis,l_mask, l_mask_regulation, masktraj, maskpc = train_loss(args, criterion_ueff, criterion_bins, criterion_edge, criterion_consistency, criterion_mask, pred, bin_edges, depth, depth_var, pc_image, img, batch['pose'], mask_gt)
            if(pred.shape[1]==2):
                mask_weight = pred[:, 1:, :, :]
                pred = mask_weight * pred[:, :1, :, :] + (1-mask_weight)*pc_image[:, 0:, :, :]
            elif (pred.shape[1]==3):
                mask_weight = (pred[:, 1:2]>pred[:, 0:1])
                pred =  pred[:, 2:].clone()
                pred[~mask_weight] = pc_image[~mask_weight]

            loss = args.trainconfig.mask_loss_W*l_mask + args.trainconfig.mask_regulation_W *l_mask_regulation + l_dense

            pred[pred < args.min_depth] = args.min_depth
            max_depth_gt = max(args.max_depth, args.max_pc_depth)
            pred[pred > max_depth_gt] = max_depth_gt
            pred[torch.isinf(pred)] = max_depth_gt
            pred[torch.isnan(pred)] = args.min_depth

            pred = pred.detach().cpu()
            depth = depth.cpu()
            pc_image = pc_image.cpu()
            train_metrics.update(compute_errors(depth[masktraj], pred[masktraj], 'traj/'), depth[masktraj].shape[0])



            writer.add_scalar("Loss/train/l_chamfer", l_chamfer/args.batch_size, global_step=epoch*len(train_loader)+i)
            writer.add_scalar("Loss/train/l_sum", loss/args.batch_size, global_step=epoch*len(train_loader)+i)
            writer.add_scalar("Loss/train/l_dense", l_dense/args.batch_size, global_step=epoch*len(train_loader)+i)
            writer.add_scalar("Loss/train/l_edge", l_edge/args.batch_size, global_step=epoch*len(train_loader)+i)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()
            if step_count % 5 == 0:
                # wandb.log({f"Loss/train/{criterion_ueff.name}": l_dense.item()/args.batch_size}, step=step_count)
                wandb.log({f"Loss/train/{criterion_ueff.name}": l_dense.item()/args.batch_size}, step=step_count)
                wandb.log({f"Loss/train/{criterion_bins.name}": l_chamfer.item()/args.batch_size}, step=step_count)
                wandb.log({f"Loss/train/{criterion_edge.name}": l_edge.item()/args.batch_size}, step=step_count)
                wandb.log({f"Loss/train/MASKLoss": args.trainconfig.mask_loss_W * l_mask.item()/args.batch_size}, step=step_count)
                wandb.log({f"Loss/train/RegulationMask": args.trainconfig.mask_regulation_W * l_mask_regulation.item()/args.batch_size}, step=step_count)
                wandb.log({"Loss/train/l_sum": loss/args.batch_size}, step=step_count)

            step_count += 1
            scheduler.step()

            time_core += time.time()
            ########################################################################################################

            if step_count % args.trainconfig.validate_every == 0:

                ################################# Validation loop ##################################################
                model.eval()
                metrics, val_si = validate(args, model, test_loader, criterion_ueff, criterion_bins, criterion_edge, criterion_consistency, criterion_mask, epoch, epochs, device, step_count)
                [writer.add_scalar("test/"+k, v, step_count) for k,v in metrics.items()]
                print("Validated: {}".format(metrics))
                save_checkpoint(model, optimizer, epoch, f"{experiment_name}_latest.pt",
                                            root=saver.data_dir)
                                        
                print(f"Total time spent: {time_total+time.time()}, core time spent:{time_core}")
                time_total = -time.time()
                time_core = 0.
                if metrics['traj/abs_rel'] < best_loss:
                    save_checkpoint(model, optimizer, epoch, f"{experiment_name}_best.pt",
                                             root=saver.data_dir)
                    best_loss = metrics['traj/abs_rel']
                model.train()
                #################################################################################################
        wandb.log({f"train/{k}": v for k, v in train_metrics.get_value().items()}, step=step_count)
    return model


def validate(args, model, test_loader, criterion_ueff, criterion_bins, criterion_edge, criterion_consistency, criterion_mask, epoch, epochs, device='cpu', step = 0):
    global count_val
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
            if(args.modelconfig.use_adabins):
                bin_edges, pred = model(img)
            else:
                if(args.modelconfig.ablation == "onlyPC"):
                    img[:, 0:3] = 0
                elif(args.modelconfig.ablation == "onlyRGB"):
                    img[:, 3] = 0 
                bin_edges, pred = None, model(img)
                # bin_edges, pred = None, model(img[: ,0:3])
            pc_image = batch["pc_image"].to(device)
            if(args.trainconfig.sprase_traj_mask):
                pred[:, 2:] = pred[:, 2:] + pc_image # with or withour pc_image
            else:
                pred[:, 2:] = pred[:, 2:]
            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='nearest')
            l_dense, l_chamfer, l_edge, l_consis, l_mask, l_mask_regulation, masktraj, maskpc = train_loss(args, criterion_ueff, criterion_bins, criterion_edge, criterion_consistency, criterion_mask, pred, bin_edges, depth, depth_var, pc_image, img, batch['pose'], mask_gt)
            if(pred.shape[1]==2):
                mask_weight = pred[:, 1:, :, :]
                pred = mask_weight * pred[:, :1, :, :] + (1-mask_weight)*pc_image[:, 0:, :, :]
            elif (pred.shape[1]==3):
                mask_weight = (pred[:, 1:2]>pred[:, 0:1])
                pred =  pred[:, 2:].clone()
                pred[~mask_weight] = pc_image[~mask_weight]
            loss = args.trainconfig.mask_loss_W*l_mask + args.trainconfig.mask_regulation_W *l_mask_regulation + l_dense

            writer.add_scalar("Loss/test/l_chamfer", l_chamfer, global_step=count_val)
            writer.add_scalar("Loss/test/l_sum", loss, global_step=count_val)
            writer.add_scalar("Loss/test/l_dense", l_dense, global_step=count_val)

            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            depth_var = depth_var.squeeze().unsqueeze(0).unsqueeze(0)
            mask = depth > args.min_depth
            count_val = count_val + 1
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
    data_dir = os.path.join(saver_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(data_dir)
    with open(os.path.join(data_dir, "args.txt"), "w") as f:
        for key, value in args.__dict__.items():
            f.write(key + ': ' + str(value) + '\n')

    for cfg in [TrainConfig(**vars(args.trainconfig)), 
                ModelConfig(**vars(args.modelconfig))]:
        assert is_dataclass(cfg)
        dict_data = asdict(cfg)
        name = type(cfg).__name__ + ".yaml"
        with open(os.path.join(data_dir, name), 'w') as f:
            dump(dict_data, f, Dumper=RoundTripDumper)


    writer = SummaryWriter(log_dir=saver.data_dir, flush_secs=60)

    model = UnetAdaptiveBins.build(**asdict(args.modelconfig))

    args.epoch = 0
    args.last_epoch = -1
    train(model, args, epochs=args.trainconfig.epochs, lr=args.trainconfig.lr, device=args.gpu, root=args.root,
          experiment_name=args.name, optimizer_state_dict=None)
