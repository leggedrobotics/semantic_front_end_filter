import argparse
import os
import sys
import uuid
from matplotlib import image
import time
from datetime import datetime


import numpy as np
from scipy import ndimage # this has to happend before import torch on euler
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
# from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
from simple_parsing import ArgumentParser
import matplotlib.pyplot as plt

from . import model_io
from . import models
from . import utils
from .cfgUtils import parse_args, TrainConfig, ModelConfig, asdict
from .experimentSaver import ConfigurationSaver
from .dataloader import DepthDataLoader
from .loss import EdgeAwareLoss, SILogLoss, BinsChamferLoss, UncertaintyLoss
from .utils import RunningAverage, colorize

DTSTRING = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
PROJECT = "semantic_front_end_filter-adabins"
logging = True

count_val = 0

def is_rank_zero(args):
    return args.rank == 0


import matplotlib


def colorize(value, vmin=10, vmax=40, cmap='plasma'):
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


def log_images(samples, model, name, step, maxImages = 5, device = None, use_adabins = False):
    if(device is None):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # depth = colorize(depth, vmin=args.min_depth, vmax=args.max_depth)
    # pred = colorize(pred, vmin=args.min_depth, vmax=args.max_depth)
    # wandb.log(
    #     {
    #         "Input": [wandb.Image(img)],
    #         "GT": [wandb.Image(depth)],
    #         "Prediction": [wandb.Image(pred)]
    #     }, step=step)
    
    inputimgs = []
    trajlabels = []
    pclabels = []
    filepaths = []
    predictions = []
    trajerrors = []
    pcimgerrors = []
    errordistributions = []

    for i, sample in zip(range(maxImages), samples):

        inputimg = np.moveaxis(sample["image"][0][:3,:,:].numpy(),0,2)
        # inputimg = np.moveaxis(sample["image"][0].numpy(),0,2)
        inputimg.max(), inputimg.min()
        inputimg = (inputimg-inputimg.min())/(inputimg.max()- inputimg.min())
        inputimg = inputimg[:,:,::-1] # the changing between RGB and BGR
        inputimgs.append(wandb.Image(inputimg[:,:,:3]))
        filepaths.append(sample["path"][:1])
        depth = sample["depth"][0][0].numpy()
        trajlabels.append(wandb.Image(colorize(depth, vmin = 0, vmax=40)))

        pc_img = sample["pc_image"][0][0].numpy()
        pclabels.append(wandb.Image(colorize(pc_img,vmin = 0, vmax=40)))
        
        if(use_adabins):
            _, images = model(sample["image"][None,0,...].to(device))
        else:
            images = model(sample["image"][None,0,...].to(device))
        # bins, images = None, model(sample["image"])
        pred = images[0].detach()
        predictions.append(wandb.Image(colorize(pred[0].cpu().numpy(), vmin = 0, vmax=40)))
        
        pred = nn.functional.interpolate(pred[None,...], torch.tensor(depth).shape[-2:], mode='nearest')
        pred = pred[0][0].cpu().numpy()
        diff = pred- depth
        mask_traj = depth>1e-9
        diff[~mask_traj] = 0
        trajerrors.append(wandb.Image(colorize(diff,vmin = -5, vmax=5 )))

        pcdiff = pred- pc_img
        mask_pc = pc_img>1e-9
        pcdiff[~mask_pc] = 0
        pcimgerrors.append(wandb.Image(colorize(pcdiff, vmin = -5, vmax=5)))
        
        fig = plt.figure()
        plt.plot(pred[mask_pc].reshape(-1), pcdiff[mask_pc].reshape(-1), "x",ms=1,alpha = 0.2,label = "pc_img_err")
        plt.plot(pred[mask_traj&(pc_img>1e-9)].reshape(-1), diff[mask_traj&(pc_img>1e-9)].reshape(-1), "x",ms=1,alpha = 0.2, label = "traj_label_err")
        plt.plot(pred[mask_traj&(pc_img<1e-9)].reshape(-1), diff[mask_traj&(pc_img<1e-9)].reshape(-1), "x",ms=1,alpha = 0.2, label = "traj_unlabel_err")
        plt.title("err vs distance")
        plt.xlabel("depth_prediction")
        plt.ylabel("err")
        plt.ylim((-20,20))
        plt.legend()
        errordistributions.append(wandb.Image(fig))
        plt.cla()
        plt.close(fig)

    columns = ["filepaths","image"]
    data = [[f, img ] for f,img in zip(filepaths,inputimgs)]
    result_image_table = wandb.Table(columns = columns, data = data)
    # result_image_table.add_column("image", inputimgs)
    result_image_table.add_column("trajlabel", trajlabels)
    result_image_table.add_column("pclabel", pclabels)
    # result_image_table.add_column("filepath", filepaths)
    result_image_table.add_column("pred", predictions)
    result_image_table.add_column("traj_label_error", trajerrors)
    result_image_table.add_column("pc_label_error", pcimgerrors)
    result_image_table.add_column("distribution", errordistributions)

    wandb.log({name: result_image_table}, step=step)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ###################################### Load model ##############################################
    input_channel = 3 if args.load_pretrained else 4

    model = models.UnetAdaptiveBins.build(**asdict(args.modelconfig))

    ## Load pretrained kitti
    if args.load_pretrained:
        model,_,_ = model_io.load_checkpoint("./models/AdaBins_kitti.pt", model)
    
    if input_channel == 3:
        model.transform()
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

def train_loss(args, criterion_ueff, criterion_bins, criterion_edge, pred, bin_edges, depth, depth_var, pc_image, image):

    if(args.trainconfig.sprase_traj_mask):
        masktraj = (depth > args.min_depth) & (depth < args.max_depth) & (pc_image > 1e-9)
    else:
        masktraj = (depth > args.min_depth) & (depth < args.max_depth)
    depth[~masktraj] = 0.
    l_dense = args.trainconfig.traj_label_W * criterion_ueff(pred, depth, depth_var, mask=masktraj.to(torch.bool), interpolate=True)
    mask0 = depth < 1e-9 # the mask of places with on label
    maskpc = mask0 & (pc_image > 1e-9) & (pc_image < args.max_pc_depth) # pc image have label
    depth_var_pc = depth_var if args.trainconfig.pc_label_uncertainty else torch.ones_like(depth_var)
    l_dense += args.trainconfig.pc_image_label_W * criterion_ueff(pred, pc_image, depth_var_pc, mask=maskpc.to(torch.bool), interpolate=True)

    l_edge = criterion_edge(pred, image, interpolate = True)
    if bin_edges is not None and args.trainconfig.w_chamfer > 0:
        l_chamfer = criterion_bins(bin_edges, depth)
    else:
        l_chamfer = torch.Tensor([0]).to(l_dense.device)
    return l_dense, l_chamfer, l_edge, masktraj, maskpc


def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None,
          optimizer_state_dict=None):
    global PROJECT
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ###################################### Logging setup #########################################
    print(f"Training {experiment_name}")

    should_write = ((not args.distributed) or args.rank == 0)
    should_log = should_write and logging
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        wandb.init(project=PROJECT, name=DTSTRING+"_"+args.trainconfig.wandb_name, entity="semantic_front_end_filter", config=args, tags=tags, notes=args.notes)
        # wandb.init(mode="disabled", project=PROJECT, entity="semantic_front_end_filter", config=args, tags=tags, notes=args.notes)

        # wandb.watch(model)
    ################################################################################################

    train_loader = DepthDataLoader(args, 'train').data
    test_loader = DepthDataLoader(args, 'online_eval').data
    
    ###################################### losses ##############################################
    # criterion_ueff = SILogLoss()
    criterion_ueff = UncertaintyLoss(args.trainconfig)
    criterion_bins = BinsChamferLoss() if args.chamfer else None
    criterion_edge = EdgeAwareLoss(args.trainconfig)
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
        train_metrics = utils.RunningAverageDict()
        ################################# Train loop ##########################################################
        if should_log: wandb.log({"Epoch": epoch}, step=step_count)
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
            if(args.modelconfig.use_adabins):
                bin_edges, pred = model(img)
            else:
                bin_edges, pred = None, model(img)
            pc_image = batch["pc_image"].to(device)
            l_dense, l_chamfer, l_edge, masktraj, maskpc = train_loss(args, criterion_ueff, criterion_bins, criterion_edge, pred, bin_edges, depth, depth_var, pc_image, img)
            loss = l_dense + args.trainconfig.w_chamfer * l_chamfer + args.trainconfig.edge_aware_label_W * l_edge

            if(pred.shape != depth.shape): # need to enlarge the output prediction
                pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='nearest')

            pred[pred < args.min_depth] = args.min_depth
            max_depth_gt = max(args.max_depth, args.max_pc_depth)
            pred[pred > max_depth_gt] = max_depth_gt
            pred[torch.isinf(pred)] = max_depth_gt
            pred[torch.isnan(pred)] = args.min_depth

            pred = pred.detach().cpu()
            depth = depth.cpu()
            pc_image = pc_image.cpu()
            train_metrics.update(utils.compute_errors(depth[masktraj], pred[masktraj], 'traj/'))
            train_metrics.update(utils.compute_errors(pc_image[maskpc], pred[maskpc], 'pc/'))


            # writer.add_scalar("Loss/train/l_chamfer", l_chamfer/args.batch_size, global_step=epoch*len(train_loader)+i)
            # writer.add_scalar("Loss/train/l_sum", loss/args.batch_size, global_step=epoch*len(train_loader)+i)
            # writer.add_scalar("Loss/train/l_dense", l_dense/args.batch_size, global_step=epoch*len(train_loader)+i)
            # writer.add_scalar("Loss/train/l_edge", l_edge/args.batch_size, global_step=epoch*len(train_loader)+i)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()
            if should_log and step_count % 5 == 0:
                wandb.log({f"Loss/train/{criterion_ueff.name}": l_dense.item()/args.batch_size}, step=step_count)
                wandb.log({f"Loss/train/{criterion_bins.name}": l_chamfer.item()/args.batch_size}, step=step_count)
                wandb.log({f"Loss/train/{criterion_edge.name}": l_edge.item()/args.batch_size}, step=step_count)
                wandb.log({"Loss/train/l_sum": loss/args.batch_size}, step=step_count)

            step_count += 1
            scheduler.step()

            time_core += time.time()
            ########################################################################################################

            if should_write and step_count % args.trainconfig.validate_every == 0:

                ################################# Validation loop ##################################################
                model.eval()
                metrics, val_si = validate(args, model, test_loader, criterion_ueff, criterion_bins, criterion_edge, epoch, epochs, device)
                # [writer.add_scalar("test/"+k, v, step_count) for k,v in metrics.items()]
                # print("Validated: {}".format(metrics))
                if should_log:
                    wandb.log({
                        f"Test/{criterion_ueff.name}": val_si.get_value(),
                        # f"Test/{criterion_bins.name}": val_bins.get_value()
                    }, step=step_count)

                    wandb.log({f"test/{k}": v for k, v in metrics.items()}, step=step_count)
                    model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_latest.pt",
                                             root=saver.data_dir)
                                            
                    print(f"Total time spent: {time_total+time.time()}, core time spent:{time_core}")
                    time_total = -time.time()
                    time_core = 0.
                if metrics['traj/abs_rel'] < best_loss and should_write:
                    model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_best.pt",
                                             root=saver.data_dir)
                    best_loss = metrics['traj/abs_rel']
                model.train()
                #################################################################################################
        wandb.log({f"train/{k}": v for k, v in train_metrics.get_value().items()}, step=step_count)
        if (epoch+1)%2==0:
            log_images(test_loader, model, "vis/test", step_count, use_adabins=args.modelconfig.use_adabins)
            log_images(train_loader, model, "vis/train", step_count, use_adabins=args.modelconfig.use_adabins)
    return model


def validate(args, model, test_loader, criterion_ueff, criterion_bins, criterion_edge, epoch, epochs, device='cpu'):
    global count_val
    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation") if args.tqdm else test_loader:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            depth_var = batch['depth_variance'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            if(args.modelconfig.use_adabins):
                bin_edges, pred = model(img)
            else:
                bin_edges, pred = None, model(img)
            pc_image = batch["pc_image"].to(device)
            l_dense, l_chamfer, l_edge, masktraj, maskpc = train_loss(args, criterion_ueff, criterion_bins, criterion_edge, pred, bin_edges, depth, depth_var, pc_image, img)
            loss = l_dense + args.trainconfig.w_chamfer * l_chamfer + args.trainconfig.edge_aware_label_W * l_edge

            # writer.add_scalar("Loss/test/l_chamfer", l_chamfer, global_step=count_val)
            # writer.add_scalar("Loss/test/l_sum", loss, global_step=count_val)
            # writer.add_scalar("Loss/test/l_dense", l_dense, global_step=count_val)

            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            depth_var = depth_var.squeeze().unsqueeze(0).unsqueeze(0)
            mask = depth > args.min_depth
            count_val = count_val + 1
            val_si.append(l_dense.item())
            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='nearest')

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
            metrics.update(utils.compute_errors(gt_depth[valid_mask_traj], pred[valid_mask_traj], 'traj/'))
            metrics.update(utils.compute_errors(pc_image[valid_mask_pc], pred[valid_mask_pc], 'pc/'))
            metrics.update({ "l_chamfer": l_chamfer, "l_sum": loss, "/l_dense": l_dense})

        return metrics.get_value(), val_si


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)



if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--name", default="UnetAdaptiveBins")
    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")
    parser.add_argument("--tqdm", default=False, action="store_true", help="show tqdm progress bar")

    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='', type=str, help="Wandb tags, seperate by `,`")

    args = parse_args(parser, flatten = True)

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


    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.trainconfig.workers
    args.ngpus_per_node = ngpus_per_node
    
    saver_dir = os.path.join(args.root,"checkpoints")
    saver = ConfigurationSaver(log_dir=saver_dir,
                            save_items=[os.path.realpath(__file__)],
                            args=args,
                            dataclass_configs=[TrainConfig(**vars(args.trainconfig)), 
                                ModelConfig(**vars(args.modelconfig))])
                
    # writer = SummaryWriter(log_dir=saver.data_dir, flush_secs=60)

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
