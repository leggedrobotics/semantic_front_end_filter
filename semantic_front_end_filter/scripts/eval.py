# This file is part of SemanticFrontEndFilter.
#
# SemanticFrontEndFilter is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SemanticFrontEndFilter is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SemanticFrontEndFilter.  If not, see <https://www.gnu.org/licenses/>.


import os
import torch
from torch import nn
import numpy as np
from semantic_front_end_filter.utils.file_util import load_checkpoint, load_param_from_path
import semantic_front_end_filter.models as models
from semantic_front_end_filter.cfgs import parse_args
from simple_parsing import ArgumentParser
import yaml

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl  
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.transforms import Bbox
from ruamel.yaml import YAML
from tqdm import tqdm
from semantic_front_end_filter.utils.evalIoU_cuda import computeIoUs

# color configs
mpl.rc('font',family='Times New Roman')
paper_colors_rgb_u8 = {
        "orange": (251,151,39),
        "mangenta": (150,36,145),
        "blue": (67,110,176),
        "red": (210, 43,38),
        "cyan": (66, 173, 187),
        "green": (167,204,110),
        "red_light": (230,121,117),
        "orange_light": (252,188,115),
        "mangenta_light": (223,124,218),
        "blue_light": (137, 166,210),
        "cyan_light": (164, 216, 223), 
        "green_light": (192,218,152),
    }

for key, value in paper_colors_rgb_u8.items():
    paper_colors_rgb_u8[key] = np.array(value)/256

def cal_err_bins(depth_list, err_list_raw, err_list):
    bins = ["0-2m", "2-4m", "4-6m", "6-8m", "8-10m"]
    bins_list = np.zeros_like(depth_list).astype(str)
    depth_range = 2
    range_num = int(10/depth_range)

    # build bins
    min_depth = 0
    max_depth = min_depth + depth_range
    means_raw, stds_raw = np.array([]), np.array([])
    means_ours, stds_ours = np.array([]), np.array([])

    for i in range(range_num):
        selected = np.logical_and((depth_list>min_depth), (depth_list<max_depth))
        bins_list[selected] = bins[i]
        err_list_raw_selected = err_list_raw[selected]
        err_list_selected = err_list[selected]
        means_raw = np.append(means_raw, err_list_raw_selected.mean())
        stds_raw = np.append(stds_raw, err_list_raw_selected.std())
        means_ours = np.append(means_ours, err_list_selected.mean())
        stds_ours = np.append(stds_ours, err_list_selected.std())
        min_depth += depth_range
        max_depth += depth_range

    return (means_raw, stds_raw), (means_ours,stds_ours), bins_list



if __name__ == '__main__':
    parser = ArgumentParser()
    out_dir = os.path.dirname(os.path.abspath(__file__)) + "/../results/"
    parser.add_argument(
        "--model", default="")
    parser.add_argument(
        "--outdir", default=out_dir)
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
    args= parse_args(parser)

    model_cfg, train_cfg = load_param_from_path(os.path.dirname(args.model))    
    args.modelconfig.ablation = model_cfg['ablation']
    args.trainconfig.sprase_traj_mask = train_cfg['sprase_traj_mask']
    print(args.modelconfig.ablation, "skip_connection: ", args.trainconfig.sprase_traj_mask)

    model = models.UnetAdaptiveBins.build(**model_cfg)                                        
    model = load_checkpoint(args.model ,model)[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Compute IoUs
    depth_list_G, err_list_G, err_list_raw_G = computeIoUs(model, args, loader='offline_eval', env='grassland', print_result=False, depth_limit=10)
    depth_list_H, err_list_H, err_list_raw_H = computeIoUs(model, args, loader='offline_eval', env='hillside', print_result=False, depth_limit=10)
    depth_list_F, err_list_F, err_list_raw_F = computeIoUs(model, args, loader='offline_eval', env='forest', print_result=False, depth_limit=10)


    # bins = ["0-2m", "2-4m", "4-6m", "6-8m", "8-10m"]
    raw, ours, bins_list_G = cal_err_bins(depth_list_G, np.abs(err_list_raw_G), np.abs(err_list_G))
    df_err_G = pd.DataFrame({"err": np.concatenate([np.abs(err_list_raw_G), np.abs(err_list_G)], axis = 0), 
                        "bins": np.concatenate([bins_list_G, bins_list_G]),
                        "model":np.concatenate([np.full(err_list_raw_G.shape, "raw"), np.full(err_list_raw_G.shape, "ours")]),
                            "env": np.concatenate([np.full(err_list_raw_G.shape, "Grassland"), np.full(err_list_raw_G.shape, "Grassland")])})

    raw, ours, bins_list_H = cal_err_bins(depth_list_H, np.abs(err_list_raw_H), np.abs(err_list_H))
    df_err_H = pd.DataFrame({"err": np.concatenate([np.abs(err_list_raw_H), np.abs(err_list_H)], axis = 0), 
                        "bins": np.concatenate([bins_list_H, bins_list_H]),
                        "model":np.concatenate([np.full(err_list_raw_H.shape, "raw"), np.full(err_list_raw_H.shape, "ours")]),
                            "env": np.concatenate([np.full(err_list_raw_H.shape, "Hillside"), np.full(err_list_raw_H.shape, "Hillside")])})

    raw, ours, bins_list_F = cal_err_bins(depth_list_F, np.abs(err_list_raw_F), np.abs(err_list_F))
    df_err_F = pd.DataFrame({"err": np.concatenate([np.abs(err_list_raw_F), np.abs(err_list_F)], axis = 0), 
                        "bins": np.concatenate([bins_list_F, bins_list_F]),
                        "model":np.concatenate([np.full(err_list_raw_F.shape, "raw"), np.full(err_list_raw_F.shape, "ours")]),
                            "env": np.concatenate([np.full(err_list_raw_F.shape, "Forest"), np.full(err_list_raw_F.shape, "Forest")])})

    df_err = pd.concat([df_err_G, df_err_H, df_err_F])

    fontsize=40
    mpl.rc('font',family='Times New Roman')
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
        
    g = sns.catplot(data=df_err, x="bins", y = 'err', hue="model", col="env", kind="box", order = ["0-2m", "2-4m", "4-6m", "6-8m", "8-10m"], hue_order=["ours", "raw"], fliersize=2, legend_out=False, palette =[paper_colors_rgb_u8["orange"], paper_colors_rgb_u8["blue"]] )
    g.axes[0][0].set_title("Grassland", fontsize=fontsize)
    g.axes[0][0].set_xlabel("")
    g.axes[0][0].set_ylabel("Err(m)", fontsize=fontsize)

    g.axes[0][1].set_xlabel("Distance Range(m)", fontsize=fontsize)
    g.axes[0][1].set_title("Hillside", fontsize=fontsize)

    g.axes[0][2].set_xlabel("")
    g.axes[0][2].set_title("Forest", fontsize=fontsize)

    g.axes[0][0].legend(fontsize=30)
    for ax in g.axes[0]: 
        plt.setp(ax.get_xticklabels(), fontsize=30, text="")
        plt.setp(ax.get_yticklabels(), fontsize=30)
        ax.set_xticklabels(["0-2", "2-4", "4-6", "6-8", "8-10"])

    # plt.show()
    plt.savefig(args.outdir + "raw_vs_ours.pdf", bbox_inches='tight')