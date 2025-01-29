# This file is part of MyProject.
#
# MyProject is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MyProject is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MyProject.  If not, see <https://www.gnu.org/licenses/>.


import os
import json
from matplotlib.lines import Line2D
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any
from segments import SegmentsClient
from segments.utils import load_label_bitmap_from_url, load_image_from_url
from ruamel.yaml import YAML
import msgpack
import msgpack_numpy as m
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn as nn
import argparse

m.patch()
mpl.rc('font',family='Times New Roman')


def unpackMsgpack(path):
    with open(path, "rb") as file:
        data = msgpack.unpackb(file.read())
    return data

def visualize(data):
    cbar_size = 20
    title_size = 26
    fig, axs = plt.subplots(3, 2,figsize=(15, 10))
    
    # rgb image input    
    image_bgr = np.moveaxis(data["image"], 0, 2) / 255.0   # shape: (H, W, 3), in BGR order
    image_rgb = image_bgr[..., ::-1] # convert to RGB
    axs[0,0].imshow(image_rgb)
    axs[0,0].set_title("Input RGB Image", size = title_size)
    
    # pc_input
    pc_img = data["pc_image"].squeeze().copy()
    pc_img[pc_img==0] = np.nan
    x, y = np.where((pc_img!=0) & (pc_img<100))
    axs[1,0].imshow(pc_img,vmin = 0, vmax=40)
    sc = axs[1,0].scatter(y, x, c = pc_img[x, y],  vmin = 0, vmax = 20, cmap="plasma")
    axs[1,0].set_title("Input PC Depth", size = title_size)
    # colorbar
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(sc, cax = cax, ax = axs[1, 0])
    cbar.ax.tick_params(labelsize=cbar_size) 
        
    # depth label
    # depth mean
    depth = data["depth_var"][0].copy()
    depth[data["depth_var"][1]>1] = np.nan
    sc = axs[0,1].imshow(depth,vmin = 0, vmax=20, cmap="plasma")
    axs[0,1].set_title("SSDE Label - Mean", size = title_size)
    # colorbar
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(sc, cax = cax, ax = axs[0, 1])
    cbar.ax.tick_params(labelsize=cbar_size) 
    # dpeth var
    depth_var = data["depth_var"][1].copy()
    depth_var[depth_var>1] = np.nan
    sc = axs[1,1].imshow(depth_var,vmin = 0, vmax=1, cmap="plasma")
    axs[1,1].set_title("SSDE Label - Variance", size = title_size)
    # colorbar
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(sc, cax = cax, ax = axs[0, 1])
    cbar.ax.tick_params(labelsize=cbar_size) 
    
    # semantic label
    semantic_bitmap = data["anomaly_mask"]        
    bit_recolor = semantic_bitmap.copy()
    bit_recolor[semantic_bitmap == 1] = 2
    bit_recolor[semantic_bitmap == 2] = 1
    axs[2, 1].imshow(image_rgb)
    axs[2, 1].imshow(bit_recolor, alpha = 0.5)
    axs[2, 1].set_title("SSSeg Label", size = title_size)

    legend_elements = [Line2D([0], [0], marker='o', linewidth = 0, color='#a280aa', label='Obstacles',
                            markerfacecolor='#a280aa', markersize=title_size/2),
                    Line2D([0], [0], marker='o', linewidth = 0, color='#fef392', label='Support Surface',
                            markerfacecolor='#fef392', markersize=title_size/2)]
    axs[2, 1].legend(handles = legend_elements, fontsize = title_size/2)

    [axi.set_axis_off() for axi in axs.ravel()]

    plt.show()


if __name__ == "__main__":
    # This file visualize what the dataset contains
    # In the left column, the input RGB image and the input PC depth image are shown
    # In the right column, the SSDE label and the SSSeg label are shown
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/media/anqiao/AnqiaoT7/semantic_front_end_filter/extract_trajectories", help="Path to the dataset")
    parser.add_argument("--datum", type=str, default="Reconstruct_2022-07-21-10-47-29_0/traj_5_datum_10.msgpack", help="Path to the datum")
    args = parser.parse_args()
    
    traj_path = os.path.join(args.dataset, args.datum)
    data = unpackMsgpack(traj_path)
    visualize(data) 