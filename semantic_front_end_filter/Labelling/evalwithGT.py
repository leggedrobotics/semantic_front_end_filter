import os
import json
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any
from segments import SegmentsClient
from segments.utils import load_label_bitmap_from_url, load_image_from_url
from ruamel.yaml import YAML
import msgpack
import msgpack_numpy as m
from semantic_front_end_filter.adabins import models, model_io
import torch
import cv2 
from semantic_front_end_filter_ros.scripts.deploy_foward import RosVisulizer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn as nn

m.patch()

def get_semantic_bitmap(
    instance_bitmap: Optional[npt.NDArray[np.uint32]] = None,
    annotations: Optional[Dict[str, Any]] = None,
    id_increment: int = 0,
) -> Optional[npt.NDArray[np.uint32]]:
    """Convert an instance bitmap and annotations dict into a segmentation bitmap.
    Args:
        instance_bitmap: A :class:`numpy.ndarray` with :class:`numpy.uint32` ``dtype`` where each unique value represents an instance id. Defaults to :obj:`None`.
        annotations: An annotations dictionary. Defaults to :obj:`None`.
        id_increment: Increment the category ids with this number. Defaults to ``0``.
    Returns:
        An array here each unique value represents a category id.
    """

    if instance_bitmap is None or annotations is None:
        return None

    instance2semantic = [0] * (max([a.id for a in annotations], default=0) + 1)
    for annotation in annotations:
        instance2semantic[annotation.id] = annotation.category_id + id_increment
    instance2semantic = np.array(instance2semantic)

    semantic_label = instance2semantic[np.array(instance_bitmap, np.uint32)]
    return semantic_label

def unpackMsgpack(path):
    with open(path, "rb") as file:
        data = msgpack.unpackb(file.read())
    return data

class maskBasedRMSE:
    def __init__(self) -> None:
        self.traj_rmse_list = []
        self.pc_rmse_list = []        
        self.traj_err_list = np.array([])
        self.pc_err_list = np.array([])
        self.traj_mask_valid_list = []
        self.pc_mask_valid_list = []
        self.names = ["traj", "pc"]
        self.range_name = "None"

    def cal_rmse(reference, prediction):
        return np.sqrt(np.mean((reference - prediction)**2))
        
    def cal_and_append(self, reference, prediction, mask, name):
        if ~mask.any():
            print("mask is zero")
            return
        err = np.sqrt(np.mean((reference[mask] - prediction[mask])**2))
        # err = np.mean(reference[mask] - prediction[mask])
        assert name in self.names
        if name == "traj":
            self.traj_rmse_list.append(err)
            self.traj_mask_valid_list.append(np.count_nonzero(mask))
            self.traj_err_list = np.append(self.traj_err_list, prediction[mask] - reference[mask])
        if name == "pc":
            self.pc_rmse_list.append(err)
            self.pc_mask_valid_list.append(np.count_nonzero(mask))
            self.pc_err_list = np.append(self.pc_err_list, prediction[mask] - reference[mask])

        
    def get_mean_rmse(self, name):
        assert name in self.names
        if name == "traj":
            err_weighted_list = np.array(self.traj_rmse_list) * np.array(self.traj_mask_valid_list)/sum(self.traj_mask_valid_list)
        if name == "pc":
            err_weighted_list =  np.array(self.pc_rmse_list) * np.array(self.pc_mask_valid_list)/sum(self.pc_mask_valid_list)
        return sum(err_weighted_list)

    # def get_whole_rmse(self, name):
    #     assert name in self.names
    #     if name == "traj":
    #         err_weighted = np.sqrt(((np.array(self.traj_rmse_list) ** 2) * np.array(self.traj_mask_valid_list)/sum(self.traj_mask_valid_list)).sum())
    #         err_weighted_std = ((np.array(self.traj_rmse_list) - err_weighted)*np.array(self.traj_mask_valid_list)/sum(self.traj_mask_valid_list)).std()
    #     if name == "pc":
    #         err_weighted =  np.sqrt(((np.array(self.pc_rmse_list) ** 2) * np.array(self.pc_mask_valid_list)/sum(self.pc_mask_valid_list)).sum())
    #         err_weighted_std = ((np.array(self.pc_rmse_list) - err_weighted)*np.array(self.pc_mask_valid_list)/sum(self.pc_mask_valid_list)).std()
    #     return err_weighted
    def get_whole_rmse(self, name):
        assert name in self.names
        if name == "traj":
            err_weighted_rmse = np.sqrt(np.mean(self.traj_err_list ** 2))
            err_weighted_mean = np.mean(self.traj_err_list)
            err_weighted_std= np.std(self.traj_err_list)
        if name == "pc":
            err_weighted_rmse = np.sqrt(np.mean(self.pc_err_list ** 2))
            err_weighted_mean = np.mean(self.pc_err_list)
            err_weighted_std= np.std(self.pc_err_list)
        return err_weighted_rmse, err_weighted_mean, err_weighted_std

    def print_info(self):
        traj_mean = self.get_whole_rmse("traj")[0]
        pc_mean = self.get_whole_rmse("pc")[0]
        traj_max, traj_min = np.array(self.traj_rmse_list).max(), np.array(self.traj_rmse_list).min()
        pc_max, pc_min = np.array(self.pc_rmse_list).max(), np.array(self.pc_rmse_list).min()
        print("%d images are accounted for soft objects, %d images are accounted for rigid objects" %(len(self.traj_rmse_list), len(self.pc_rmse_list)))
        print("error on soft: mean = %.3f, range = (%.3f, %.3f)" %(traj_mean, traj_min, traj_max))
        print("error on rigid: mean = %.3f, range = (%.3f, %.3f)" %(pc_mean, pc_min, pc_max))

    def print_latest_info(self):
        print("the latest error on soft: mean = %.3f" %(self.traj_rmse_list[-1]))
        print("the latest error on rigid: mean = %.3f" %(self.pc_rmse_list[-1]))
    
    def print_important_info(self):
        print(self.get_whole_rmse("traj"), ', ', self.get_whole_rmse("pc"), end=', ')

if __name__ == "__main__":
    """When changing the different labeled set, please change the dataset_identifier, 
        traj_path and parameter of unpackMsgpack()"""

    key = "002502c9bfbc0a2f44271dbb5ff3ee82ca6c439a"
    client = SegmentsClient(key)
    # dataset_identifier = "yangcyself/Zurich-Reconstruct_2022-08-13-08-48-50_0"
    dataset_identifier = "Anqiao/Italy-Reconstruct_2022-07-18-20-34-01_0"

    # Get dataset from cloud and disk
    samples = client.get_samples(dataset_identifier)
    # traj_path = "/media/anqiao/Semantic/Data/extract_trajectories_006_Zurich_slim/extract_trajectories/" + dataset_identifier.split('-', 1)[-1]
    traj_path = "/media/anqiao/Semantic/Data/extract_trajectories_006_Italy_slim/extract_trajectories/" + dataset_identifier.split('-', 1)[-1]
    
    # Build model
    model_path = "/home/anqiao/tmp/semantic_front_end_filter/adabins/checkpoints/2022-11-04-02-05-45_edge5/UnetAdaptiveBins_best.pt"
    # model_path = "/media/anqiao/Semantic/Models/2022-08-29-23-51-44_fixed/UnetAdaptiveBins_latest.pt"
    model_cfg = YAML().load(open(os.path.join(os.path.dirname(model_path), "ModelConfig.yaml"), 'r'))
    model_cfg["input_channel"] = 4
    # model_cfg["deactivate_bn"] = False
    # model_cfg["skip_connection"] = False
    model = models.UnetAdaptiveBins.build(**model_cfg)                                        
    model = model_io.load_checkpoint(model_path ,model)[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rosv = RosVisulizer("pointcloud", camera_calibration_path = "/home/anqiao/tmp/semantic_front_end_filter/semantic_front_end_filter/Labelling/Example_Files/alphasense")
    
    # Calculate rmse
    rmse = maskBasedRMSE()
    rmse0_2 = maskBasedRMSE()
    rmse2_4 = maskBasedRMSE()
    rmse4_6 = maskBasedRMSE()
    rmse6_8 = maskBasedRMSE()
    rmse8_10 = maskBasedRMSE()
    for num, sample in enumerate(samples):
        if num ==1 or num ==2:
            continue        
        if num <100:
            continue
        # if num < 127:
            # continue
        # Get labels
        try:
            label = client.get_label(sample.uuid, labelset='ground-truth')
        except:
            print("picture "+sample.name+" is not labeled")
            break

        instance_bitmap = load_label_bitmap_from_url(  label.attributes.segmentation_bitmap.url )
        semantic_bitmap = get_semantic_bitmap(instance_bitmap, label.attributes.annotations) # 2->soft 1->rigid
        if (2 not in semantic_bitmap) & (1 not in semantic_bitmap): 
            continue
        if 2 not in semantic_bitmap:
            semantic_bitmap[semantic_bitmap!=1]=2
        if 1 not in semantic_bitmap:
            semantic_bitmap[semantic_bitmap!=2]=1 

        # Get corresponding predction
        # data = unpackMsgpack(traj_path + '/' + sample.name.split('.')[0]+'.msgpack')
        data = unpackMsgpack(traj_path + '/' + (sample.name.split('.')[0]).split('_', 1)[-1]+'.msgpack')
        pc_img = torch.Tensor(data['pc_image'].squeeze()[None, ...]).to(device)
        image = torch.Tensor(data['image']).to(device)
        input = torch.cat([image/255., pc_img],axis=0)
        input = input[None, ...]
        mean, std = [0.387, 0.394, 0.404, 0.120], [0.322, 0.32, 0.30,  1.17]
        for i, (m, s) in enumerate(zip(mean, std)):
            input[0, i, ...] = (input[0, i, ...] - m)/s
        # input[0, 0:3] = 0
        pred = model(input)[0][0]
        # pred = nn.functional.interpolate(pred[None, None, ...], torch.tensor(pc_img[0]).shape, mode='nearest')
        pred_show = pred.detach().cpu().numpy().copy()
        # pred = pred[0, 0]
        # pred = pc_img.squeeze()
        pred [(pc_img[0]==0)] = torch.nan

        # pred[(pc_img[0]-pred)>0] = torch.nan
        # fusing raw lidar points
        # pred_fusion = rosv.raycastCamera.fuse(pred.T.clone(), pc_img.squeeze().T, torch.Tensor(data['pose']))
        # pred_fusion = pred_fusion.T    
        pred = pred.detach().cpu().numpy()
        # Get two masks 
        pc_mask = ~np.isnan(pred) & (semantic_bitmap==2) & (data['pc_image'].squeeze()>1e-9) & (data["depth_var"][0].squeeze()<10)  & (data['pc_image'].squeeze()<10)
        traj_mask = ~np.isnan(pred) & (semantic_bitmap==1) & (data["depth_var"][0]>0) & (data["depth_var"][1]<0.025) & (data['pc_image'].squeeze()>1e-9)  &  (data['pc_image'].squeeze()<10) & (data["depth_var"][0].squeeze()<10) & (data['pc_image'].squeeze()>1e-9)     

        # Calculate rmse
        if traj_mask.any():
            rmse.cal_and_append(data['depth_var'][0], pred, traj_mask, name="traj")
            rmse0_2.cal_and_append(data['depth_var'][0], pred, traj_mask & (data["depth_var"][0]<2), name="traj")
            rmse2_4.cal_and_append(data['depth_var'][0], pred, traj_mask & (data["depth_var"][0]>=2) & (data["depth_var"][0]<=4), name="traj")
            rmse4_6.cal_and_append(data['depth_var'][0], pred, traj_mask & (data["depth_var"][0]>=4) & (data["depth_var"][0]<=6), name="traj")
            rmse6_8.cal_and_append(data['depth_var'][0], pred, traj_mask & (data["depth_var"][0]>=6) & (data["depth_var"][0]<=8), name="traj")
            rmse8_10.cal_and_append(data['depth_var'][0], pred, traj_mask & (data["depth_var"][0]>=8) & (data["depth_var"][0]<=10), name="traj")
        if pc_mask.any():
            rmse.cal_and_append(data['pc_image'].squeeze(), pred, pc_mask, name="pc")
            rmse0_2.cal_and_append(data['pc_image'].squeeze(), pred, pc_mask & (data['pc_image'].squeeze()<2), name="pc")
            rmse2_4.cal_and_append(data['pc_image'].squeeze(), pred, pc_mask & (data['pc_image'].squeeze()>=2) & (data['pc_image'].squeeze()<=4), name="pc")
            rmse4_6.cal_and_append(data['pc_image'].squeeze(), pred, pc_mask & (data['pc_image'].squeeze()>=4) & (data['pc_image'].squeeze()<=6), name="pc")           
            rmse6_8.cal_and_append(data['pc_image'].squeeze(), pred, pc_mask & (data['pc_image'].squeeze()>=6) & (data['pc_image'].squeeze()<=8), name="pc")           
            rmse8_10.cal_and_append(data['pc_image'].squeeze(), pred, pc_mask & (data['pc_image'].squeeze()>=8) & (data['pc_image'].squeeze()<=10), name="pc")           
        print(num)
        
        try:
            print(rmse0_2.pc_rmse_list[-1], rmse0_2.pc_mask_valid_list[-1])
            # print(rmse0_2.traj_rmse_list[-1], rmse0_2.traj_mask_valid_list[-1])
        except:
            print("no data")
        # Plot the result
        PLOT = True
        if(PLOT ):
            fig, axs = plt.subplots(2, 4,figsize=(16, 8))
            # fig.suptitle(traj_path + '/' + sample.name.split('.')[0]+'.msgpack')
            if(axs.ndim==1):
                axs = axs[None,...]
            # axs[0,0].imshow(inputimg[:,:,:3])
            image = cv2.cvtColor(np.moveaxis(data["image"], 0, 2)/255, cv2.COLOR_BGR2RGB)
            axs[0,0].imshow(image)
            # axs[0,0].set_title("Input RGB image")

            depth = data["depth_var"][0].copy()
            depth[data["depth_var"][1]>0.05] = 0
            axs[0,1].imshow(depth,vmin = 0, vmax=40)
            # axs[0,1].set_title("Support surface label")
            
            # pc_img[pc_img==0] = np.nan
            pc_img = data["pc_image"].squeeze().copy()
            # pc_img[pc_mask] = 0
            x, y = np.where(pc_img!=0)
            axs[0,2].scatter(y, x, c = pc_img[x, y],  vmin = 0, vmax = 20)
            # pc_img = data["pc_image"].squeeze()
            print(pc_img.shape)
            axs[0,2].imshow(pc_img,vmin = 0, vmax=40)
            # axs[0,2].set_title("Point cloud label & 4th channel of input")

            # pc_diff = pc_img - depth
            # pc_diff[depth<1e-9] = 0
            # pc_diff[pc_img<1e-9] = 0
            # axs[0,3].imshow(pc_diff,vmin = -5, vmax=5)
            # axs[0,3].set_title("pc - traj")

            # second line
            # axs[1,0].imshow(image)
            axs[0,0].imshow(semantic_bitmap, alpha = 0.2)
            # axs[1,0].set_title("segmantation")
            
            pred_traj_diff = pred - depth
            pred_traj_diff[~traj_mask] = np.nan
            x, y = np.where(~np.isnan(pred_traj_diff))
            sc = axs[0, 3].scatter(y, x, c = pred_traj_diff[x, y], vmin = -5, vmax = 5)        # axs[1,1].imshow(pred_traj_diff)
            # axs[1,0].set_title("prediction difference")
            divider = make_axes_locatable(axs[0, 3])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(sc, cax = cax, ax = axs[0, 3])
            axs[0, 3].imshow(semantic_bitmap, alpha = 0.5)

            pred_pc_diff = pred - pc_img
            pred_pc_diff[~pc_mask] = np.nan
            pred_pc_diff[pred_pc_diff>10] = 0
            x, y = np.where(~np.isnan(pred_pc_diff))
            axs[0,3].scatter(y, x, c = pred_pc_diff[x, y], vmin = -5, vmax = 5)

            axs[1,0].imshow(pred_show)
            # axs[1,2].imshow(semantic_bitmap, alpha = 0.5)
            # axs[1,2].imshow(pred_traj_pc)
            # axs[1,2].set_title("prediction difference with pc label")
            # rmse.print_latest_info()
            [axi.set_axis_off() for axi in axs.ravel()]

            # axs[1,1].set_axis_off()
            # axs[1,2].set_axis_off()
            fig.tight_layout()
            plt.show()
            plt.close()
            # break
        # axs[0,3].set_title("Input pc")


    # output  
    print(str(num) + " images are loaded")
    # rmse.print_info()
    # print("0-2")
    # rmse0_2.print_info()
    # print("2-4")
    # rmse2_4.print_info() 
    # print(rmse2_4.pc_rmse_list, rmse2_4.pc_mask_valid_list)   
    # print("4-6")
    # rmse4_6.print_info()
    # print("6-8")
    # rmse6_8.print_info()
    # print("8-10")
    # rmse8_10.print_info()
    rmselist = [rmse0_2, rmse2_4, rmse4_6, rmse6_8, rmse8_10]
    for rmse_i in rmselist:
        rmse_i.print_important_info()