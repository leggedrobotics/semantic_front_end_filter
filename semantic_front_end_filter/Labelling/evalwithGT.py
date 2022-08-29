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
    traj_rmse_list = []
    pc_rmse_list = []
    traj_mask_valid_list = []
    pc_mask_valid_list = []
    names = ["traj", "pc"]
    def cal_and_append(self, reference, prediction, mask, name):
        err = np.sqrt(np.mean((reference[mask] - prediction[mask])**2))
        assert name in self.names
        if name == "traj":
            self.traj_rmse_list.append(err)
            self.traj_mask_valid_list.append(np.count_nonzero(mask))
        if name == "pc":
            self.pc_rmse_list.append(err)
            self.pc_mask_valid_list.append(np.count_nonzero(mask))
        
    def get_mean_rmse(self, name):
        assert name in self.names
        if name == "traj":
            err_weighted_list = np.array(self.traj_rmse_list) * np.array(self.traj_mask_valid_list)/sum(self.traj_mask_valid_list)
        if name == "pc":
            err_weighted_list =  np.array(self.pc_rmse_list) * np.array(self.pc_mask_valid_list)/sum(self.pc_mask_valid_list)
        return sum(err_weighted_list)
    
    def print_info(self):
        traj_mean = self.get_mean_rmse("traj")
        pc_mean = self.get_mean_rmse("pc")
        traj_max, traj_min = np.array(self.traj_rmse_list).max(), np.array(self.traj_rmse_list).min()
        pc_max, pc_min = np.array(self.pc_rmse_list).max(), np.array(self.pc_rmse_list).min()
        print("%d images are accounted for soft objects, %d images are accounted for rigid objects" %(len(self.traj_rmse_list), len(self.pc_rmse_list)))
        print("error on soft: mean = %.3f, range = (%.3f, %.3f)" %(traj_mean, traj_min, traj_max))
        print("error on rigid: mean = %.3f, range = (%.3f, %.3f)" %(pc_mean, pc_min, pc_max))

    def print_latest_info(self):
        print("the latest error on soft: mean = %.3f" %(self.traj_rmse_list[-1]))
        print("the latest error on rigid: mean = %.3f" %(self.pc_rmse_list[-1]))

if __name__ == "__main__":
    # key = os.getenv("SEGMENTS_AI_API_KEY")
    key = "002502c9bfbc0a2f44271dbb5ff3ee82ca6c439a"
    client = SegmentsClient(key)
    dataset_identifier = "yangcyself/Zurich-Reconstruct_2022-08-13-08-48-50_0"
    # get dataset from cloud and disk
    samples = client.get_samples(dataset_identifier)
    traj_path = "/media/anqiao/Semantic/Data/extract_trajectories_006_Zurich_slim/extract_trajectories/" + dataset_identifier.split('-', 1)[-1]
    # build model
    model_path = "/home/anqiao/tmp/semantic_front_end_filter/adabins/checkpoints/2022-08-28-11-30-58/UnetAdaptiveBins_latest.pt"
    model_cfg = YAML().load(open(os.path.join(os.path.dirname(model_path), "ModelConfig.yaml"), 'r'))
    model = models.UnetAdaptiveBins.build(n_bins=256, input_channel = 4, min_val=0.001, max_val=10,
                                            norm="linear", use_adabins=model_cfg["use_adabins"], deactivate_bn = model_cfg["deactivate_bn"], skip_connection = model_cfg["skip_connection"])
    model = model_io.load_checkpoint(model_path ,model)[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    rmse = maskBasedRMSE()
    for num, sample in enumerate(samples):
        # get labels
        try:
            label = client.get_label(sample.uuid, labelset='ground-truth')
        except:
            print("picture "+sample.name+" is not labeled")
            continue
        # if sample.label:
        #     label = client.get_label(sample.uuid, labelset='ground-truth')  
        # else:
        #     continue

        instance_bitmap = load_label_bitmap_from_url(  label.attributes.segmentation_bitmap.url )
        semantic_bitmap = get_semantic_bitmap(instance_bitmap, label.attributes.annotations) # 2->soft 1->rigid
        # preprocess labels
        if (2 not in semantic_bitmap) & (1 not in semantic_bitmap): 
            continue
        if 2 not in semantic_bitmap:
            semantic_bitmap[semantic_bitmap!=1]=2
        if 1 not in semantic_bitmap:
            semantic_bitmap[semantic_bitmap!=2]=1 

        # Get corresponding predction
        data = unpackMsgpack(traj_path + '/' + sample.name.split('.')[0]+'.msgpack')
        pc_img = torch.Tensor(data['pc_image'].squeeze()[None, ...]).to(device)
        image = torch.Tensor(data['image']).to(device)
        input = torch.cat([image/255., pc_img],axis=0)
        input = input[None, ...]
        mean, std = [0.387, 0.394, 0.404, 0.120], [0.322, 0.32, 0.30,  1.17]
        for i, (m, s) in enumerate(zip(mean, std)):
            input[0, i, ...] = (input[0, i, ...] - m)/s
        pred = model(input)[0][0]
        # pred = pc_img.squeeze()
        pred [(pc_img[0]==0)] = torch.nan
        pred = pred.detach().cpu().numpy()

        # Get two masks 
        pc_mask = (semantic_bitmap==2) & (data['pc_image'].squeeze()>1e-9) & (data['pc_image'].squeeze()<10) & (data["depth_var"][0].squeeze()<10)
        traj_mask = (semantic_bitmap==1) & (data["depth_var"][0]>0) & (data["depth_var"][1]<0.025) & (data['pc_image'].squeeze()>1e-9) & (data['pc_image'].squeeze()<10) & (data["depth_var"][0].squeeze()<10)      

        # Calculate rmse
        if traj_mask.any():
            rmse.cal_and_append(data['depth_var'][0], pred, traj_mask, name="traj")
        if pc_mask.any():
           rmse.cal_and_append(data['pc_image'].squeeze(), pred, pc_mask, name="pc")
        print(num)

        PLOT = False
        # plot the result
        if(PLOT):
            fig, axs = plt.subplots(2, 4,figsize=(20, 20))
            fig.suptitle(traj_path + '/' + sample.name.split('.')[0]+'.msgpack')
            if(axs.ndim==1):
                axs = axs[None,...]
            # axs[0,0].imshow(inputimg[:,:,:3])
            image = cv2.cvtColor(np.moveaxis(data["image"], 0, 2)/255, cv2.COLOR_BGR2RGB)
            axs[0,0].imshow(image)
            axs[0,0].set_title("Input")

            depth = data["depth_var"][0]
            axs[0,1].imshow(depth,vmin = 0, vmax=40)
            axs[0,1].set_title("traj label")

            pc_img = data["pc_image"].squeeze()
            print(pc_img.shape)
            axs[0,2].imshow(pc_img,vmin = 0, vmax=40)
            axs[0,2].set_title("pc label")

            pc_diff = pc_img - depth
            pc_diff[depth<1e-9] = 0
            pc_diff[pc_img<1e-9] = 0
            axs[0,3].imshow(pc_diff,vmin = -5, vmax=5)
            axs[0,3].set_title("pc - traj")

            # second line
            axs[1,0].imshow(image)
            axs[1,0].imshow(semantic_bitmap, alpha = 0.5)
            axs[1,0].set_title("segmantation")
            
            pred_traj_diff = pred - depth
            pred_traj_diff[~traj_mask] = np.nan
            x, y = np.where(~np.isnan(pred_traj_diff))
            axs[1,1].scatter(y, x, c = pred_traj_diff[x, y], vmin = -5, vmax = 5)        # axs[1,1].imshow(pred_traj_diff)
            axs[1,1].imshow(semantic_bitmap, alpha = 0.5)
            axs[1,1].set_title("prediction difference with traj label")

            pred_pc_diff = pred - pc_img
            pred_pc_diff[~pc_mask] = np.nan
            x, y = np.where(~np.isnan(pred_pc_diff))
            axs[1,2].scatter(y, x, c = pred_pc_diff[x, y], vmin = -5, vmax = 5)
            axs[1,2].imshow(semantic_bitmap, alpha = 0.5)
            # axs[1,2].imshow(pred_traj_pc)
            axs[1,2].set_title("prediction difference with pc label")
            rmse.print_latest_info()

            plt.show()
            plt.close()
            # break
        # axs[0,3].set_title("Input pc")


    # statistics 
    print(str(num) + " images are loaded")
    rmse.print_info()
    # print(str(len(rmse.traj_rmse_list)) + '/' + str(num) + " images are accounted")
    # print("traj_rmse: ", rmse.get_mean_rmse("traj"))
    # print("pc_rmse: ", rmse.get_mean_rmse("pc"))

    # img = load_image_from_url