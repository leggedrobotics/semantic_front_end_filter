#!/usr/bin/env python
"""
Chenyu 2022-08-19
Read pointclouds from Reconstructed Rosbag and generate elevation maps, compare it to ground truth and analysis
The structure is like a python-powered rosbag simulator. 
    Callback functions are used to process all messages from rosbag. 
"""

# Core
import sys, os, time
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt  
import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch
from torch import nn

# IO
from ruamel.yaml import YAML
import yaml
from argparse import ArgumentParser
import pickle as pkl
import msgpack
import msgpack_numpy as m
m.patch()

# Ros
import rospy # for `Duration`
import rosbag
import tf2_py
from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_from_matrix, quaternion_from_matrix
from sensor_msgs.msg import Image

# Labelling
from semantic_front_end_filter.Labelling.messages.imageMessage import Camera, getImageId, rgb_msg_to_image
from semantic_front_end_filter.Labelling.messages.pointcloudMessage import rospcmsg_to_pcarray, ros_pc_msg2
from semantic_front_end_filter.Labelling.messages.messageToVectors import msg_to_body_ang_vel, msg_to_body_lin_vel, msg_to_rotmat, msg_to_command, \
    msg_to_pose, msg_to_joint_positions, msg_to_joint_velocities, msg_to_joint_torques, msg_to_grav_vec
from semantic_front_end_filter.Labelling.GroundfromTrajs import GFT

# Adabins
from semantic_front_end_filter.adabins.pointcloudUtils import RaycastCamera
from semantic_front_end_filter.adabins.models import UnetAdaptiveBins
from semantic_front_end_filter.adabins.model_io import load_checkpoint, load_param_from_path, load_train_param_from_path
from semantic_front_end_filter.adabins.elevation_vis import WorldViewElevationMap
from semantic_front_end_filter.adabins.elevation_eval_util import ElevationMapEvaluator


# Common rosbagPlayer
from semantic_front_end_filter.common import RosbagPlayer

def main (modelname, overwrite = False):
    # #### SA Configurations
    # rosbagpath = "/Data/20211007_SA_Monkey_ANYmal_Chimera/chimera_mission_2021_10_09/mission8_locomotion/Reconstruct_2022-04-25-19-10-16_0.bag"
    # foottrajpath = "/Data/extract_trajectories_002/Reconstruct_2022-04-25-19-10-16_0/FeetTrajs.msgpack"
    # groundmappath = "/Data/extract_trajectories_002/Reconstruct_2022-04-25-19-10-16_0/GroundMap.msgpack"
    # model_path = "checkpoints/2022-08-03-16-26-08/UnetAdaptiveBins_latest.pt"
    # image_topic = "/alphasense_driver_ros/cam4/dropped/debayered/compressed"
    # pc_topic = "/bpearl_rear/point_cloud"
    # TF_BASE = "base"
    # TF_MAP = "map"
    
    #### Italy Configurations
    
    ## Testing set
    # rosbagpath = "/Data/Italy_0820/18-20-34-01/Reconstruct_2022-07-18-20-34-01_0.bag" # testing set
    # foottrajpath = "/Data/Italy_0820/18-20-34-01/FeetTrajs.msgpack"
    # groundmappath = "/Data/Italy_0820/18-20-34-01/GroundMap.msgpack"
    # OUTDIR_NAME = "Italy_2022-07-18-20-34-01_0"
    # START_TIME = 10
    # END_TIME = 200

    ## Train set
    rosbagpath = "/Data/Italy_0820/19-20-06-22/Reconstruct_2022-07-19-20-06-22_0.bag"
    foottrajpath = "/Data/Italy_0820/19-20-06-22/FeetTrajs.msgpack"
    groundmappath = "/Data/Italy_0820/19-20-06-22/GroundMap.msgpack"
    OUTDIR_NAME = "Italy_2022-07-19-20-06-22_0"
    START_TIME = 100
    END_TIME = 300


    model_path = f"checkpoints/{modelname}/UnetAdaptiveBins_latest.pt"
    # model_path = f"checkpoints/{modelname}/UnetAdaptiveBins_best.pt"
    image_topic = "/alphasense_driver_ros/cam4/debayered"
    pc_topic = "/bpearl_rear/point_cloud"
    TF_BASE = "base"
    TF_MAP = "map"

    # #### Zurich Configurations
    # rosbagpath = "/Data/hongeberg/mission_data/Reconstruct/Reconstruct_2022-08-13-08-48-50_0.bag"
    # foottrajpath = "/Data/Italy_0820/FeetTrajs.msgpack"
    # groundmappath = "/Data/Italy_0820/GroundMap.msgpack"
    # model_path = f"checkpoints/{modelname}/UnetAdaptiveBins_best.pt"
    # image_topic = "/alphasense_driver_ros/cam4/debayered"
    # pc_topic = "/bpearl_rear/point_cloud"
    # TF_BASE = "base"
    # TF_MAP = "map"

    GENERATE_VIDEO = True
    if GENERATE_VIDEO: # this should be corresponded to the `play`
        # outputdir = f"checkpoints/{modelname}/Identity"
        outputdir = f"checkpoints/{modelname}/{OUTDIR_NAME}Vid_T{START_TIME}--{END_TIME}"
    else:
        outputdir = f"checkpoints/{modelname}/{OUTDIR_NAME}_T{START_TIME}--{END_TIME}"

    if not overwrite and os.path.exists(outputdir):
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raycastCamera = RaycastCamera(device=device) # WARN: This raycastcamera is hard coded with `tf_base_to_sensor`, however, it seems to be constant
    gft = GFT(FeetTrajsFile = foottrajpath, InitializeGP=False)
    foot_holds = {k : np.array(gft.getContactPoints(v)[0]) for k,v in gft.FeetTrajs.items()} # A dict of the contact points of each foot
    foot_holds_array = np.vstack(list(foot_holds.values()))

    elevation_pred = WorldViewElevationMap(resolution = None, map_length = None, init_with_initialize_map = None)
    elevation_pc = WorldViewElevationMap(resolution = None, map_length = None, init_with_initialize_map = None)
    elevation_fh = WorldViewElevationMap(resolution = None, map_length = None, init_with_initialize_map = None)
    # Evaluators used to evaluate the error
    evaluator_pred = ElevationMapEvaluator(groundmappath, elevation_pred.param)
    evaluator_pc = ElevationMapEvaluator(groundmappath, elevation_pc.param)
    evaluator_smooth = ElevationMapEvaluator(groundmappath, elevation_pc.param)
    evaluator_fh = ElevationMapEvaluator(groundmappath, elevation_fh.param)
    player = RosbagPlayer(rosbagpath)

    ## Initialize model
    model_cfg = load_param_from_path(model_path)
    # model_cfg["input_channel"] = 4
    model = UnetAdaptiveBins.build(**model_cfg)

    train_cfg = load_train_param_from_path(model_path)

    model = load_checkpoint(model_path, model)[0]
    model.to(device)

    ## Define shared variables
    player._shared_var.update(dict(
        pcbuffer=[],
    ))
    elev_pred_buffer = []
    if(GENERATE_VIDEO):
        player._shared_var.update(dict(
            video_frame_count = 0,
        ))
    video_frame_freq = 10
    image_list=[]
    pred_list=[]
    pred_elev_list=[]
    pc_elev_list=[]
    gt_list=[]
    errer_list_count=[] # the length of current evaluator.error_list
    ###########
    ## Definining callbacks

    def pred_and_checkerr(image, pc, pose,v):
        """
        Make the prediction based on image, pointcloud and robot current pose
        arg pose: x,y,z,rx,ry,rz,rw
        """
        image = torch.Tensor(image).to(device)
        points = torch.Tensor(pc).to(device)
        # pose = torch.Tensor(pose).to(device)
        # get pc image
        pc_img = torch.zeros_like(image[:1, ...]).to(device).float()
        pc_img,visible,proj_point = raycastCamera.project_cloud_to_depth(
                        pose, points, pc_img, return_detail=True)
        # filter the points to the only visble ones
        points = points[visible] # TODO: Decide whether should be filter the raw points
        # get prediction
        model_in = torch.cat([image/255., pc_img],axis=0)
        model_in = model_in[None, ...]
        if(model_cfg.get("ablation", None) == "onlyPC"):
            model_in[:, 0:3] = 0
        elif(model_cfg.get("ablation",None) == "onlyRGB"):
            model_in[:, 3] = 0 
        for i, (m, s) in enumerate(zip([0.387, 0.394, 0.404, 0.120], [0.322, 0.32, 0.30,  1.17])):
            model_in[0, i, ...] = (model_in[0, i, ...] - m)/s

        pred = model(model_in)
        if(train_cfg.get("sprase_traj_mask", False)):
            pred[:, 2:] = pred[:, 2:] + pc_img[None,0,...].to(device) # with or without pc_image
        else:
            pred[:, 2:] = pred[:, 2:]

        # print("pred_shape",pred.shape)
        # print("pc_img_shape",pc_img.shape)
        mask_weight = (pred[0, 1:2] > pred[0, 0:1])
        pred_origin = pred[0, 2:]
        pred = pred[0, 2:].clone()
        pred[~mask_weight] = pc_img[~mask_weight]
        pred = pred[0]
        # pred = model(model_in)[0][0]
        pred = nn.functional.interpolate(pred[None,None,...], torch.tensor(pc_img).shape[-2:], mode='nearest')[0][0]
        pred [(pc_img[0]==0)] = np.nan
        pred = pred.T

        # get elevation from prediction
        pred_pts = raycastCamera.project_depth_to_cloud(pose, pred)
        pred_pts = pred_pts[~torch.isnan(pred_pts[:,0])]
        # pred_pts = pred_pts[pred_pts[:,2]<pose[2]] # TODO: Decide whether this hieight mask is necessary
        pred_points = pred_pts.detach().cpu().numpy()
        points = points.cpu().numpy().astype(pred_points.dtype) # float_64
        # points = points[points[:,2]<pose[2]] # TODO: Decide whether this hieight mask is necessary
        fh_points = foot_holds_array[np.sum((foot_holds_array[:,:2] - pose[:2])**2, axis = 1)<1**2]

        elevation_pred.move_to_and_input(pose[0:3], pred_points)
        elevmap_pred = elevation_pred.get_elevation_map()
        elevation_pc.move_to_and_input(pose[0:3], points)
        elevmap_pc = elevation_pc.get_elevation_map()
        minfmap_pc = elevation_pc.get_layer_map("min_filter")
        smotmap_pc = elevation_pc.get_layer_map("smooth")
        elevation_fh.move_to_and_input(pose[0:3], fh_points)
        elevmap_fh = elevation_fh.get_elevation_map()
        # TMP YCY FILL THE Nan of the elevmap_fh
        fh_nan_mask = np.isnan(elevmap_fh)
        elevmap_fh[fh_nan_mask] = np.interp(np.flatnonzero(fh_nan_mask), np.flatnonzero(~fh_nan_mask), elevmap_fh[~fh_nan_mask])
        
        rz = Rotation.from_quat(pose[3:]).as_euler('xyz',degrees=False)[2]

        error_pred = evaluator_pred.compute_error_against_gpmap(elevmap_pred, pose[:2], rz)
        error_pc = evaluator_pc.compute_error_against_gpmap(elevmap_pc, pose[:2], rz)
        error_smooth = evaluator_smooth.compute_error_against_gpmap(smotmap_pc,  pose[:2], rz)
        error_fh = evaluator_fh.compute_error_against_gpmap(elevmap_fh, pose[:2], rz)
        
        elev_pred_buffer.append((elevmap_pred, pose))
        if(GENERATE_VIDEO):
            if(not v["video_frame_count"]):
                v["video_frame_count"] = video_frame_freq
                image_list.append(model_in.squeeze(0).cpu().numpy())
                pred_list.append(pred.detach().cpu().numpy().T)
                pred_elev_list.append(elevmap_pred)
                pc_elev_list.append(elevmap_pc)
                gt=evaluator_pred.get_gpmap_at_xy(pose[:2])
                gt_list.append(gt)
                errer_list_count.append(len(evaluator_pred.error_list))
            v["video_frame_count"] -=1
        
    def image_cb(topic, msg, t, tf_buffer, v):
        if(not len(v["pcbuffer"])): return
        
        img = rgb_msg_to_image(msg, raycastCamera.camera.is_debayered, raycastCamera.camera.rb_swap, ("compressed" in topic))
        img = np.moveaxis(img, 2, 0)

        if not (tf_buffer.can_transform_core(TF_MAP, TF_BASE,  msg.header.stamp)[0]): return 
        tf = tf_buffer.lookup_transform_core(TF_MAP, TF_BASE, msg.header.stamp)
        pose = msg_to_pose(tf)  # pose in fixed ref frame (odom or map)

        pc = np.concatenate(v["pcbuffer"],axis = 0)
        pred_and_checkerr(img, pc, pose, v)

        v["pcbuffer"] = v["pcbuffer"][-1:]

        ## Breakpoint vis
        # plt_img = np.moveaxis(img[:3,...], 0, 2)
        # plt_img = (plt_img-plt_img.min())/(plt_img.max()-plt_img.min())
        # plt.imshow(plt_img[:,:,::-1])
        # plt.show()
    player.register_callback(image_topic, image_cb)

    
    def pointcloud_cb(topic, msg, t, tf_buffer, v):
        if not (tf_buffer.can_transform_core(TF_MAP, msg.header.frame_id,  msg.header.stamp)[0]): return
        tf = tf_buffer.lookup_transform_core(TF_MAP, msg.header.frame_id,  msg.header.stamp)
        pose = msg_to_pose(tf)
        pc_array = rospcmsg_to_pcarray(msg, pose)

        v["pcbuffer"].append(pc_array[:,:3])
    player.register_callback(pc_topic, pointcloud_cb)

    start_time_ = player.bag.get_start_time()+START_TIME if(START_TIME>0) else player.bag.get_end_time()-START_TIME
    end_time_ = player.bag.get_start_time()+END_TIME if(END_TIME>0) else player.bag.get_end_time()-END_TIME
    player.play(start_time=start_time_, end_time=end_time_)

        # # player.play(end_time=player.bag.get_start_time()+5)
        # player.play(end_time=player.bag.get_start_time()+200)# play from start to before entering the forest
        # player.play(start_time=player.bag.get_start_time()+420)# play from in the foresst


    ## Output
    try:
        os.makedirs(outputdir)
    except Exception as e:
        print(e)
        pass

    # with open(os.path.join(outputdir, "evaluators.pkl"),"wb") as f:
    #     pkl.dump((evaluator_pred, evaluator_pc, evaluator_fh), f)
    # with open(os.path.join(outputdir, "elevations.pkl"),"wb") as f:
    #     pkl.dump(elev_pred_buffer, f)

    ### Generate Animation
    if(GENERATE_VIDEO):
        import matplotlib.animation as animation
        fig, axs = plt.subplots(2,3, figsize=(30,20))
        image_to_show_list = [((np.moveaxis(im[:3,...], 0, 2)-im[:3,...].min())
                    /(im[:3,...].max()-im[:3,...].min()))[:,:,::-1] for im in image_list]
        imgdatasource = [image_to_show_list, pred_list, pred_elev_list, pc_elev_list, gt_list]
        elevmin = np.min([m[~np.isnan(m)].min() for m in pred_elev_list+pc_elev_list])
        elevmax = np.max([m[~np.isnan(m)].max() for m in pred_elev_list+pc_elev_list])
        print("elev min and max", elevmin, elevmax)
        ims = [
            axs[0][0].imshow(image_to_show_list[0]),
            axs[0][1].imshow(pred_list[0], vmin = 0.1, vmax = 20),
            axs[1][0].imshow(pred_elev_list[0], vmin = elevmin, vmax = elevmax),
            axs[1][1].imshow(pc_elev_list[0], vmin = elevmin, vmax = elevmax),
            axs[1][2].imshow(gt_list[0], vmin = elevmin, vmax = elevmax)]
        lines = [axs[0][2].plot( [abs(e)[~np.isnan(e)].mean() for e in e_list[:10]], label = n)[0] 
            for e_list, n in zip([evaluator_pred.error_list, evaluator_pc.error_list], ["pred", "pc"] )]
        axs[0][2].legend()
        axs[0][2].set_xlim((0,100))
        axs[0][2].set_ylim((0,1.5))
        
        # initialization function: plot the background of each frame
        def init():
            # im.set_data(np.random.random((5,5)))
            for im, d in zip(ims, imgdatasource):
                im.set_data(d[0])
            lines[0].set_data(np.arange(0, 10), [abs(e)[~np.isnan(e)].mean() for e in evaluator_pred.error_list[:10]])
            lines[1].set_data(np.arange(0, 10), [abs(e)[~np.isnan(e)].mean() for e in evaluator_pc.error_list[:10]])
            return ims+lines

        # animation function.  This is called sequentially
        def animate(i):
            i = i%len(imgdatasource[0])
            for im, d in zip(ims, imgdatasource):
                im.set_data(d[i])
            lines[0].set_data(np.arange(0, min(100, errer_list_count[i])), [abs(e)[~np.isnan(e)].mean() 
                for e in evaluator_pred.error_list[max(0, errer_list_count[i]-100):errer_list_count[i]]])
            lines[1].set_data(np.arange(0, min(100, errer_list_count[i])), [abs(e)[~np.isnan(e)].mean() 
                for e in evaluator_pc.error_list[max(0, errer_list_count[i]-100):errer_list_count[i]]])
            return ims+lines

        anim = animation.FuncAnimation(
                                fig, 
                                func = animate, 
                                init_func = init,
                                frames = len(imgdatasource[0]),
                                interval = 1000 / 5, # in ms
                                )
        anim.save(os.path.join(outputdir, 'test_anim.mp4'))

    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1 import ImageGrid
    # fig, axs = plt.subplots(3,3, figsize=(30,30))
    fig = plt.figure(figsize = (10,8))
    row_rmse = ImageGrid(fig, 311,
                nrows_ncols = (1,4),
                axes_pad = 0.05,
                cbar_location = "right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.05
    )
    row_var = ImageGrid(fig, 312,
                nrows_ncols = (1,4),
                axes_pad = 0.05,
                cbar_location = "right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.05
    )
    row_count = ImageGrid(fig, 313,
                nrows_ncols = (1,4),
                axes_pad = 0.05,
                cbar_location = "right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad=0.05
    )


    ## Error maps
    row_rmse[0].imshow(evaluator_pred.get_err_map()[::-1,::-1], vmin=0, vmax=0.5, cmap='plasma')
    row_rmse[1].imshow(evaluator_pc.get_err_map()[::-1,::-1], vmin=0, vmax=0.5, cmap='plasma')
    row_rmse[2].imshow(evaluator_smooth.get_err_map()[::-1,::-1], vmin=0, vmax=0.5, cmap='plasma')
    imc = row_rmse[3].imshow(evaluator_fh.get_err_map()[::-1,::-1], vmin=0, vmax=0.5, cmap='plasma')
    [r.axis('off') for r in row_rmse]
    # cbar = plt.colorbar(imc, cax=row_rmse.cbar_axes[0])
    # cbar.ax.tick_params(labelsize=28)
    ticks = np.linspace(0,0.5,2)
    cb = plt.colorbar(imc, cax=row_rmse.cbar_axes[0], ticks=list(ticks))
    cb.ax.set_yticklabels(['0', f'{ticks[1]}'])
    cb.ax.tick_params(labelsize=16, rotation = -90, length = 14, width = 2, direction="inout")

    ## Variations
    maxvar = max(evaluator_pred.get_max_var(),evaluator_pc.get_max_var(),evaluator_fh.get_max_var())
    row_var[0].imshow(evaluator_pred.get_var_map()[::-1,::-1], vmin=0, vmax=maxvar, cmap='plasma')
    row_var[1].imshow(evaluator_pc.get_var_map()[::-1,::-1], vmin=0, vmax=maxvar, cmap='plasma')
    row_var[2].imshow(evaluator_smooth.get_var_map()[::-1,::-1], vmin=0, vmax=maxvar, cmap='plasma')
    imc = row_var[3].imshow(evaluator_fh.get_var_map()[::-1,::-1], vmin=0, vmax=maxvar, cmap='plasma')
    [r.axis('off') for r in row_var]
    # cbar = plt.colorbar(imc, cax=row_var.cbar_axes[0])
    # cbar.ax.tick_params(labelsize=28)
    ticks = np.linspace(0,maxvar,2)
    cb = plt.colorbar(imc, cax=row_var.cbar_axes[0], ticks=list(ticks))
    cb.ax.set_yticklabels(['0', f' {int(ticks[1])}'])
    cb.ax.tick_params(labelsize=16, rotation = -90, length = 14, width = 2, direction="inout")


    ## Counts
    maxcount = evaluator_pred.get_max_count()
    row_count[0].imshow(evaluator_pred.error_count[::-1,::-1], vmin=0, vmax=maxcount, cmap='plasma')
    row_count[1].imshow(evaluator_pc.error_count[::-1,::-1], vmin=0, vmax=maxcount, cmap='plasma')
    row_count[2].imshow(evaluator_smooth.error_count[::-1,::-1], vmin=0, vmax=maxcount, cmap='plasma')
    imc = row_count[3].imshow(evaluator_fh.error_count[::-1,::-1], vmin=0, vmax=maxcount, cmap='plasma')
    [r.axis('off') for r in row_count]
    # cbar = plt.colorbar(imc, cax=row_count.cbar_axes[0])
    # cbar.ax.tick_params(labelsize=28)
    ticks = np.linspace(0,maxcount,2)
    cb = plt.colorbar(imc, cax=row_count.cbar_axes[0], ticks=list(ticks))
    cb.ax.set_yticklabels(['0', f' {int(ticks[1])}'])
    cb.ax.tick_params(labelsize=16, rotation = -90, length = 14, width = 2, direction="inout")

    plt.subplots_adjust(
        hspace = 0.05
    )
    plt.savefig(os.path.join(outputdir, "error_map2.png"), bbox_inches='tight', pad_inches=0.1)
    # plt.subplot_tool()

    # plt.show()

    # plt.figure()
    # plt.imshow(evaluator_pred.get_err_map() - evaluator_pc.get_err_map(), cmap='plasma')
    # plt.colorbar()
    # plt.savefig(os.path.join(outputdir, "error_diff.png"), bbox_inches='tight', pad_inches=0)
    # plt.show()

    with open(os.path.join(outputdir, "eval_result.txt"), "w")as f: 
        f.write("ours rmse: %.3f\n"%(evaluator_pred.get_rmse()))
        f.write("pc rmse: %.3f\n"%(evaluator_pc.get_rmse()))
        f.write("smooth rmse: %.3f\n"%(evaluator_smooth.get_rmse()))
        f.write("fh rmse: %.3f\n"%(evaluator_fh.get_rmse()))
        f.write("ours err var: %.3f\n"%(evaluator_pred.get_errvar()))
        f.write("pc err var: %.3f\n"%(evaluator_pc.get_errvar()))
        f.write("smooth err var: %.3f\n"%(evaluator_smooth.get_errvar()))
        f.write("fh err var: %.3f\n"%(evaluator_fh.get_errvar()))
        f.write("ours mean err: %.3f\n"%(evaluator_pred.get_mean_err()))
        f.write("pc mean err: %.3f\n"%(evaluator_pc.get_mean_err()))
        f.write("smooth mean err: %.3f\n"%(evaluator_smooth.get_mean_err()))
        f.write("fh mean err: %.3f\n"%(evaluator_fh.get_mean_err()))

if __name__ == "__main__":
    for m in ["2023-02-28-12-00-40_fixed"]:#,"2022-09-05-23-28-07", "2022-09-06-00-11-32"]:
        main(m, overwrite=True)
    # from glob import glob
    # for m in glob("checkpoints/2022-08-29-*"):
    #     print("Running model:",m)
    #     main(os.path.basename(m), overwrite = False)
