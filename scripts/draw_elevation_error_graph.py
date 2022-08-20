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
from semantic_front_end_filter.adabins.model_io import load_checkpoint
from semantic_front_end_filter.adabins.elevation_vis import WorldViewElevationMap
from semantic_front_end_filter.adabins.elevation_eval_util import ElevationMapEvaluator

class RosbagPlayer:
    """
    This 'simulator' holds a rosbag, 
        execute registered callback functions when play
        also holds tf_buffer to facilitate an easy look up
    Note: this is only for processing bag data, not for adding new topics and messages
    """
    def __init__(self, rosbagpath):

        self.bag = rosbag.Bag(rosbagpath)
        duration = self.bag.get_end_time() - self.bag.get_start_time()
        self.tf_buffer = tf2_py.BufferCore(rospy.Duration(duration))
        
        # read the whole tf history
        for topic, msg, t in self.bag.read_messages(topics=['/tf_static']):
            for transform in msg.transforms:
                self.tf_buffer.set_transform_static(transform, 'rosbag')

        tf_times = []
        for topic, msg, t in self.bag.read_messages(topics=['/tf']):
            for transform in msg.transforms:
                self.tf_buffer.set_transform(transform, 'rosbag')
                tf_times.append(transform.header.stamp)

        self._callbacks = {}

    def register_callback(self, topic, func):
        """
        arg topic: the topic of the callback function
        arg func: have the signature: (topic, msg, t, tf_buffer) 
        """
        self._callbacks[topic] = func

    def play(self, start_time=None, end_time=None):
        """
        Play the rosbag and call the callbacks
        """
        start_time = start_time if start_time is None else rospy.Time(start_time)
        end_time = end_time if end_time is None else rospy.Time(end_time)
        for topic, msg, t in self.bag.read_messages(
            topics=list(self._callbacks.keys()),
            start_time=start_time, end_time=end_time):
            self._callbacks[topic](topic, msg, t, self.tf_buffer)


def load_param_from_path(data_path):
    model_cfg = YAML().load(open(os.path.join(data_path, "ModelConfig.yaml"), 'r'))
    return model_cfg


if __name__ == "__main__":

    rosbagpath = "/Data/20211007_SA_Monkey_ANYmal_Chimera/chimera_mission_2021_10_09/mission8_locomotion/Reconstruct_2022-04-25-19-10-16_0.bag"
    foottrajpath = "/Data/extract_trajectories_002/Reconstruct_2022-04-25-19-10-16_0/FeetTrajs.msgpack"
    groundmappath = "/Data/extract_trajectories_002/Reconstruct_2022-04-25-19-10-16_0/GroundMap.msgpack"
    model_path = "checkpoints/2022-08-03-16-26-08/UnetAdaptiveBins_latest.pt"
    image_topic = "/alphasense_driver_ros/cam4/dropped/debayered/compressed"
    pc_topic = "/bpearl_rear/point_cloud"
    TF_BASE = "base"
    TF_MAP = "map"
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raycastCamera = RaycastCamera(device=device)
    elevation_pred = WorldViewElevationMap(resolution = 0.1, map_length = 10, init_with_initialize_map = False)
    elevation_pc = WorldViewElevationMap(resolution = 0.1, map_length = 10, init_with_initialize_map = False)
    # Evaluators used to evaluate the error
    evaluator_pred = ElevationMapEvaluator(groundmappath, elevation_pred.param)
    evaluator_pc = ElevationMapEvaluator(groundmappath, elevation_pc.param)
    player = RosbagPlayer(rosbagpath)

    ## Initialize model
    model_cfg = load_param_from_path(os.path.dirname(model_path))
    model_cfg={k:v for k,v in model_cfg.items() if k in [
        "n_bins", "min_val", "max_val", "norm", "use_adabins", "deactivate_bn", "skip_connection"
    ]}
    model = UnetAdaptiveBins.build(input_channel = 4, **model_cfg)

    model = load_checkpoint(model_path, model)[0]
    model.to(device)

    ## Define shared variables
    pcbuffer=[]

    ###########
    ## Definining callbacks

    def pred_and_checkerr(image, pc, pose):
        """
        Make the prediction based on image, pointcloud and robot current pose
        arg pose: x,y,z,rx,ry,rz,rw
        """
        image = torch.Tensor(image).to(device)
        points = torch.Tensor(pc).to(device)
        # pose = torch.Tensor(pose).to(device)
        # get pc image
        pc_img = torch.zeros_like(image[:1, ...]).to(device).float()
        pc_img,visible = raycastCamera.project_cloud_to_depth(
                        pose, points, pc_img, return_visible=True)
        # filter the points to the only visble ones
        points = points[visible] # TODO: Decide whether should be filter the raw points
        # get prediction
        model_in = torch.cat([image/255., pc_img],axis=0)
        model_in = model_in[None, ...]
        for i, (m, s) in enumerate(zip([0.387, 0.394, 0.404, 0.120], [0.322, 0.32, 0.30,  1.17])):
            model_in[0, i, ...] = (model_in[0, i, ...] - m)/s
        pred = model(model_in)[0][0]

        pred [(pc_img[0]==0)] = np.nan
        pred = pred.T

        # get elevation from prediction
        pred_pts = raycastCamera.project_depth_to_cloud(pose, pred)
        pred_pts = pred_pts[~torch.isnan(pred_pts[:,0])]
        # pred_pts = pred_pts[pred_pts[:,2]<pose[2]] # TODO: Decide whether this hieight mask is necessary
        pred_points = pred_pts.detach().cpu().numpy()
        points = points.cpu().numpy().astype(pred_points.dtype) # float_64
        # points = points[points[:,2]<pose[2]] # TODO: Decide whether this hieight mask is necessary

        elevation_pred.move_to_and_input(pose[0:3], pred_points)
        elevmap_pred = elevation_pred.get_elevation_map()
        elevation_pc.move_to_and_input(pose[0:3], points)
        elevmap_pc = elevation_pc.get_elevation_map()
        
        rz = Rotation.from_quat(pose[3:]).as_euler('xyz',degrees=False)[2]

        error_pred = evaluator_pred.compute_error_against_gpmap(elevmap_pred, pose[:2], rz)
        error_pc = evaluator_pc.compute_error_against_gpmap(elevmap_pc, pose[:2], rz)
        
        ## Breakpoint vis
        # gt = evaluator_pred.get_gpmap_at_xy(pose[:2])
        # plt.figure()
        # plt.imshow(error_pred, vmin=-1, vmax=1)
        # error_pred_mask = ~np.isnan(error_pred)
        # plt.title("pred_rmse: %f"%(np.sqrt(np.sum(error_pred[error_pred_mask]**2))/error_pred_mask.sum()))
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(error_pc, vmin=-1, vmax=1)
        # error_pc_mask = ~np.isnan(error_pc)
        # plt.title("pc_rmse: %f"%(np.sqrt(np.sum(error_pc[error_pc_mask]**2))/error_pc_mask.sum()))
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(gt)
        # plt.show()


    def image_cb(topic, msg, t, tf_buffer):
        global pcbuffer
        if(not len(pcbuffer)): return
        
        img = rgb_msg_to_image(msg, raycastCamera.camera.is_debayered, raycastCamera.camera.rb_swap, ("compressed" in topic))
        img = np.moveaxis(img, 2, 0)

        if not (tf_buffer.can_transform_core(TF_MAP, TF_BASE,  msg.header.stamp)[0]): return 
        tf = tf_buffer.lookup_transform_core(TF_MAP, TF_BASE, msg.header.stamp)
        pose = msg_to_pose(tf)  # pose in fixed ref frame (odom or map)

        pc = np.concatenate(pcbuffer,axis = 0)
        pred_and_checkerr(img, pc, pose)

        pcbuffer = pcbuffer[-1:]

        ## Breakpoint vis
        # plt_img = np.moveaxis(img[:3,...], 0, 2)
        # plt_img = (plt_img-plt_img.min())/(plt_img.max()-plt_img.min())
        # plt.imshow(plt_img[:,:,::-1])
        # plt.show()
    player.register_callback(image_topic, image_cb)

    
    def pointcloud_cb(topic, msg, t, tf_buffer):
        global pcbuffer
        if not (tf_buffer.can_transform_core(TF_MAP, msg.header.frame_id,  msg.header.stamp)[0]): return
        tf = tf_buffer.lookup_transform_core(TF_MAP, msg.header.frame_id,  msg.header.stamp)
        pose = msg_to_pose(tf)
        pc_array = rospcmsg_to_pcarray(msg, pose)

        pcbuffer.append(pc_array[:,:3])
    player.register_callback(pc_topic, pointcloud_cb)

    player.play()# play from start to end
    # player.play(end_time=player.bag.get_start_time()+100)


    meanerr_pred = evaluator_pred.error_sum / evaluator_pred.error_count
    plt.figure()
    plt.imshow(meanerr_pred, vmin=0, vmax=2, cmap='plasma')
    error_pred_mask = ~np.isnan(meanerr_pred)
    plt.title("pred_rmse: %f"%(np.sqrt(np.sum(meanerr_pred[error_pred_mask]**2))/float(error_pred_mask.sum())))
    plt.colorbar()

    meanerr_pc = evaluator_pc.error_sum / evaluator_pc.error_count
    plt.figure()
    plt.imshow(meanerr_pc, vmin=0, vmax=2, cmap='plasma')
    error_pc_mask = ~np.isnan(meanerr_pc)
    plt.title("pc_rmse: %f"%(np.sqrt(np.sum(meanerr_pc[error_pc_mask]**2))/float(error_pc_mask.sum())))
    plt.colorbar()
    plt.show()