"""
Chenyu Yang Jun 3
This file is used to extract the predicted height point clouds from ros bag
"""
#!/usr/bin/env python

import os

import pandas
import rospy
import rospkg
from ruamel.yaml import YAML
from argparse import ArgumentParser
import cv2
from sensor_msgs.msg import Image
import rosbag

from cv_bridge import CvBridge, CvBridgeError
import tf2_py
from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_from_matrix, quaternion_from_matrix

from numpy import savetxt
import numpy as np
import matplotlib.pyplot as plt  # Just for debug.

import msgpack
import msgpack_numpy as m

import math
import time

import collections

from messages.gridMapMessage import GridMapFromMessage
from messages.imageMessage import Camera, getImageId, rgb_msg_to_image
from messages.pointcloudMessage import rospcmsg_to_pcarray, ros_pc_msg2
from messages.messageToVectors import msg_to_body_ang_vel, msg_to_body_lin_vel, msg_to_rotmat, msg_to_command, \
    msg_to_pose, msg_to_joint_positions, msg_to_joint_velocities, msg_to_joint_torques, msg_to_grav_vec

m.patch()

import pandas as pd
import subprocess
import yaml

from ExtractDepthImage import DIFG
from GroundfromTrajs import GFT

import pickle as pkl

# https://stackoverflow.com/questions/41493282/in-python-pandas-how-can-i-re-sample-and-interpolate-a-dataframe
# https://stackoverflow.com/questions/48068938/set-new-index-for-pandas-dataframe-interpolating
def reindex_and_interpolate(df, new_index):
    return df.reindex(df.index.union(new_index)).interpolate(method='index', limit_direction='both').loc[new_index]


def reindex_union(df, new_index, method='nearest'):
    # return df.reindex(df.index.union(new_index), method=method)
    return df.reindex(new_index, method=method, tolerance=0.5)


# From rosbag_pandas
def get_bag_info(bag_file):
    '''Get uamle dict of the bag information
    by calling the subprocess -- used to create correct sized
    arrays'''
    # Get the info on the bag
    bag_info = yaml.load(subprocess.Popen(
        ['rosbag', 'info', '--yaml', bag_file],
        stdout=subprocess.PIPE).communicate()[0])
    return bag_info


def get_length(topics, yaml_info):
    '''
    Find the length (# of rows) in the created dataframe
    '''
    length_per_topic = {}
    total = 0
    info = yaml_info['topics']
    for topic in topics:
        for t in info:
            if t['topic'] == topic:
                total = total + t['messages']
                length_per_topic[topic] = t['messages']
                break
    return total, length_per_topic




class DataBuffer:
    def __init__(self):
        self.indices = {}
        self.data = {}
        self.data_id = {}

    def append(self, stamp, data, name):

        if name in self.data:
            self.data[name].append(data)
            self.indices[name].append(stamp)
            self.data_id[name].append(self.data_id[name][-1] + 1)
        else:
            self.data[name] = [data]
            self.indices[name] = [stamp]
            self.data_id[name] = [0]


def extractAndSyncTrajs(file_name, out_dir, cfg, cameras):
    # Make sure output directory exists.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load ground surface estimator
    GroundMap_filepath = os.path.join(out_dir, "GroundMap.msgpack")
    
    ## if the map file not exist, then generate it
    FeetTrajs_filepath = os.path.join(out_dir, "FeetTrajs.msgpack")
    assert os.path.exists(FeetTrajs_filepath) 
    gft = GFT(FeetTrajsFile = FeetTrajs_filepath, InitializeGP=False)
    foot_holds = {k : np.array(gft.getContactPoints(v)[0]) for k,v in gft.FeetTrajs.items()} # A dict of the contact points of each foot
    # for value in gft.FeetTrajs.values():
    #     newPoints, _ = self.getContactPoints(value)
    # gft.save(out_dir, GPMap=True)
    

    # Load parameters
    ELEVATION_MAP_TOPIC = cfg['ELEVATION_MAP_TOPIC']
    CAM_PREFIX = cfg['CAM_PREFIX']
    CAM_NAMES = cfg['CAM_NAMES']

    STATE_TOPIC = cfg['STATE_TOPIC']
    COMMAND_TOPIC = cfg['COMMAND_TOPIC']

    TF_BASE = cfg['TF_BASE']
    TF_POSE_REF_LIST = cfg['TF_POSE_REF_LIST']

    POINT_CLOUD_SUFFIX = cfg['POINT_CLOUD_SUFFIX']
    POINT_CLOUD_NAMES = cfg['POINT_CLOUD_NAMES']


    state_topics = [STATE_TOPIC, COMMAND_TOPIC]
    img_topics = [ELEVATION_MAP_TOPIC]

    img_topics += [cam.topic_id for cam in cameras.values()]
    pointcloud_topics = [name+POINT_CLOUD_SUFFIX for name in POINT_CLOUD_NAMES]
    pointcloud_topics.append("/pred_pc")
    elev_topics = [ELEVATION_MAP_TOPIC]

    dt = rospy.Duration.from_sec(cfg['dt'])
    proprio_dt = rospy.Duration.from_sec(cfg['proprioception_dt'])

    resample_freq = str(dt.to_sec()) + ' S'
    proprioception_resample_freq = str(cfg['proprioception_dt']) + ' S'
    proprioception_history_half_length = int(cfg['proprioception_history_half_length'])
    proprioception_decimation = math.ceil(cfg['dt']/cfg['proprioception_dt'])

    time_margin = cfg['proprioception_dt'] * cfg['proprioception_history_half_length']
    skip_steps = math.ceil(time_margin / dt.to_sec())

    print("skip_steps", skip_steps)
    print("proprioception_decimation", proprioception_decimation)

    local_map_shape = cfg['local_map_shape']

    # dump YAML CFG
    cfg['bag_file_name'] = file_name
    with open(out_dir + "/data_extraction.yaml", 'w') as yaml_file:
        YAML().dump(cfg, yaml_file)

    # yaml_info = get_bag_info(file_name)

    print('Parsing bag \'' + file_name + '\'...')
    bag = rosbag.Bag(file_name)

    command_stamps = []

    print("Check command sequences")
    for topic, msg, t in bag.read_messages(topics=COMMAND_TOPIC):
        command = msg_to_command(msg)  # in baseframe
        if np.sum(np.abs(command)) > 0:
            command_stamps.append(msg.header.stamp)

    command_stamps = list(set(command_stamps))  # remove duplicates
    command_stamps.sort()

    # Divide command_stamps into sub sequences
    time_stamps_sequences = []
    continuous_stamps = []

    min_traj_len = cfg['min_traj_len'] * dt.to_sec()
    max_traj_len = cfg['max_traj_len'] * dt.to_sec()

    
    for i in range(1, len(command_stamps)):
        dt_ = (command_stamps[i] - command_stamps[i - 1]).to_sec()

        seq_length = 0
        if len(continuous_stamps) > 1:
            seq_length = (continuous_stamps[-1] - continuous_stamps[0]).to_sec()

        if dt_ > cfg['dt'] or seq_length >= max_traj_len or i == (len(command_stamps) - 1):
            if seq_length >= min_traj_len:
                time_stamps_sequences.append(continuous_stamps)
            continuous_stamps = []
        continuous_stamps.append(command_stamps[i])

    print("Number of sub sequences: {}".format(len(time_stamps_sequences)))

    tf_buffer = tf2_py.BufferCore(rospy.Duration.from_sec(max_traj_len * dt.to_sec() * 1.1))

    for topic, msg, t in bag.read_messages(topics=['/tf_static']):
        for transform in msg.transforms:
            tf_buffer.set_transform_static(transform, 'rosbag')

    for cam_id in cfg['CAM_NAMES']:
        cameras[cam_id].update_static_tf(
            tf_buffer.lookup_transform_core(TF_BASE, cameras[cam_id].frame_key, time_stamps_sequences[0][0]))

    for topic, msg, t in bag.read_messages(topics=['/tf'], end_time=time_stamps_sequences[0][0]):
        for transform in msg.transforms:
            if transform.header.frame_id in TF_POSE_REF_LIST:
                tf_buffer.set_transform(transform, 'rosbag')

    # Process trajectories
    for traj_idx, traj in enumerate(time_stamps_sequences[2:]):
        print(traj_idx, "-th sequence: ")

        start_time = traj[0]
        end_time = traj[-1]

        # first fill up tf buffer.
        for topic, msg, t in bag.read_messages(topics=['/tf'], start_time=start_time,
                                               end_time=end_time):
            for transform in msg.transforms:
                if transform.header.frame_id in TF_POSE_REF_LIST:
                    tf_buffer.set_transform(transform, 'rosbag')

        # Update states
        command_data = DataBuffer()
        pose_data = DataBuffer()
        velocity_data = DataBuffer()
        state_data = DataBuffer()
        image_data = DataBuffer()
        map_data = DataBuffer()
        pointcloud_data = DataBuffer()
        
        footHoldHeights_data = DataBuffer()

        if cfg['save_proprioception']:
            proprio_data = DataBuffer()

        for topic, msg, t in bag.read_messages(topics=state_topics + pointcloud_topics + elev_topics, start_time=start_time,
                                               end_time=end_time):

            # ANYmal state
            if topic == STATE_TOPIC:
                lin_vel = msg_to_body_lin_vel(msg)  # in baseframe
                ang_vel = msg_to_body_ang_vel(msg)
                joint_pos = msg_to_joint_positions(msg)
                joint_vel = msg_to_joint_velocities(msg)
                joint_tor = msg_to_joint_torques(msg)
                e_g = msg_to_grav_vec(msg)

                vel_concat = np.concatenate([lin_vel, ang_vel])

                velocity_data.append(msg.header.stamp.to_sec(), vel_concat, 'base')

                state_data.append(msg.header.stamp.to_sec(), joint_pos, 'joint_position')

                if cfg['save_proprioception']:
                    proprio_state = np.concatenate([e_g, vel_concat, joint_pos, joint_vel, joint_tor])
                    proprio_data.append(msg.header.stamp.to_sec(), proprio_state, 'proprio')


            # Elevation map
            if ELEVATION_MAP_TOPIC in topic:
                grid_map = GridMapFromMessage(msg, cfg['map_idx'])
                map_data.append(msg.info.header.stamp.to_sec(), grid_map, 'elevation_map')

            if POINT_CLOUD_SUFFIX in topic:
                map_frame_id = "map"
                if not (tf_buffer.can_transform_core(map_frame_id, msg.header.frame_id,  msg.header.stamp)[0]): continue
                tf = tf_buffer.lookup_transform_core(map_frame_id, msg.header.frame_id,  msg.header.stamp)
                pose = msg_to_pose(tf)
                pc_array = rospcmsg_to_pcarray(msg, pose)
                success, cam_id = getImageId(topic, POINT_CLOUD_NAMES)
                if not success:
                    print('Unknown point cloud topic: ' + topic)
                    continue
                pointcloud_data.append(msg.header.stamp.to_sec(), pc_array,  cam_id)

            if "pred_pc" in topic:
                if (not len(pointcloud_data.data.keys())): continue # skip if there is no point cloud value
                tf = tf_buffer.lookup_transform_core("map", TF_BASE, t)
                pred_pc = np.array(list(ros_pc_msg2.read_points(msg, skip_nans=True)))
                pose = np.array(msg_to_pose(tf))
                footHoldHeights_data.append(t.to_sec(), pose, "pose")
                for foot_hold_k, foot_hold_v in foot_holds.items():
                    # print("pose", pose) # 7x1 array
                    # print("foot_hold_v type", type(foot_hold_v)) # class numpy.npdarray
                    # print("foot_hold_v shape", np.array(foot_hold_v).shape) # (117630, 3)
                    # print("pred_pc type", type(pred_pc)) # numpy.ndarray
                    # print("pred_pc shape", pred_pc.shape) # (59655, 3)
                    foot_hold_distance = np.sqrt(np.sum((foot_hold_v[:,:2] - pose[:2])**2, axis = 1))

                    QueryFootHoldD = 5
                    foot_hold_distance[foot_hold_distance<QueryFootHoldD] = np.inf
                    foothold_sort_idx = np.argsort(foot_hold_distance)
                    for foothold_idx in foothold_sort_idx[:200]:
                        foot_hold_pos = foot_hold_v[foothold_idx]

                        ## Calculate prediction pose from foot_hold_pos
                        dist_pred_pc_foot_hold = np.sqrt(np.sum((pred_pc[:,:2] - foot_hold_pos[:2])**2, axis = 1))
                        k = 4
                        pred_pc_mask = np.argpartition(dist_pred_pc_foot_hold, k)[:k]
                        if(np.sum(dist_pred_pc_foot_hold[pred_pc_mask] > 5)): # no valid point cloud found                            
                            # print("min point cloud distances:", dist_pred_pc_foot_hold[pred_pc_mask])
                            continue
                        pred_pc_poses = pred_pc[pred_pc_mask]
                        w = pred_pc_poses[:,:2] @ np.linalg.inv(pred_pc_poses[:,:2].T @ pred_pc_poses[:,:2]) @ foot_hold_pos[:2]
                        pred_pc_pos = pred_pc_poses.T @ w 
                        break # this if no break happens, enter the `for else` clause
                    else:
                        print("No foot hold covered")
                        continue

                    ## Calculate prediction pose from lidar points
                    # print({k:v[-1].shape for k,v in pointcloud_data.data.items()}) # (32185,)
                    pc = np.concatenate([v[-1] for v in pointcloud_data.data.values()])
                    dist_pc_foot_hold = np.sqrt(np.sum((pc[:,:2] - foot_hold_pos[:2])**2, axis = 1))
                    k = 10
                    pc_mask = np.argpartition(dist_pc_foot_hold, k)[:k]
                    if(np.sum(dist_pc_foot_hold[pc_mask] > 10)): # no valid point cloud found
                        print("foot hold not covered")
                        print("min point cloud distances:", dist_pc_foot_hold[pc_mask])
                        continue
                    pc_poses = pc[pc_mask]
                    w = pc_poses[:,:2] @ np.linalg.inv(pc_poses[:,:2].T @ pc_poses[:,:2]) @ foot_hold_pos[:2]
                    pc_pos = pc_poses.T @ w 

                    footHoldHeights_data.append(t.to_sec(), foot_hold_pos, foot_hold_k)
                    footHoldHeights_data.append(t.to_sec(), pred_pc_pos, "prediction_%s"%foot_hold_k)
                    footHoldHeights_data.append(t.to_sec(), pc_pos, "pointcloud_%s"%foot_hold_k)


                    ## Calculate the prediction pose from height map
                    if(len(map_data.data.keys())):

                        elev_map = map_data.data["elevation_map"][-1]
                        if not (tf_buffer.can_transform_core(map_frame_id, elev_map.frame_id,  t)[0]): continue
                        tf = tf_buffer.lookup_transform_core(map_frame_id, elev_map.frame_id,  t)
                        elev_map_pose = msg_to_pose(tf)
                        position = np.array(elev_map_pose[:3])
                        R = np.array(quaternion_matrix(elev_map_pose[3:]))[:3,:3]
                        # elev_pos = foot_hold_pos[:]
                        elev_pos = np.linalg.inv(R) @ (foot_hold_pos - position)
                        elev_map_ind = elev_map.getIndexFromPosition(elev_pos[:2])
                        try:
                            elev_pos[2] = elev_map.at(elev_map_ind)
                            # elev_pos_W = elev_pos[:]
                            elev_pos_W = R @ elev_pos + position
                            footHoldHeights_data.append(t.to_sec(), elev_pos_W, "heightmap_%s"%foot_hold_k)
                        except Exception as e:
                            print(e)
                            

        _name = "traj_" + str(traj_idx)
        out_file = os.path.join(out_dir, _name)
        out_file += '.pkl'
        with open(out_file,"wb") as f:
            pkl.dump(footHoldHeights_data, f)

def main():
    # Load cfg
    cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)) , 'data_extraction_SA.yaml')
    print("cfg_path :",cfg_path)
    parser = ArgumentParser()
    parser.add_argument('--cfg_path', default=cfg_path, help='Directory where data will be saved.')
    args = parser.parse_args()
    cfg_path = args.cfg_path

    cfg = YAML().load(open(cfg_path, 'r'))

    bag_file_path = cfg['bagfile']
    output_path = cfg['outdir']
    camera_calibration_path = cfg['calibration']
    print("camera_calibration_path :",camera_calibration_path)
    
    isdir = os.path.isdir(bag_file_path)

    # Load camera info
    cameras = {}
    for cam_id in cfg['CAM_NAMES']:
        cameras[cam_id] = Camera(camera_calibration_path,
                                 cam_id,
                                 cfg)



    # Get all bag files
    bagfiles = []
    if isdir:
        for root, dirs, files in os.walk(bag_file_path):
            for file in files:
                if file.endswith('.bag'):
                    bagfiles.append(os.path.join(root, file))
    else:
        assert bag_file_path.endswith('.bag'), 'Specified file is not a bag file.'
        bagfiles.append(bag_file_path)

    for i, bagfile in enumerate(bagfiles):
        print('Extracting file ' + str(i + 1) + '/' + str(len(bagfiles)))
        # Get file basename for subfolder.
        basename = os.path.splitext(os.path.basename(bagfile))[0]
        # print(basename)
        extractAndSyncTrajs(bagfile, os.path.join(output_path, basename), cfg, cameras)

    print("DONE!")


if __name__ == '__main__':
    main()
