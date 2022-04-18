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

from numpy import empty, savetxt
import numpy as np
import matplotlib.pyplot as plt  # Just for debug.

import msgpack
import msgpack_numpy as m

import math
import raisimpy
import time

import collections

from messages.gridMapMessage import GridMapFromMessage
from messages.imageMessage import Camera, getImageId, rgb_msg_to_image
from messages.messageToVectors import msg_to_body_ang_vel, msg_to_body_lin_vel, msg_to_rotmat, msg_to_command, \
    msg_to_pose, msg_to_joint_positions, msg_to_joint_velocities, msg_to_joint_torques, msg_to_grav_vec

m.patch()

import pandas as pd
import subprocess
import yaml


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


def resample_ffill(data, resampled_idx, tolerance, additional = None):
    dfs = []
    for key in data.data_id.keys():
        df = pd.DataFrame(data=data.data[key], index=data.indices[key])
        df.index = pd.to_datetime(df.index, unit='s')
        df = df[~df.index.duplicated(keep='first')]
        df = df.reindex(resampled_idx, method='ffill', tolerance=str(tolerance) + 's')  # ffill
        dfs.append(df)
    keys = list(data.data_id.keys())

    if additional is not None:
        keys.append(additional[0])
        dfs.append(additional[1])

    return pd.concat(dfs, axis=1, keys=keys)


def resample_dataid_ffill(data, resampled_idx, tolerance):
    dfs = []
    for key in data.data_id.keys():
        df = pd.DataFrame(data=data.data_id[key], index=data.indices[key])
        df.index = pd.to_datetime(df.index, unit='s')
        df = df[~df.index.duplicated(keep='first')]
        df = df.reindex(resampled_idx, method='ffill', tolerance=str(tolerance) + 's')  # ffill
        dfs.append(df)
    return pd.concat(dfs, axis=1, keys=data.data_id.keys())

def ros_pos_to_raisim_gc(pose, joint_positions):
    q = np.zeros(19)
    q[:3] = pose[:3]
    q[3] = pose[6]
    q[4:7] = pose[3:6]
    q[7:] = joint_positions
    return q

class Data:
    def __init__(self):
        self.images = []
        self.poses = []


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


def extractFeetTrajs(file_name, out_dir, cfg, raisim_objects):
    STATE_TOPIC = cfg['STATE_TOPIC']
    TF_BASE = cfg['TF_BASE']
    TF_POSE_REF_LIST = cfg['TF_POSE_REF_LIST']

    # Make sure output directory exists.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('Parsing bag \'' + file_name + '\'...')
    bag = rosbag.Bag(file_name)

    # Dict stores feet trajectories
    footTrajectory = {}
    for foot_indicies in raisim_objects["foot_indices"]:
        footTrajectory[foot_indicies] = []
    
    # Get TF_Static & TF
    tf_buffer = tf2_py.BufferCore(rospy.Duration.from_sec((bag.get_end_time() - bag.get_start_time()) * 1.1))
    latestTFstamp = rospy.Time.from_sec(0)
    tfFlag = 0
    for topic, msg, t in bag.read_messages(topics=['/tf_static', '/tf']):
        if topic == 'tf_static':
            for transform in msg.transforms:
                tf_buffer.set_transform_static(transform, 'rosbag')
        else:
            for transform in msg.transforms:
                if transform.header.frame_id in TF_POSE_REF_LIST:
                    tf_buffer.set_transform(transform, 'rosbag')
            if(tfFlag==0):
                # Add offset in case STATE_TOPIC is much earlier than tf
                latestTFstamp = t + rospy.Duration.from_sec(2)
                tfFlag = 1

    # Get Trajectories
    basePose = []
    for topic, msg, t in bag.read_messages(topics=[STATE_TOPIC], start_time=latestTFstamp):
        # if(tf_buffer.canTransform('msf_body_imu_map', TF_BASE, t)):
        # tf = tf_buffer.lookup_transform_core('msf_body_imu_map', TF_BASE, t)
        try:
            tf = tf_buffer.lookup_transform_core('map', TF_BASE, t)
        except Exception as e:
            print(e)
            continue
        basePose = msg_to_pose(tf)
        # Update Pose & Get Feet Trajs
        if(basePose != []):
            if topic == STATE_TOPIC:
                joint_pos = msg_to_joint_positions(msg)
                q = ros_pos_to_raisim_gc(basePose, joint_pos)   
                # Rasim -> feet traj
                raisim_objects['anymal'].setGeneralizedCoordinate(q)
                raisim_objects['world'].integrate1()

                # get foot positions
                # TODO: filter out feet in contact using foot contact state
                for foot_idx in raisim_objects['foot_indices']:
                    footTrajectory[foot_idx].append((raisim_objects['anymal'].getFramePosition(foot_idx)).tolist())  
    
    # Convert to 2D np.array & Change indeicies to 
    for foot_index in raisim_objects["foot_indices"]:
        # footTrajectory[foot_index] = np.array(footTrajectory[foot_index])
        footTrajectory[raisim_objects['anymal'].getFrameByIdx(foot_index).name] = footTrajectory[foot_index]
        del footTrajectory[foot_index]

    
    # Save 
    out_file = out_dir+"/FeetTrajs.msgpack"
    with open(out_file, "wb") as outfile:
        file_dat = msgpack.packb(footTrajectory)
        outfile.write(file_dat)



def main():
    # Load cfg
    cfg_path = os.path.dirname(os.path.realpath(__file__)) + '/data_extraction_SA.yaml'
    parser = ArgumentParser()
    parser.add_argument('--cfg_path', default=cfg_path, help='Directory where data will be saved.')
    args = parser.parse_args()
    cfg_path = args.cfg_path

    cfg = YAML().load(open(cfg_path, 'r'))

    bag_file_path = cfg['bagfile']
    output_path = cfg['outdir']
    camera_calibration_path = cfg['calibration']

    isdir = os.path.isdir(bag_file_path)

    # Load camera info
    # cameras = {}
    # for cam_id in cfg['CAM_NAMES']:
    #     cameras[cam_id] = Camera(camera_calibration_path,
    #                              cam_id,
    #                              cfg)

    # SETUP ANYMAL SIMULATION FOR COLLISION DETECTION
    import raisimpy as raisim
    dir_path = os.path.dirname(os.path.realpath(__file__))
    urdf_path = dir_path + '/../rsc/robot/chimera/urdf/anymal_minimal.urdf'  # TODO: load from sim2real repo
    raisim_objects = {}
    raisim_objects['world'] = raisim.World()
    raisim_objects['anymal'] = raisim_objects['world'].addArticulatedSystem(urdf_path)
    raisim_objects['terrain'] = None

    raisim_objects['foot_indices'] = []
    raisim_objects['foot_indices'].append(raisim_objects['anymal'].getFrameIdxByName("LF_shank_fixed_LF_FOOT"))
    raisim_objects['foot_indices'].append(raisim_objects['anymal'].getFrameIdxByName("RF_shank_fixed_RF_FOOT"))
    raisim_objects['foot_indices'].append(raisim_objects['anymal'].getFrameIdxByName("LH_shank_fixed_LH_FOOT"))
    raisim_objects['foot_indices'].append(raisim_objects['anymal'].getFrameIdxByName("RH_shank_fixed_RH_FOOT"))

    # launch raisim server (for visualization)
    if cfg['visualize']:
        raisim_objects['server'] = raisim.RaisimServer(raisim_objects['world'])
        raisim_objects['server'].launchServer(8080)

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
        # extractAndSyncTrajs(bagfile, os.path.join(output_path, basename), cfg, cameras, raisim_objects)
        
        extractFeetTrajs(bagfile, os.path.join(output_path, basename), cfg, raisim_objects)

    print("DONE!")


if __name__ == '__main__':
    main()
