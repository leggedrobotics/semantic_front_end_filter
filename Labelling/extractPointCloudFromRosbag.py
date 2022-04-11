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
import raisimpy
import time

import collections

from messages.gridMapMessage import GridMapFromMessage
from messages.imageMessage import Camera, getImageId, rgb_msg_to_image
from messages.pointcloudMessage import rospcmsg_to_pcarray
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


def extractAndSyncTrajs(file_name, out_dir, cfg, cameras, raisim_objects):
    # Make sure output directory exists.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

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

        if cfg['save_proprioception']:
            proprio_data = DataBuffer()

        for topic, msg, t in bag.read_messages(topics=img_topics + state_topics + pointcloud_topics, start_time=start_time,
                                               end_time=end_time):

            # command
            if topic == COMMAND_TOPIC:
                command = msg_to_command(msg)  # in baseframe
                command_data.append(msg.header.stamp.to_sec(), command, 'base')

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

            # RGB images.
            if CAM_PREFIX in topic:
                success, cam_id = getImageId(topic, CAM_NAMES)
                img = rgb_msg_to_image(msg, cameras[cam_id].is_debayered, cameras[cam_id].rb_swap, ("compressed" in topic))

                if success:
                    image_data.append(msg.header.stamp.to_sec(), img, cam_id)
                else:
                    print('Unknown RGB image topic: ' + topic)

            # Elevation map
            if ELEVATION_MAP_TOPIC in topic:
                grid_map = GridMapFromMessage(msg, cfg['map_idx'])
                map_data.append(msg.info.header.stamp.to_sec(), grid_map, 'elevation_map')

            if POINT_CLOUD_SUFFIX in topic:
                map_frame_id = "msf_body_imu_map"
                if not (tf_buffer.can_transform_core(map_frame_id, msg.header.frame_id,  msg.header.stamp)[0]): continue
                tf = tf_buffer.lookup_transform_core(map_frame_id, msg.header.frame_id,  msg.header.stamp)
                pose = msg_to_pose(tf)
                pc_array = rospcmsg_to_pcarray(msg, pose)
                success, cam_id = getImageId(topic, POINT_CLOUD_NAMES)
                if not success:
                    print('Unknown point cloud topic: ' + topic)
                    continue
                pointcloud_data.append(msg.header.stamp.to_sec(), pc_array,  cam_id)

        # Update poses & commands
        for i, stamp in enumerate(command_data.indices['base']):
            stamp_ros = rospy.Time.from_sec(stamp)
            command_base = command_data.data['base'][i]

            for key in TF_POSE_REF_LIST:
                if (tf_buffer.can_transform_core(key, TF_BASE, stamp_ros)[0]):
                    tf = tf_buffer.lookup_transform_core(key, TF_BASE, stamp_ros)
                    pose = msg_to_pose(tf)  # pose in fixed ref frame (odom or map)
                    pose_data.append(stamp, pose, key)

                    R_T = np.zeros([2, 3, 3])
                    R_T[:] = msg_to_rotmat(tf)

                    command_transformed = command_base.copy()
                    command_transformed[:3] = R_T[0] @ command_base[:3]
                    command_data.append(stamp, command_transformed, key)

                    # tf = tf_buffer.lookup_transform_core(key, 'cam4_sensor_frame', stamp_ros)
                    # cameras['cam4'].update_pose_from_base_pose(pose)
                    # print(cameras['cam4'].translation)
                    # print(cameras['cam4'].rotation)
                    # # print(tf)
                    # r = msg_to_rotmat(tf)
                    # print(r)
                    # exit()

        # Transform velocities
        for i, stamp in enumerate(velocity_data.indices['base']):
            stamp_ros = rospy.Time.from_sec(stamp)
            velocity_base = velocity_data.data['base'][i]

            for key in TF_POSE_REF_LIST:
                if tf_buffer.can_transform_core(key, TF_BASE, stamp_ros)[0]:
                    tf = tf_buffer.lookup_transform_core(key, TF_BASE, stamp_ros)
                    R_T = np.zeros([2, 3, 3])
                    R_T[:] = msg_to_rotmat(tf)
                    vel_transformed = np.matmul(R_T, np.reshape(velocity_base, [2, 3, 1]))
                    vel_transformed = vel_transformed.reshape([6])
                    velocity_data.append(stamp, vel_transformed, key)

        # PROCESS AND SAVE
        print("Start processing")
        cmd_dfs = []
        for key in command_data.data_id.keys():
            df = pd.DataFrame(data=command_data.data[key], index=command_data.indices[key])
            df.index = pd.to_datetime(df.index, unit='s')
            cmd_dfs.append(df)
        cmd_df = pd.concat(cmd_dfs, axis=1, keys=command_data.data_id.keys())
        cmd_df = cmd_df.resample(resample_freq).apply(lambda x: x.mean() if x.isnull().sum() == 0 else np.nan)

        resampled_idx = cmd_df.index

        # Hack to sync indices
        dummy = pd.DataFrame(data=np.zeros(len(resampled_idx)), index=resampled_idx)
        dummy_resampled = dummy.resample(proprioception_resample_freq).nearest()
        if cfg['save_proprioception']:
            proprio_df = resample_ffill(proprio_data, dummy_resampled.index, proprio_dt.to_sec())
            proprio_df = np.array(proprio_df)

        first_index = dummy_resampled.index.tolist().index(dummy.index[skip_steps])
        # A = dummy.index[skip_steps + 25]
        # B = dummy_resampled.index[first_index + 25 * proprioception_decimation]  # Should be A = B

        # reindex_and_interpolate
        if cfg['save_proprioception']:
            proprio_stacked = []
            proprio_index_start = first_index - proprioception_history_half_length
            proprio_index_end = first_index + proprioception_history_half_length

            for datum_idx in range(len(resampled_idx)):

                if proprio_index_end >= proprio_df.shape[0] or datum_idx < skip_steps:
                    datum = np.full(proprio_df.shape[1], np.nan)
                else:
                    datum = proprio_df[proprio_index_start: proprio_index_end].flatten()
                    proprio_index_start += proprioception_decimation
                    proprio_index_end += proprioception_decimation
                proprio_stacked.append(datum)


            proprio_stacked_df = pd.DataFrame(data=proprio_stacked, index=resampled_idx)
            state_df = resample_ffill(state_data, resampled_idx, dt.to_sec(), additional=("proprioception", proprio_stacked_df))
        else:
            state_df = resample_ffill(state_data, resampled_idx, dt.to_sec())

        img_df = resample_dataid_ffill(image_data, resampled_idx, 1.0)  # tol > 0.7 second (darpa: img updates ~ 1.5 Hz)
        map_df = resample_dataid_ffill(map_data, resampled_idx, 0.3)  # tol > 0.2 second (map updates ~ 5 Hz)
        pointcloud_df = resample_dataid_ffill(pointcloud_data, resampled_idx, 0.15)  # tol > 0.1 second (pointcloud updates ~ 9.9 Hz)
        pose_df = resample_ffill(pose_data, resampled_idx, dt.to_sec())
        vel_df = resample_ffill(velocity_data, resampled_idx, dt.to_sec())

        transition_vel_dfs = []
        for key in velocity_data.data_id.keys():
            df = pd.DataFrame(data=velocity_data.data[key], index=velocity_data.indices[key])
            df.index = pd.to_datetime(df.index, unit='s')
            df = df[~df.index.duplicated(keep='first')]

            df.loc[resampled_idx[0]] = np.nan
            df = df.resample(resample_freq).mean()
            df = df.reindex(resampled_idx, method='nearest')
            transition_vel_dfs.append(df)

        transition_vel_df = pd.concat(transition_vel_dfs, axis=1, keys=velocity_data.data_id.keys())

        total_df = pd.concat([cmd_df, img_df, map_df, pose_df, vel_df, state_df, transition_vel_df, pointcloud_df],
                             axis=1,
                             keys=['cmd', 'img', 'map', 'pose', 'velocity', 'state', 'transition_velocity', 'pc'])

        # total_df.to_csv("/media/jolee/Samsung_T5/Monkey_Trip_SA/chimera_mission_2021_10_07/mission6/text_df.csv")
        total_df.dropna(axis=0, inplace=True, how='any')

        previous_index = total_df.index[0]
        discontinuity = [0]
        list_of_dfs = []

        for i, idx in enumerate(total_df.index):
            diff = (idx - previous_index).total_seconds()
            if diff > dt.to_sec():
                discontinuity.append(i)
            previous_index = idx

        if len(discontinuity) > 0:
            for i in range(len(discontinuity)):
                if i == len(discontinuity) - 1:
                    df = total_df.iloc[discontinuity[i]:]
                else:
                    df = total_df.iloc[discontinuity[i]:discontinuity[i + 1]]
                list_of_dfs.append(df)

        # Process each traj segment

        # read keys
        cmd_keys = list_of_dfs[0]['cmd'].columns.remove_unused_levels().levels[0]
        pose_keys = list_of_dfs[0]['pose'].columns.remove_unused_levels().levels[0]

        velocity_keys = list_of_dfs[0]['velocity'].columns.remove_unused_levels().levels[0]
        img_keys = list_of_dfs[0]['img'].columns.remove_unused_levels().levels[0]
        map_key = list_of_dfs[0]['map'].columns.remove_unused_levels().levels[0][0]
        pc_keys = list_of_dfs[0]['pc'].columns.remove_unused_levels().levels[0]
        print("map_key:", map_key)
        print("pc_keys :",pc_keys)
        print("img_keys :",img_keys)
                
        transition_vel_keys = list_of_dfs[0]['transition_velocity'].columns.remove_unused_levels().levels[0]

        datum_idx = 0
        for df_id, df in enumerate(list_of_dfs):
            print('segment:', df_id + 1, "/", len(list_of_dfs), ", len:", len(df.index))
            trajectory_info = dict()
            trajectory_info['foot_positions'] = []
            trajectory_info['body_contacts'] = []

            # Get local maps & collision detection using RaiSim
            for time_idx, map_idx in enumerate(np.array(df['map'])):
                map_idx = int(map_idx)
                elev_map = map_data.data[map_key][map_idx]
                pose_ref = elev_map.frame_id
                pose = np.array(df['pose'][pose_ref].iloc[time_idx])

                if raisim_objects['terrain'] is not None:
                    raisim_objects['world'].removeObject(raisim_objects['terrain'])
                raisim_objects['terrain'] = elev_map.getRaisimMap(raisim_objects['world'])

                q = ros_pos_to_raisim_gc(pose, df['state']['joint_position'].iloc[time_idx])

                raisim_objects['anymal'].setGeneralizedCoordinate(q)
                raisim_objects['world'].integrate1()
                num_non_foot_contacts = 0

                foot_positions = []

                # get foot positions
                # TODO: filter out feet in contact using foot contact state
                for foot_idx in raisim_objects['foot_indices']:
                    foot_positions.append(raisim_objects['anymal'].getFramePosition(foot_idx))

                # body collision points
                body_collision_points = []
                for contact in raisim_objects['anymal'].getContacts():
                    if contact.getlocalBodyIndex() not in [3, 6, 9, 12]:
                        num_non_foot_contacts += 1
                        body_collision_points.append(contact.get_position())

                trajectory_info['foot_positions'].append(foot_positions)
                if len(body_collision_points):
                    trajectory_info['body_contacts'].append(body_collision_points)

            trajectory_info['foot_positions'] = np.array(trajectory_info['foot_positions'])

            if len(trajectory_info['body_contacts']):
                trajectory_info['body_contacts'] = np.array(trajectory_info['body_contacts'])

            
            ### Project the pointclouds to get its position in images
            pointcloudinfo = dict()
            pointcloudinfo["pc"] = []
            pointcloudinfo["field"] = [("xzy",3), ("i",1),("rgb",3)] + [(c,3) for c in CAM_NAMES]
            for time_idx, map_idx in enumerate(np.array(df['map'])):
                cloudpoints = []
                map_idx = int(map_idx)
                elev_map = map_data.data[map_key][map_idx]
                pose = np.array(df['pose'][elev_map.frame_id].iloc[time_idx])
                for pck  in pc_keys:
                    pc_idx = int(df['pc'][pck].iloc[time_idx][0])
                    cloud = pointcloud_data.data[pck][pc_idx]
                    # cache some variables
                    imgs = []
                    for cam_id in CAM_NAMES:
                        cameras[cam_id].update_pose_from_base_pose(pose)

                        image_idx = int(df['img'][cam_id].iloc[time_idx][0])
                        image = image_data.data[cam_id][int(image_idx)]
                        assert image.shape[1] == cameras[cam_id].image_width  
                        assert image.shape[0] == cameras[cam_id].image_height  
                        imgs.append(image)
                    for p in cloud:
                        p = np.array(p)
                        img_pos = []
                        rgb = np.zeros(3)
                        for cam_id,image in zip(CAM_NAMES, imgs):
                            proj_point, proj_jac = cameras[cam_id].project_point(p[:3])
                            proj_point = np.reshape(proj_point, [-1])
                            camera_heading = cameras[cam_id].pose[1][:3, 2]
                            point_dir = p[:3] - cameras[cam_id].pose[0]
                            visible = np.sum(camera_heading * point_dir) > 1.0
                            img_pos.append(np.hstack([visible, proj_point]))
                            if(visible 
                                    and 0.0 <= proj_point[0] < cameras[cam_id].image_width 
                                    and 0.0 <= proj_point[1] < cameras[cam_id].image_height
                                ):
                                proj_point_ind = proj_point.astype(int)
                                rgb = image[proj_point_ind[1], proj_point_ind[0], :]
                        p = np.hstack([p, rgb]+img_pos)
                        cloudpoints.append(p)
                pointcloudinfo["pc"].append(cloudpoints)
                print("pc prograss: %d/%d, pc size %d"%(time_idx,len(df['map']),len(cloudpoints)))
            ### Extract Image patches & save
            print("Extract image patches & save per patch")
            for time_idx, map_idx in enumerate(np.array(df['map'])):

                save_dict = dict()
                save_dict['commands'] = {}
                save_dict['pose'] = {}
                save_dict['velocity'] = {}

                save_dict['images'] = {}
                save_dict['map'] = None
                save_dict['image_patches'] = []
                save_dict['image_patches_loc'] = []

                save_dict['pointcloud'] = pointcloudinfo['pc'][time_idx]

                # Save robot state
                for key in cmd_keys:
                    save_dict['commands'][key] = np.array(df['cmd'][key].iloc[time_idx], dtype=np.float32)

                for key in pose_keys:
                    save_dict['pose'][key] = np.array(df['pose'][key].iloc[time_idx], dtype=np.float32)

                for key in velocity_keys:
                    save_dict['velocity'][key] = np.array(df['velocity'][key].iloc[time_idx], dtype=np.float32)

                # Save images
                for key in img_keys:
                    image_idx = int(df['img'][key].iloc[time_idx][0])
                    image = np.moveaxis(image_data.data[key][image_idx], 2, 0)
                    save_dict['images'][key] = image.astype(np.float32)

                # Save local map
                map_idx = int(map_idx)
                elev_map = map_data.data[map_key][map_idx]
                pose = np.array(df['pose'][elev_map.frame_id].iloc[time_idx])
                yaw = euler_from_quaternion(pose[3:])[2]
                local_map = elev_map.getLocalMap(pose[:3], yaw, local_map_shape)
                save_dict['map'] = local_map

                # Collision detection using Raisim + Perspective projection
                if raisim_objects['terrain'] is not None:
                    raisim_objects['world'].removeObject(raisim_objects['terrain'])
                raisim_objects['terrain'] = elev_map.getRaisimMap(raisim_objects['world'])

                q = ros_pos_to_raisim_gc(pose, df['state']['joint_position'].iloc[time_idx])
                raisim_objects['anymal'].setGeneralizedCoordinate(q)
                raisim_objects['world'].integrate1()

                foot_positions_distance = trajectory_info['foot_positions'].copy()
                foot_positions_distance = np.linalg.norm(foot_positions_distance - pose[:3], axis=-1)

                idx = foot_positions_distance < cfg['foot_projection_range']
                foot_positions_in_fov = trajectory_info['foot_positions'][idx]

                objs = []
                projected_points = {}
                for cam_id in CAM_NAMES:
                    projected_points[cam_id] = []
                    cameras[cam_id].update_pose_from_base_pose(pose)
                    cyl = raisim_objects['world'].addCylinder(0.05, 0.1, 0.1, collision_group = 0)
                    cyl.setPose(cameras[cam_id].pose[0], cameras[cam_id].pose[1][:3, :3])
                    objs.append(cyl)

                    for j in range(foot_positions_in_fov.shape[0]):
                        camera_heading = cameras[cam_id].pose[1][:3, 2]
                        visible = True

                        ray_N = 1
                        angs = np.random.uniform(-1.0, 1.0, [ray_N, 3])

                        mean_dist = 0.0
                        dists = []

                        test_value = None
                        for ray_id in range(ray_N):
                            point_dir = foot_positions_in_fov[j] + 0.03 * angs[ray_id] - cameras[cam_id].pose[0]

                            # When the camera is not facing the object
                            test_value = np.sum(camera_heading * point_dir)
                            if np.sum(camera_heading * point_dir) < 1.0:
                                visible = False
                                break
                            else:
                                dist = np.linalg.norm(point_dir)
                                ray_col_list = raisim_objects['world'].rayTest(cameras[cam_id].pose[0], point_dir, dist,
                                                                               True)
                                if ray_col_list.size() > 0:
                                    col_pos = ray_col_list.at(0).getPosition()
                                    # box = raisim_objects['world'].addBox(0.05, 0.05, 0.05, 0.1, collision_group=0)
                                    # box.setPosition(col_pos)
                                    # box.setAppearance("blue")
                                    # objs.append(box)

                                    dist = np.linalg.norm(col_pos - foot_positions_in_fov[j])
                                    dists.append(dist)
                                    mean_dist += dist

                        mean_dist /= ray_N

                        if mean_dist > cfg['ray_col_margin']:
                            visible = False

                        if visible:
                            # project point
                            proj_point, proj_jac = cameras[cam_id].project_point(foot_positions_in_fov[j])
                            proj_point = np.reshape(proj_point, [-1])

                            if 0.0 <= proj_point[0] < cameras[cam_id].image_width \
                                    and 0.0 <= proj_point[1] < cameras[cam_id].image_height:
                                proj_point = proj_point.astype(int)
                                projected_points[cam_id].append((foot_positions_in_fov[j], proj_point, test_value))
                            else:
                                visible = False

                        if cfg['visualize']:
                            if not visible:
                                sphere = raisim_objects['world'].addSphere(0.03, 0.1, collision_group=0)
                                sphere.setAppearance("green")
                                sphere.setPosition(foot_positions_in_fov[j])
                                objs.append(sphere)
                            else:
                                sphere = raisim_objects['world'].addSphere(0.075, 0.1, collision_group=0)
                                sphere.setAppearance("red")
                                sphere.setPosition(foot_positions_in_fov[j])
                                objs.append(sphere)

                    image_idx = int(df['img'][cam_id].iloc[time_idx][0])
                    image = np.moveaxis(image_data.data[cam_id][int(image_idx)], 2, 0)

                    if cfg['visualize']:
                        image = np.moveaxis(image, 0, 2)

                    patch_id = 0
                    print("num_patches_[", cam_id, "]: ", len(projected_points[cam_id]))
                    for pos_world, pos_img, debug in projected_points[cam_id]:
                        if cfg['visualize']:
                            w = int(cfg['image_patch_shape'][0] / 2)
                            h = int(cfg['image_patch_shape'][1] / 2)
                            image = cv2.rectangle(img=image,
                                                  pt1=(pos_img[0] - w, pos_img[1] - h),
                                                  pt2=(pos_img[0] + w, pos_img[1] + h),
                                                  color=(255, 0, 0), thickness=3)

                        def project_edge(mid, width, lim):
                            start = mid - int(width / 2)
                            end = start + width
                            if start < 0:
                                print("start_out")
                                start = 0
                            if end >= lim - 1:
                                start = lim - width - 1
                                print("end_out")

                            end = start + width

                            return (start, end)

                        patch_x = project_edge(pos_img[1], cfg['image_patch_shape'][0], cameras[cam_id].image_height)
                        patch_y = project_edge(pos_img[0], cfg['image_patch_shape'][1], cameras[cam_id].image_width)

                        if cfg['visualize']:
                            patch = image[patch_x[0]:patch_x[1], patch_y[0]:patch_y[1]]
                            description = str(patch_x[0]) + ", " + str(patch_y[0])
                            # cv2.imshow(str(patch_id) + "_" + description + "_" + str(debug), patch)
                        else:
                            patch = image[:, patch_x[0]:patch_x[1], patch_y[0]:patch_y[1]]

                        save_dict['image_patches_loc'].append(np.array(pos_world, dtype=np.float32))  # TODO: include meta info
                        save_dict['image_patches'].append(np.array(patch, dtype=np.float32))  # TODO: include meta info
                        patch_id += 1

                    if cfg['visualize']:
                        cv2.imshow(cam_id, image)

                if cfg['visualize']:
                    map_local = cv2.resize(local_map[0], (500, 500), interpolation=cv2.INTER_NEAREST)
                    # cv2.imshow('map_local', (map_local - np.min(map_local)) / (np.max(map_local) - np.min(map_local)))
                    print("show images")
                    cv2.waitKey(0)
                    for i, obj in enumerate(objs):
                        raisim_objects['world'].removeObject(obj)

                if not (cfg['skip_no_patch'] and (len(save_dict['image_patches']) == 0)):
                    _name = "traj_" + str(traj_idx) + "_datum_" + str(datum_idx)
                    out_file = os.path.join(out_dir, _name)
                    out_file += '.msgpack'
                    print(out_file)
    
                    datum_idx += 1
    
                    with open(out_file, "wb") as outfile:
                        file_dat = msgpack.packb(save_dict)
                        outfile.write(file_dat)

    print('Complete.')


def main():
    # Load cfg
    cfg_path = os.path.dirname(os.path.realpath(__file__)) + '/../../cfg/data_extraction_SA.yaml'
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
    cameras = {}
    for cam_id in cfg['CAM_NAMES']:
        cameras[cam_id] = Camera(camera_calibration_path,
                                 cam_id,
                                 cfg)

    # SETUP ANYMAL SIMULATION FOR COLLISION DETECTION
    import raisimpy as raisim
    dir_path = os.path.dirname(os.path.realpath(__file__))
    urdf_path = dir_path + '/../../rsc/robot/chimera/urdf/anymal_minimal.urdf'  # TODO: load from sim2real repo
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
        extractAndSyncTrajs(bagfile, os.path.join(output_path, basename), cfg, cameras, raisim_objects)

    print("DONE!")


if __name__ == '__main__':
    main()
