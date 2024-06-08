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


import sensor_msgs.point_cloud2 as ros_pc_msg2
# from ruamel.yaml import YAML
import yaml
from .messageToVectors import msg_to_pose
from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_from_matrix, quaternion_from_matrix
import os
import numpy as np


def getImageId(topic, candidates):
    for name in candidates:
        if (topic.find(name) != -1):
            return True, name
    return False, None


def rospcmsg_to_pcarray(ros_cloud, cam_pose):
    """ 
    Reference: https://github.com/anshulpaigwar/Attentional-PointNet/blob/master/tools/pcl_helper.py 
    Converts a ROS PointCloud2 message to a pcl PointXYZRGB

        Args:
            ros_cloud (PointCloud2): ROS PointCloud2 message

        Returns:
            pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
    """
    position = np.array(cam_pose[:3])
    R = np.array(quaternion_matrix(cam_pose[3:]))

    field_names = [f.name for f in ros_cloud.fields]
    points_list = []

    for data in ros_pc_msg2.read_points(ros_cloud, skip_nans=True):
        data_p = data[:3]
        data_p = np.matmul(R[:3, :3], np.array(data_p)) + position
        points_list.append( tuple(data_p) + data[3:] )
    return np.array(points_list)


