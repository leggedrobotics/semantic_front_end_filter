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


"""
Chenyu 2022-06-07
Utilities for pointcloud and depth image conversion
This file defines a class raycastCamera
"""

import sys
import os

from cv2 import projectPoints
from semantic_front_end_filter import SEMANTIC_FRONT_END_FILTER_ROOT_PATH
LabellingPath = os.path.join(SEMANTIC_FRONT_END_FILTER_ROOT_PATH,"Labelling")
from semantic_front_end_filter.utils.messages.imageMessage import Camera

import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_from_matrix, quaternion_from_matrix
from scipy.spatial.transform import Rotation
import math

class Timer:
    def __init__(self, name="") -> None:
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return "Hello, World!"
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        print(f"Time {self.name}", self.start.elapsed_time(self.end))

def calculate_H_map_cam(transition, rotation):
    
    H_map_cam = np.eye(4)
    H_map_cam[:3,3] =  np.array( [transition])
    H_map_cam[:3,:3] = Rotation.from_euler('zyx', [[-math.pi-rotation[2], rotation[1], rotation[0]]]).as_matrix() # looking down
    #         H_map_cam[:3,:3] = Rotation.from_euler('zyx', [[-np.math.pi-rotation[2], rotation[1], rotation[0]]], degrees=False).as_matrix() # looking down

    H_map_cam[:3,:3] = Rotation.from_euler('yz', [0, 180], degrees=True).as_matrix() @ H_map_cam[:3,:3]
    return H_map_cam


class RaycastCamera:
    def __init__(self, camera_calibration_path = None, device = None):
        self.camera_calibration_path = os.path.join(LabellingPath, "configs") if camera_calibration_path is None else camera_calibration_path
        cam_id = "cam4"
        cfg={}
        cfg["CAM_RBSWAP"]=['']
        cfg["CAM_SUFFIX"]= '/dropped/debayered/compressed'
        cfg["CAM_PREFIX"]= '/alphasense_driver_ros/'
        
        self.device = device if device is not None \
            else ( torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        camera = Camera(self.camera_calibration_path, cam_id, cfg)
        self.camera = camera
        camera.tf_base_to_sensor = (np.array([-0.40548693, -0.00076062,  0.23253198]), 
                        np.array([[-0.00603566,  0.00181943, -0.99998013,  0.        ],
                                [ 0.99997436,  0.00386421, -0.00602859,  0.        ],
                                [ 0.00385317, -0.99999088, -0.00184271,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  1.        ]]))
        W,H = camera.image_width,camera.image_height
        pixel_cor = np.mgrid[0:W,0:H]
        pixel_cor_hom = np.concatenate( [ pixel_cor, np.ones_like(pixel_cor[None,0,:,:])], axis=0 )
        pixel_cor_hom = pixel_cor_hom
        self.pixel_cor_hom = pixel_cor_hom

        K = camera.camera_matrix
        ray_dir = (np.linalg.inv(K) @ (pixel_cor_hom.reshape(3,-1))).T
        ray_dir = ray_dir/ np.linalg.norm(ray_dir, axis=1)[:,None]
        self.ray_dir = torch.tensor(ray_dir).to(self.device)

    def project_depth_to_cloud(self, pose, depth):
        """
        depth: a torch tensor depth image
        image: the color image, can be numpy or torch
        """
        self.camera.update_pose_from_base_pose(pose.cpu())
        W,H = self.camera.image_width,self.camera.image_height

        position = self.camera.pose[0]
        # euler_bak = euler_from_matrix(self.camera.pose[1][:3,:3]) # the default order is "sxyz", see http://docs.ros.org/en/jade/api/tf/html/python/transformations.html
        euler = Rotation.from_matrix(self.camera.pose[1][:3,:3]).as_euler("xyz")
        # assert( np.sum((euler - euler_bak)**2)<1e-9)
            
        H_map_cam = calculate_H_map_cam(position, euler)
        R = torch.from_numpy( H_map_cam )[:3,:3]
        directions = self.ray_dir
        directions = (directions @ R.to(self.device))
        start_points = torch.from_numpy( H_map_cam[:3,3]).to(self.device)
        pts = start_points + depth.reshape(-1,1)*directions
        return pts

    def project_cloud_to_depth(self, pose, points, pc_img, return_detail = False):
        self.camera.update_pose_from_base_pose(pose.cpu())
        camera_matrix = torch.Tensor(self.camera.camera_matrix).to(device)
        p = torch.Tensor(self.camera.tvec).to(device)
        H = torch.inverse(torch.Tensor(self.camera.pose[1]).to(device))
        R = H[:3, :3]
        # p = H[:3, 3]
        p = torch.Tensor(self.camera.tvec).to(device)
        # p = torch.matmul(R, p)
        h = torch.Tensor(self.camera.pose[0]).to(device)
        # TODO check this
        proj_point = torch.matmul(torch.matmul((points ), R.transpose(0, 1)) + p, camera_matrix.transpose(0, 1))
        proj_point = (proj_point/proj_point[:, 2:].repeat(1, 3)).long()
        # proj_point = proj_point.long()
        # proj_point, proj_jac = self.camera.project_point(points[:,:3].astype(np.float32))#150ms
        # proj_point = np.reshape(proj_point, [-1, 2]).astype(np.int32)#80ms
        camera_heading = torch.Tensor(self.camera.pose[1][:3, 2]).to(device)
        point_dir = points[:, :3] - h
        visible = torch.matmul(point_dir, camera_heading) > 1.0
        visible = (visible & (0.0 <= proj_point[:,0])
                & (proj_point[:,0] < self.camera.image_width)
                & (0.0 <= proj_point[:, 1])
                & (proj_point[:, 1] < self.camera.image_height))
        proj_point = proj_point[visible]
        pc_distance = torch.sqrt(torch.sum((points[visible,:3] - h)**2, axis = 1))
        pc_img[0, proj_point[:,1], proj_point[:,0]] = pc_distance
        if return_detail:
            return pc_img, visible, proj_point
        return pc_img
