"""
Chenyu 2022-05-16
The visulizer in RVIZ
A node with two modes, 
    online: Subscribe to topic "cam4", "point cloud"
    offline: Load a image from dataloader and 
        project the depth image and prediction into rviz as point clouds
"""


import os
from ruamel.yaml import YAML
from argparse import ArgumentParser
import struct

import cv2

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib

import msgpack
import msgpack_numpy as m
import math
from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_from_matrix, quaternion_from_matrix
m.patch()


import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import rosgraph
assert rosgraph.is_master_online()

import sys
sys.path.append("../Labelling/")
from messages.imageMessage import Camera
from train import *

class RosVisulizer:
    def __init__(self, topic):
        self.pub = rospy.Publisher(topic, PointCloud2, queue_size=1)
        # if(rospy.)
        rospy.init_node('ros_visulizer', anonymous=True)
        
    def buildPoint(self, x,y,z,r,g,b,a=None):
        if(np.array([r,g,b]).max()<1.01):
            r = int(r * 255.0)
            g = int(g * 255.0)
            b = int(b * 255.0)
            a = 255 if a is None else int(a * 255.0)
        else:
            r = int(r)
            g = int(g)
            b = int(b)
            a = 255 if a is None else int(a)
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        return [x, y, z, rgb]

    def publish_point_cloud(self, pc, img=None):
        """
        the Pc's field is x,y,z, i, r,g,b, [camflag, projx, projy]x3
        thanks to example at https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
        """
        from sensor_msgs import point_cloud2
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header

        header = Header()
        header.frame_id = "map"
        
        fields = [
          PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          # PointField('rgb', 12, PointField.UINT32, 1),
          PointField('rgba', 12, PointField.UINT32, 1),
        ]

        pc = np.array(pc)
#         pc[:, :3] -= pc[:, :3].mean(axis=0)
        # pc[:, 4:7] /=256.
        if img is None:
            colors = pc[:, 4:7] /256.
        else: # get colors by projecting points to img
            colors = np.zeros_like(pc[:, 4:7])
            imgshape = img.shape
            # print("imgshape :",imgshape)
            assert(len(imgshape)==3 and (imgshape[0] in [1,3]))
            tmp_fxy = pc[:, 7+ 1*3: 7 + (1+1)*3]
            flag, px, py = tmp_fxy[:,0], tmp_fxy[:,1], tmp_fxy[:,2]
            flag = flag>0.5 # turn 0-1 into boolean
            flag = flag & (0<=px) & (px< imgshape[2]) & (0<=py) & (py< imgshape[1])
            # print("flag.sum :",flag.sum())
            px[~flag] = 0
            py[~flag] = 0
            px = px.astype(np.int32)
            py = py.astype(np.int32)
            feats = img[:, py, px]
            feats = feats.T
            colors = np.where(flag[:,None], feats, colors)
            # print("colors.max :",colors.max())
        cloud = point_cloud2.create_cloud(header, fields, 
            [self.buildPoint(*p[:3], *c) for p,c in zip(pc, colors)])

        self.pub.publish(cloud)

terrainpub = rospy.Publisher("terrain_pc", PointCloud2, queue_size=1)
predpub = rospy.Publisher("pred_pc", PointCloud2, queue_size=1)
def vis_from_dataset(sample):
    with open(sample["path"][0], "rb") as data_file:
        byte_data = data_file.read()
        data = msgpack.unpackb(byte_data)
    img = data['images']["cam4"].copy()
    img = img.astype(np.uint8)
    rosv.publish_point_cloud(data['pointcloud'], img)
    print("POS:", data['pointcloud'][:,:3].mean(axis = 0))

    ## Broad cast depth image
    depthimg = sample["depth"]
    camera_calibration_path = "/ros/catkin_ws/src/mnt/darpa_subt/anymal/anymal_chimera_subt/anymal_chimera_subt/config/calibrations/alphasense/"
    cam_id = "cam4"
    cfg={}
    cfg["CAM_RBSWAP"]=['']
    cfg["CAM_SUFFIX"]= '/dropped/debayered/compressed'
    cfg["CAM_PREFIX"]= '/alphasense_driver_ros/'
    camera = Camera(camera_calibration_path, cam_id, cfg)
    camera.tf_base_to_sensor = (np.array([-0.40548693, -0.00076062,  0.23253198]), 
                    np.array([[-0.00603566,  0.00181943, -0.99998013,  0.        ],
                        [ 0.99997436,  0.00386421, -0.00602859,  0.        ],
                        [ 0.00385317, -0.99999088, -0.00184271,  0.        ],
                        [ 0.        ,  0.        ,  0.        ,  1.        ]]))
    camera.update_pose_from_base_pose(data["pose"]["map"])
    K = camera.camera_matrix
    W,H = camera.image_width,camera.image_height
    pixel_cor = np.mgrid[0:W,0:H]
    pixel_cor_hom = np.concatenate( [ pixel_cor, np.ones_like(pixel_cor[None,0,:,:])], axis=0 )

    ray_dir = (np.linalg.inv(K) @ (pixel_cor_hom.reshape(3,-1))).T
    ray_dir = ray_dir/ np.linalg.norm(ray_dir, axis=1)[:,None]


    def calculate_H_map_cam(transition, rotation):
        from scipy.spatial.transform import Rotation
        H_map_cam = np.eye(4)
        H_map_cam[:3,3] =  np.array( [transition])
        H_map_cam[:3,:3] = Rotation.from_euler('zyx', [[-math.pi-rotation[2], rotation[1], rotation[0]]]).as_matrix() # looking down
        #         H_map_cam[:3,:3] = Rotation.from_euler('zyx', [[-np.math.pi-rotation[2], rotation[1], rotation[0]]], degrees=False).as_matrix() # looking down

        H_map_cam[:3,:3] = Rotation.from_euler('yz', [0, 180], degrees=True).as_matrix() @ H_map_cam[:3,:3]
        return H_map_cam

    # position = np.array(data["pose"]["map"][:3])
    # euler = np.array(euler_from_quaternion(data["pose"]["map"][3:]))
    position = camera.pose[0]
    euler = euler_from_matrix(camera.pose[1][:3,:3])
        
    H_map_cam = calculate_H_map_cam(position, euler)
    R = torch.from_numpy( H_map_cam )[:3,:3]
    directions = torch.from_numpy(ray_dir )
    directions = (directions @ R)
    start_points = torch.from_numpy( H_map_cam[:3,3])
    depth = sample["depth"][0][0].T
    pts = start_points + depth.reshape(-1,1)*directions

    header = Header()
    header.frame_id = "map"

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        # PointField('rgb', 12, PointField.UINT32, 1),
        PointField('rgba', 12, PointField.UINT32, 1),
    ]

    pts_color = pts[None,:,2,None]
    pts_color = ((pts_color-pts_color.min())/(pts_color.max()-pts_color.min())*255).numpy().astype(np.uint8)
    # im_color = cv2.applyColorMap(pts_color, cv2.COLORMAP_OCEAN)[0]
    image = np.transpose(sample["image"][0,:3,...],(2,1,0)).reshape(-1,3)
    image = (image-image.min())/(image.max()-image.min())
    im_color = image
    # colors = (pts[:,2,None] - pts[:,2].min())/(pts[:,2].max() - pts[:,2].min()) * np.ones([1,3])
    cloud = point_cloud2.create_cloud(header, fields, 
                [rosv.buildPoint(*p[:3], *c,a=0.2) for p,c in zip(pts, im_color)])

    print("terrainpub", cloud.height, cloud.width)
    terrainpub.publish(cloud)

    ## Broadcast model prediction

    checkpoint_path = "checkpoints/share/2022-05-14-00-19-41/UnetAdaptiveBins_best.pt"
    model = models.UnetAdaptiveBins.build(n_bins=args.modelconfig.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                            norm=args.modelconfig.norm)
    model,_,_ = model_io.load_checkpoint(checkpoint_path ,model) 

    _, pred = model(sample["image"])
    pred = pred[0].detach().numpy()
    pred = nn.functional.interpolate(torch.tensor(pred)[None,...], torch.tensor(depth.T).shape[-2:], mode='bilinear', align_corners=True)
    # pred = pred.reshape(-1)

    pred_pts = start_points + pred[0][0].T.reshape(-1,1)*directions

    pts_color = ((pts_color-pts_color.min())/(pts_color.max()-pts_color.min())*255).astype(np.uint8)
    im_color = cv2.applyColorMap(pts_color, cv2.COLORMAP_OCEAN)[0]
    # colors = (pts[:,2,None] - pts[:,2].min())/(pts[:,2].max() - pts[:,2].min()) * np.ones([1,3])
    cloud = point_cloud2.create_cloud(header, fields, 
                [rosv.buildPoint(*p[:3], *c, a=0.5) for p,c in zip(pred_pts, im_color)])


    predpub.publish(cloud)

if __name__ == "__main__":

    rosv = RosVisulizer("pointcloud")
    
    parser.add_argument("--model", default="")
    args = parse_args()
    # checkpoint_path = args.models
    # model = models.UnetAdaptiveBins.build(n_bins=args.modelconfig.n_bins, min_val=args.min_depth, max_val=args.max_depth,
    #                                         norm=args.modelconfig.norm)
    # model,_,_ = model_io.load_checkpoint(checkpoint_path ,model) 

    ## Loader
    args.data_path = "/Data/extract_trajectories_002"
    ## Loader
    data_loader = DepthDataLoader(args, 'online_eval').data
    # sample = next(iter(data_loader))
    data_iter = iter(data_loader)

    sample = next(data_iter)


    vis_from_dataset(sample)
    

