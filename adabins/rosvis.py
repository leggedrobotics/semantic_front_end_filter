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
        header.frame_id = "msf_body_imu_map"
        
        fields = [
          PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          # PointField('rgb', 12, PointField.UINT32, 1),
          PointField('rgba', 12, PointField.UINT32, 1),
        ]

        pc = np.array(pc)
        pc[:, :3] -= pc[:, :3].mean(axis=0)
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

if __name__ == "__main__":

    parser.add_argument("--model", default="")
    args = parse_args()
    checkpoint_path = args.models
    model = models.UnetAdaptiveBins.build(n_bins=args.modelconfig.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                            norm=args.modelconfig.norm)
    model,_,_ = model_io.load_checkpoint(checkpoint_path ,model) 

    ## Loader
    data_loader = DepthDataLoader(args, 'test').data if train_loader is None else train_loader
    sample = next(iter(data_loader))

    rosv = RosVisulizer("pointcloud")

    img = img_dict["cam4depth"].copy()
    img[img>10] = 0
    flags = (img!=0)
    img[flags] = img[flags]/ img.max()
    img = (img*255).astype(np.uint8)
    print("max,min :",img.max(),img.min())
    im_color = cv2.applyColorMap(img[0], cv2.COLORMAP_JET)
    im_color = np.moveaxis(im_color, 2, 0)
    print("im_color :",im_color.shape)
    
    rosv.publish_point_cloud(data['pointcloud'], im_color)

