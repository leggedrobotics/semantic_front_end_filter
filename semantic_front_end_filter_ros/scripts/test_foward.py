#!/usr/bin/python3
# filters the topics in replay stack for the rosvis node
# topics:
# - front camera
#   - subscribe: /alphasense_driver_ros/cam4/dropped/debayered/compressed
#       - (in forest dataset might be /alphasense_driver_ros/cam4/dropped/debayered/slow)
#   - broadcast: /semantic_filter_image
# - point cloud
#   - subscribe: /bpearl_front/point_cloud
#   - broadcast: /semantic_filter_points
# - pose (pose in map frame)
#   - subscribe: /tf
#   - broadcast: /semantic_filter_pose
from pip import main
from scipy.spatial.transform import Rotation
import message_filters
import tf
from semantic_front_end_filter.Labelling.messages.messageToVectors import msg_to_pose
from semantic_front_end_filter.Labelling.messages.pointcloudMessage import rospcmsg_to_pcarray
from semantic_front_end_filter.Labelling.messages.imageMessage import rgb_msg_to_image
import rosgraph
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
import rospy
from email import message
from semantic_front_end_filter import SEMANTIC_FRONT_END_FILTER_ROOT_PATH
from semantic_front_end_filter.adabins import model_io, models
from semantic_front_end_filter.adabins.pointcloudUtils import RaycastCamera
from semantic_front_end_filter.adabins.cfgUtils import parse_args
from threading import Lock
import torch.nn as nn
import torch
import math
import os
import sys

# from matplotlib import image
# from ruamel.yaml import YAML
import yaml
from simple_parsing import ArgumentParser
import struct

import cv2
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import msgpack
import msgpack_numpy as m
m.patch()
try:
    from cv_bridge import CvBridge
    from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_from_matrix, quaternion_from_matrix
    import rospy
    import ros_numpy
    from sensor_msgs import point_cloud2
    from sensor_msgs.msg import PointCloud2, PointField
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header
    from std_msgs.msg import Float64MultiArray
    import rosgraph
    if __name__ == '__main__':
        assert rosgraph.is_master_online()
except ModuleNotFoundError as ex:
    print("rosvis Warning: ros package fails to load")
    print(ex)


# import semantic_front_end_filter.adabins.models as models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    assert rosgraph.is_master_online()

sys.path.append("../Labelling/")


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


class RosVisulizer:
    def __init__(self, topic, camera_calibration_path="/home/anqiao/tmp/semantic_front_end_filter/anymal_c_subt_semantic_front_end_filter/config/calibrations/alphasense"):
        self.pub = rospy.Publisher(topic, PointCloud2, queue_size=1)
        # if(rospy.)
        self.camera_calibration_path = camera_calibration_path
        self.raycastCamera = RaycastCamera(
            self.camera_calibration_path, device)
        self.image_cv_bridge = CvBridge()

    def build_could_from_depth_image(self, pose, depth, image=None):
        """
        depth: a torch tensor depth image
        image: the color image, can be numpy or torch
        """

        pts = self.raycastCamera.project_depth_to_cloud(pose, depth)
        height_mask = pts[:, 2] < pose[2]
        pts = pts[height_mask]
        if(image is not None):
            im_color = image.reshape(-1, 3)
            im_color = im_color[height_mask]
        header = Header()
        header.frame_id = "map"
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        if(image is not None):
            fields.append(PointField('rgba', 12, PointField.UINT32, 1))
        # subsample
        if(pts.is_cuda):
            subsample_mask = torch.cuda.FloatTensor(
                pts.shape[0]).uniform_() < 0.3
        else:
            subsample_mask = np.random.choice(
                [True, False], size=pts.shape[0], p=[0.3, 0.7])
        pts = pts[subsample_mask].cpu()
        print("cloud lengths:", len(pts))
        if(image is not None):
            im_color = im_color[subsample_mask]
            cloud = point_cloud2.create_cloud(header, fields,
                                              [rosv.buildPoint(*p[:3], *c, a=0.2) for p, c in zip(pts, im_color)])
        else:
            cloud = point_cloud2.create_cloud(header, fields, pts)
        return cloud

    def build_imgmsg_from_depth_image(self, depth, vmin, vmax):
        depth = torch.clamp(depth, min=vmin, max=vmax)
        depth = (depth-vmin)/(vmax-vmin) * 255
        depth = depth.T
        if(isinstance(depth, np.ndarray)):
            depth = depth.astype(np.uint8)
        elif(isinstance(depth, torch.Tensor)):
            depth = depth.cpu().numpy().astype(np.uint8)
        dimage = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        return self.image_cv_bridge.cv2_to_imgmsg(dimage, "bgr8")

    def buildPoint(self, x, y, z, r, g, b, a=None):
        if(np.array([r, g, b]).max() < 1.01):
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
            colors = pc[:, 4:7] / 256.
        else:  # get colors by projecting points to img
            colors = np.zeros_like(pc[:, 4:7])
            imgshape = img.shape
            # print("imgshape :",imgshape)
            assert(len(imgshape) == 3 and (imgshape[0] in [1, 3]))
            tmp_fxy = pc[:, 7 + 1*3: 7 + (1+1)*3]
            flag, px, py = tmp_fxy[:, 0], tmp_fxy[:, 1], tmp_fxy[:, 2]
            flag = flag > 0.5  # turn 0-1 into boolean
            flag = flag & (0 <= px) & (px < imgshape[2]) & (
                0 <= py) & (py < imgshape[1])
            # print("flag.sum :",flag.sum())
            px[~flag] = 0
            py[~flag] = 0
            px = px.astype(np.int32)
            py = py.astype(np.int32)
            feats = img[:, py, px]
            feats = feats.T
            colors = np.where(flag[:, None], feats, colors)
            # print("colors.max :",colors.max())
        cloud = point_cloud2.create_cloud(header, fields,
                                          [self.buildPoint(*p[:3], *c) for p, c in zip(pc, colors)])

        self.pub.publish(cloud)


def callback(image, point_cloud):
    # print("call_back")
    # image
    with Timer("Update"):
        img = rgb_msg_to_image(image, True, False, False)
        img = np.moveaxis(img, 2, 0)
        img = torch.tensor(img)
        # pointclouds
        try:
            listener.waitForTransform(
                "map", point_cloud.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = listener.lookupTransform(
                "map", point_cloud.header.frame_id, rospy.Time(0))
        except Exception as e:
            print(e)
            return
        pose = [*trans, *rot]
        # pc_array = rospcmsg_to_pcarray(point_cloud, pose)[:,:3]
        surf = ros_numpy.numpify(point_cloud)
        pc_array = np.copy(np.frombuffer(surf.tobytes(), np.dtype(
            np.float32)).reshape(surf.shape[0], -1)[:, :3])
        global points_buffer
        points_buffer.append(pc_array)
        if(len(points_buffer) > 10):
            points_buffer = points_buffer[-10:]
        pc_array = np.concatenate(points_buffer, axis=0)

        global image_ph
        global points_ph
        points_ph = pc_array
        image_ph = img

        # update
        listener.waitForTransform(
            "map", TF_BASE, rospy.Time(0), rospy.Duration(1.0))
        (trans, rot) = listener.lookupTransform("map", TF_BASE, rospy.Time(0))
        image = image_ph.clone() if image_ph is not None else image_ph
        points = points_ph.copy() if points_ph is not None else points_ph
        # TODO change from the start
        points = torch.Tensor(points).to(device)
        pose = torch.Tensor(
            np.array([*trans, *rot]).astype(np.float64)).to(device)

    if(image is not None and pose is not None):
        with Timer("pre foward"):
            # Projection to get pc image
            pc_img = torch.zeros_like(image[:1, ...]).to(device).float()
            if(points is not None):
                pc_img = rosv.raycastCamera.project_cloud_to_depth(
                    pose, points, pc_img)
                # fig, axs = plt.subplots(1, 2,figsize=(20, 20))

                # axs[0].imshow(pc_img[0].cpu().numpy())
                # axs[1].imshow(image.moveaxis(0, 2).numpy())
            # pc_img = torch.tensor(pc_img).to(device)
            _image = image.to(device)
            _image = torch.cat([_image/255., pc_img],
                               axis=0)  # add the pc channel
            _image = _image[None, ...]
            # normalize
            for i, (m, s) in enumerate(zip([0.387, 0.394, 0.404, 0.120], [0.322, 0.32, 0.30,  1.17])):
                _image[0, i, ...] = (_image[0, i, ...] - m)/s

        with Timer("forward"):
            # global model
            pred = model(_image)
        with Timer("post forward"):
            pred = pred[0].detach()
            pred = nn.functional.interpolate(torch.tensor(pred).detach(
            )[None, ...], _image.shape[-2:], mode='bilinear', align_corners=True)
            pred = pred[0][0].T
            # pred_color = ((pred-pred.min())/(pred.max()-pred.min())*255).numpy().astype(np.uint8)
            # im_color = cv2.applyColorMap(pred_color, cv2.COLORMAP_OCEAN)
            # cloud = rosv.build_could_from_depth_image(pose, pred, None)
            predction_end = time.time()
        print("---------------------------------------------------------------------")


pose_ph = None
depth_ph = None
image_ph = None
points_ph = None
points_buffer = []
# image_topic = "alphasense_driver_ros/cam4/dropped/debayered/compressed"
image_topic = "/alphasense_driver_ros/cam4/image_raw/compressed"
pointcloud_topic = "/bpearl_rear/point_cloud"
TF_BASE = "base"

rospy.init_node('test_foward', anonymous=False)

listener = tf.TransformListener()
image_sub = message_filters.Subscriber(image_topic, Image)
pc_sub = message_filters.Subscriber(pointcloud_topic, PointCloud2)
ts = message_filters.ApproximateTimeSynchronizer([image_sub, pc_sub], 1, 1)
# message_filters.a

rosv = RosVisulizer("pointcloud")
parser = ArgumentParser()
parser.add_argument("--model", default="")
args = parse_args(parser)
model = models.UnetAdaptiveBins.build(n_bins=args.modelconfig.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                      input_channel=4,
                                      norm=args.modelconfig.norm, use_adabins=False)
model.to(device)

ts.registerCallback(callback)

rate = rospy.Rate(30)
while not rospy.is_shutdown():
    rate.sleep()
# rate = rospy.Rate(10) # 1hz
# while not rospy.is_shutdown():
#     rate.sleep()
#     with Timer("Update"):
#         # update
#         listener.waitForTransform("map", TF_BASE, rospy.Time(0), rospy.Duration(1.0))
#         (trans,rot) = listener.lookupTransform("map", TF_BASE, rospy.Time(0))
#         image = image_ph.clone() if image_ph is not None else image_ph
#         points = points_ph.copy() if points_ph is not None else points_ph
#         pose = torch.Tensor(np.array([*trans, *rot]).astype(np.float64))


#     if(image is not None and pose is not None):
#         with Timer("pre foward"):
#             # Projection to get pc image
#             pc_img = torch.zeros_like(image[:1,...]).numpy()
#             if(points is not None):
#                 pc_img = rosv.raycastCamera.project_cloud_to_depth(pose, points, pc_img)
#             pc_img = torch.tensor(pc_img).to(device)
#             _image = image.to(device)
#             _image = torch.cat([_image/255., pc_img], axis = 0) # add the pc channel
#             _image = _image[None,...]
#             #normalize
#             for i,(m,s) in enumerate(zip([0.387, 0.394, 0.404, 0.120], [0.322, 0.32, 0.30,  1.17])):
#                 _image[0,i,...] = (_image[0,i,...] - m)/s

#         with Timer("forward"):
#             pred = model(_image)
#         with Timer("post forward"):
#             pred = pred[0].detach()
#             pred = nn.functional.interpolate(torch.tensor(pred).detach()[None,...], _image.shape[-2:], mode='bilinear', align_corners=True)
#             pred = pred[0][0].T
#             # pred_color = ((pred-pred.min())/(pred.max()-pred.min())*255).numpy().astype(np.uint8)
#             # im_color = cv2.applyColorMap(pred_color, cv2.COLORMAP_OCEAN)
#             cloud = rosv.build_could_from_depth_image(pose, pred, None)
#             predction_end = time.time()
#         print("---------------------------------------------------------------------")

    # predpub.publish(cloud)
    # pc_image_pub.publish(rosv.build_imgmsg_from_depth_image(pc_img[0].T, vmin=5, vmax=30))
    # pred_image_pub.publish(rosv.build_imgmsg_from_depth_image(pred, vmin=5, vmax=30))
