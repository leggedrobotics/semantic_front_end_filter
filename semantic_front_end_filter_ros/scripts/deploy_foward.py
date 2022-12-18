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
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

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
import matplotlib.pyplot as plt
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
FLAG_MARKER = False


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
    def __init__(self, topic, camera_calibration_path):
        self.pub = rospy.Publisher(topic, PointCloud2, queue_size=1)
        # if(rospy.)
        self.camera_calibration_path = camera_calibration_path
        self.raycastCamera = RaycastCamera(
            self.camera_calibration_path, device)
        self.image_cv_bridge = CvBridge()
        print("waiting for images")

    def build_could_from_depth_image(self, pose, depth, image=None, pc_image=None, pose_bp=None):
        """
        depth: a torch tensor depth image
        image: the color image, can be numpy or torch
        """

        pts = self.raycastCamera.project_depth_to_cloud(pose, depth)
        height_mask = pts[:, 2] < pose[2]
        # pts = pts[height_mask]
        if(image is not None):
            im_color = image.reshape(-1, 3)
            im_color = im_color[height_mask]
        header = Header()
        header.frame_id = "bpearl_rear"
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
        # pts = pts[subsample_mask].cpu()
        pts = pts.cpu()
        pts = pts[~torch.isnan(pts).any(axis=1), :]
        pose_bp = pose_bp.to(dtype=torch.float64)
        Rot = torch.Tensor(quaternion_matrix(pose_bp[3:].cpu().numpy())[
                           :3, :3]).to(device, dtype=torch.float64)
        pts = torch.matmul(pts-pose_bp[:3].cpu(), Rot.cpu())

        print("cloud lengths:", len(pts))
        if(image is not None):
            im_color = im_color[subsample_mask]
            cloud = point_cloud2.create_cloud(header, fields,
                                              [rosv.buildPoint(*p[:3], *c, a=0.2) for p, c in zip(pts, im_color)])
        else:
            cloud = point_cloud2.create_cloud(header, fields, pts)

        marker = None
        if(pc_image is not None):
            print("drawing lines")
            pts_pc = self.raycastCamera.project_depth_to_cloud(pose, pc_image)
            pts_pc = pts_pc.cpu()
            pts_pc = pts_pc[~torch.isnan(pts_pc).any(axis=1), :]
            # pts = torch.matmul(pts, Rot.cpu().T)+pose_bp[:3].cpu()
            pts_pc = torch.matmul(pts_pc-pose_bp[:3].cpu(), Rot.cpu())

            distance_mask = ((depth - pc_image)).reshape(-1, 1)
            distance_mask = distance_mask[~torch.isnan(
                distance_mask).any(axis=1), :]
            distance_mask = distance_mask > 0

            if FLAG_MARKER:
                marker = Marker()
                marker.header.frame_id = "bpearl_rear"
                marker.type = marker.LINE_LIST
                marker.action = marker.ADD
                marker.scale.x = 0.02
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0
                marker.color.b = 0
                for i, (pt, pt_pc) in enumerate(zip(pts, pts_pc)):
                    # if i%10 == 0:
                    if ~distance_mask[i]:
                        marker.points.append(Point(x=pt[0], y=pt[1], z=pt[2]))
                        marker.points.append(
                            Point(x=pt_pc[0], y=pt_pc[1], z=pt_pc[2]))

        return cloud, marker

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


def callback(img_msg, point_cloud):
    print("callback")
    # image
    image = rgb_msg_to_image(img_msg, True, False, False)
    image = np.moveaxis(image, 2, 0)
    image = torch.tensor(image).to(device)

    # pose
    try:
        listener.waitForTransform(
            "map", point_cloud.header.frame_id, img_msg.header.stamp, rospy.Duration(1.0))
        (trans, rot) = listener.lookupTransform(
            "map", point_cloud.header.frame_id, img_msg.header.stamp)
        listener.waitForTransform(
            "map", "base", img_msg.header.stamp, rospy.Duration(1.0))
        (trans_base, rot_base) = listener.lookupTransform(
            "map", "base", img_msg.header.stamp)
    except Exception as e:
        print(e)
        return
    pose = torch.Tensor(np.array([*trans, *rot]).astype(np.float64)).to(device)
    pose_base = torch.Tensor(
        np.array([*trans_base, *rot_base]).astype(np.float64)).to(device)

    # pointclouds
    surf = ros_numpy.numpify(point_cloud)
    if surf.ndim == 1:
        pc_array = np.copy(np.frombuffer(surf.tobytes(), np.dtype(
            np.float32)).reshape(surf.shape[0], -1)[:, :3])
    else:
        pc_array = np.copy(np.frombuffer(surf.tobytes(), np.dtype(
            np.float32)).reshape(surf.shape[0]*surf.shape[1], -1)[:, :3])

    points = torch.Tensor(pc_array).to(device)
    Rot = torch.Tensor(quaternion_matrix(rot)[:3, :3]).to(device)
    points = torch.matmul(points, Rot.T) + pose[:3]

    if(image is not None and pose is not None):
        # Get Input
        pc_img = torch.zeros_like(image[:1, ...]).to(device).float()
        if(points is not None):
            pc_img = rosv.raycastCamera.project_cloud_to_depth(
                pose_base, points, pc_img)
            # fig, axs = plt.subplots(1, 2,figsize=(20, 20))
            # axs[0].imshow(pc_img[0].cpu().numpy())
            # axs[1].imshow(image.moveaxis(0, 2).numpy())
        _image = torch.cat([image/255., pc_img],
                           axis=0)  # add the pc channel
        _image = _image[None, ...]
        for i, (m, s) in enumerate(zip([0.387, 0.394, 0.404, 0.120], [0.322, 0.32, 0.30,  1.17])):
            _image[0, i, ...] = (_image[0, i, ...] - m)/s
        # Prediction
        pred = model(_image)
        if pred.shape[1] == 2:
            mask_weight = torch.nn.functional.sigmoid(pred[:, 1:, :, :])
            pred = mask_weight * pred[:, :1, :, :] + \
                (1-mask_weight)*_image[0, 3:, ...]
        if pred.shape[1] == 3:
            mask_weight = (pred[:, 1:2] > pred[:, 0:1])
            pred_origin = pred[:, 2:]
            pred = pred[:, 2:].clone()
            pred[~mask_weight] = _image[:, 3:, ...][~mask_weight]
        pred = pred[0].detach()
        pred = nn.functional.interpolate(torch.tensor(
            pred)[None, ...], torch.tensor(pc_img).shape[-2:])[0]
        m = torch.logical_or((pc_img < 1e-9), (pc_img > 10))
        # m = torch.logical_or(m, (pred - pc_img)<0)
        pred[m] = torch.nan
        pc_img[m] = torch.nan
        pred = pred[0].T
        # pred_color = ((pred-pred.min())/(pred.max()-pred.min())*255).numpy().astype(np.uint8)
        # im_color = cv2.applyColorMap(pred_color, cv2.COLORMAP_OCEAN)
        cloud, marker = rosv.build_could_from_depth_image(
            pose_base, pred, None, pc_img.squeeze().T, pose)

        if marker is not None:
            marker.header.stamp = point_cloud.header.stamp
            lines_pub.publish(marker)
        cloud.header.stamp = point_cloud.header.stamp
        predpub.publish(cloud)
        # pc_image_pub.publish(rosv.build_imgmsg_from_depth_image(pc_img[0].T, vmin=5, vmax=30))
        pred_image_pub.publish(
            rosv.build_imgmsg_from_depth_image(pred, vmin=5, vmax=30))
        predction_end = time.time()
        print("---------------------------------------------------------------------")


if __name__ == "__main__":
    pose_ph = None
    depth_ph = None
    image_ph = None
    points_ph = None
    points_buffer = []
    # image_topic = "alphasense_driver_ros/cam4/dropped/debayered/compressed"
    image_topic = "/alphasense_driver_ros/cam4/debayered"
    # image_topic = "/alphasense_driver_ros/cam4/image_raw/compressed"
    pointcloud_topic = "/bpearl_rear/point_cloud"
    pts_lines_topic = "/bpearl_rear/raw_predtion_lines"
    prediction_topic = "/prediction/forward"
    camera_calibration_path = "/home/anqiao/tmp/semantic_front_end_filter/anymal_c_subt_semantic_front_end_filter/config/calibrations/alphasense"
    TF_BASE = "base"

    rospy.init_node('test_foward', anonymous=False)

    # Sub and Pub
    listener = tf.TransformListener()
    image_sub = message_filters.Subscriber(image_topic, Image)
    pc_sub = message_filters.Subscriber(pointcloud_topic, PointCloud2)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, pc_sub], 1, 1)

    pred_image_pub = rospy.Publisher("pred_depth_image", Image, queue_size=1)
    pc_image_pub = rospy.Publisher("pointcloud_image", Image, queue_size=1)
    predpub = rospy.Publisher(prediction_topic, PointCloud2, queue_size=1)
    lines_pub = rospy.Publisher(pts_lines_topic, Marker, queue_size=1)

    # Build model
    rosv = RosVisulizer("pointcloud", camera_calibration_path)
    # model_path = "/media/anqiao/Semantic/Models/2022-08-29-23-51-44_fixed/UnetAdaptiveBins_latest.pt"
    # model_path = "/home/anqiao/tmp/semantic_front_end_filter/adabins/checkpoints/2022-11-04-02-05-45_edge5/UnetAdaptiveBins_best.pt"
    model_path = "/home/anqiao/tmp/semantic_front_end_filter/adabins/checkpoints/2022-11-19-11-44-00_reg0.0002+GrassForest/UnetAdaptiveBins_best.pt"
    model_path = "/home/anqiao/tmp/semantic_front_end_filter/checkpoints/2022-12-07-17-56-21/UnetAdaptiveBins_best.pt"
    model_cfg = yaml.load(open(os.path.join(os.path.dirname(
        model_path), "ModelConfig.yaml"), 'r'), Loader=yaml.FullLoader)
    model_cfg["input_channel"] = 4
    model = models.UnetAdaptiveBins.build(**model_cfg)
    model = model_io.load_checkpoint(model_path, model)[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ts.registerCallback(callback)

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        rate.sleep()
