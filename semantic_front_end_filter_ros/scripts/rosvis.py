#!/usr/bin/python3
"""
Chenyu 2022-05-16
The visulizer in RVIZ
A node with two modes, 
    online: Subscribe to topic "cam4", "point cloud"
    offline: Load a image from dataloader and 
        project the depth image and prediction into rviz as point clouds
"""


import os
import sys
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
import math
from scipy.spatial.transform import Rotation 
import torch
import torch.nn as nn
try:
    from cv_bridge import CvBridge
    from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_from_matrix, quaternion_from_matrix
    import rospy
    from sensor_msgs import point_cloud2
    from sensor_msgs.msg import PointCloud2, PointField
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header
    from std_msgs.msg import Float64MultiArray
    import rosgraph
    assert rosgraph.is_master_online()
except ModuleNotFoundError as ex:
    print("rosvis Warning: ros package fails to load")
    print(ex)


from threading import Lock


from semantic_front_end_filter.adabins.cfgUtils import parse_args
from semantic_front_end_filter.adabins.pointcloudUtils import RaycastCamera
# import semantic_front_end_filter.adabins.models as models
from semantic_front_end_filter.adabins import model_io, models
from semantic_front_end_filter import SEMANTIC_FRONT_END_FILTER_ROOT_PATH
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEBUG_MODE=True

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
    def __init__(self, topic, camera_calibration_path="/home/integration/git/semantic_front_end_filter/anymal_c_subt_semantic_front_end_filter/config/calibrations/alphasense"):        
        self.pub = rospy.Publisher(topic, PointCloud2, queue_size=1)
        # if(rospy.)
        self.camera_calibration_path = camera_calibration_path
        self.raycastCamera = RaycastCamera(self.camera_calibration_path, device)
        self.image_cv_bridge = CvBridge()
    

    def build_could_from_depth_image(self, pose, depth, image = None):
        """
        depth: a torch tensor depth image
        image: the color image, can be numpy or torch
        """
        
        pts = self.raycastCamera.project_depth_to_cloud(pose, depth)
        height_mask = pts[:,2] < pose[2]
        pts = pts[height_mask]
        if(image is not None):
            im_color = image.reshape(-1,3)
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
            subsample_mask = torch.cuda.FloatTensor(pts.shape[0]).uniform_() < 0.3
        else:
            subsample_mask = np.random.choice([True, False], size=pts.shape[0], p=[0.3, 0.7])
        pts = pts[subsample_mask].cpu()
        print("cloud lengths:", len(pts))
        if(image is not None):
            im_color = im_color[subsample_mask]
            cloud = point_cloud2.create_cloud(header, fields, 
                        [rosv.buildPoint(*p[:3], *c,a=0.2) for p,c in zip(pts, im_color)])
        else:
            cloud = point_cloud2.create_cloud(header, fields, pts)
        return cloud

    def build_imgmsg_from_depth_image(self, depth, vmin, vmax):
        depth = torch.clamp(depth,min = vmin,max = vmax)
        depth = (depth-vmin)/(vmax-vmin) * 255
        depth = depth.T
        if(isinstance(depth, np.ndarray)):
            depth = depth.astype(np.uint8)
        elif(isinstance(depth, torch.Tensor)):
            depth = depth.cpu().numpy().astype(np.uint8)
        dimage = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        return self.image_cv_bridge.cv2_to_imgmsg(dimage, "bgr8")

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

rospy.init_node('ros_visulizer', anonymous=True)
terrainpub = rospy.Publisher("terrain_pc", PointCloud2, queue_size=1)
predpub = rospy.Publisher("pred_pc", PointCloud2, queue_size=1)
pred_image_pub = rospy.Publisher("pred_depth_image", Image, queue_size=1)
pc_image_pub = rospy.Publisher("pointcloud_image", Image, queue_size=1)
def vis_from_dataset(sample):
    with open(sample["path"][0], "rb") as data_file:
        byte_data = data_file.read()
        data = msgpack.unpackb(byte_data)
    img = data['images']["cam4"].copy()
    img = img.astype(np.uint8)
    rosv.publish_point_cloud(data['pointcloud'], img)
    print("POS:", data['pointcloud'][:,:3].mean(axis = 0))

    ## Broad cast depth image
    depth = sample["depth"][0][0].T

    image = np.transpose(sample["image"][0,:3,...],(2,1,0))
    image = (image-image.min())/(image.max()-image.min())
    cloud = rosv.build_could_from_depth_image(data["pose"]["map"], depth, image)
    print("terrainpub", cloud.height, cloud.width)
    terrainpub.publish(cloud)

    ## Broadcast model prediction

    checkpoint_path = "checkpoints/share/2022-05-14-00-19-41/UnetAdaptiveBins_best.pt"
    modelcfg = model_io.load_param_from_path(checkpoint_path)
    model = models.UnetAdaptiveBins.build(**modelcfg)
    model,_,_ = model_io.load_checkpoint(checkpoint_path ,model) 

    _, pred = model(sample["image"])
    pred = pred[0].detach().numpy()
    pred = nn.functional.interpolate(torch.tensor(pred).detach()[None,...], torch.tensor(depth.T).shape[-2:], mode='bilinear', align_corners=True)
    pred = pred[0][0].T
    pred_color = ((pred-pred.min())/(pred.max()-pred.min())*255).numpy().astype(np.uint8)
    im_color = cv2.applyColorMap(pred_color, cv2.COLORMAP_OCEAN)

    cloud = rosv.build_could_from_depth_image(data["pose"]["map"], pred, im_color)
    predpub.publish(cloud)

if __name__ == "__main__":

    MODE = "MODE_NODE"
    rosv = RosVisulizer("pointcloud")
    parser = ArgumentParser()
    parser.add_argument("--model", default="")
    args = parse_args(parser )
    if(MODE=="MODE_DATASET"):
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

    elif(MODE=="MODE_NODE"):
        
        checkpoint_path = os.path.join(
            SEMANTIC_FRONT_END_FILTER_ROOT_PATH,
            "adabins", 
            "/home/anqiao/tmp/semantic_front_end_filter/adabins/checkpoints/2022-07-26-00-11-50/UnetAdaptiveBins_latest.pt")
        modelcfg = model_io.load_param_from_path(checkpoint_path)
        model = models.UnetAdaptiveBins.build(**modelcfg)
        # model,_,_ = model_io.load_checkpoint(checkpoint_path ,model) 
        model.to(device)

        # the place holders for the values
        pose_ph = None
        depth_ph = None
        image_ph = None
        points_ph = None

        pose_mutex = Lock()
        depth_mutex = Lock()
        image_mutex = Lock()
        points_mutex = Lock()

        show_pred_flag = True
        show_depth_flag = True

        def pose_callback(data):
            global pose_ph
            pose_mutex.acquire()
            try:
                pose_ph = torch.tensor(data.data)
            finally:
                pose_mutex.release()

        def depth_callback(data):
            global depth_ph
            depth_mutex.acquire()
            try:
                shape = [d.size for d in data.layout.dim]
                depth_ph = torch.tensor(data.data).reshape(shape)
            finally:
                depth_mutex.release()
    
        def image_callback(data):
            global image_ph
            image_mutex.acquire()
            try:
                shape = [d.size for d in data.layout.dim]
                image_ph = torch.tensor(data.data).reshape(shape)
            finally:
                image_mutex.release()

        def points_callback(data):
            global points_ph
            points_mutex.acquire()
            try:
                shape = [d.size for d in data.layout.dim]
                points_ph = np.array(data.data).reshape(shape)
            finally:
                points_mutex.release()

        rospy.Subscriber("semantic_filter_pose", Float64MultiArray, pose_callback)
        rospy.Subscriber("semantic_filter_depth", Float64MultiArray, depth_callback)
        rospy.Subscriber("semantic_filter_image", Float64MultiArray, image_callback)
        rospy.Subscriber("semantic_filter_points", Float64MultiArray, points_callback)

        rate = rospy.Rate(1000) # 1000hz
        while not rospy.is_shutdown():
            rate.sleep()
            with Timer("full"):
                # start = time.time()
                # image_mutex.acquire()
                # image = image_ph.clone() if image_ph is not None else image_ph
                # image_mutex.release()
                # depth_mutex.acquire()
                # depth = depth_ph.clone() if depth_ph is not None else depth_ph
                # depth_mutex.release()
                # pose_mutex.acquire()
                # pose = pose_ph.clone() if pose_ph is not None else pose_ph
                # pose_mutex.release()
                # points_mutex.acquire()
                # points = points_ph.copy() if points_ph is not None else points_ph
                # points_mutex.release()
                start = time.time()
                image = image_ph.clone() if image_ph is not None else image_ph
                depth = depth_ph.clone() if depth_ph is not None else depth_ph
                pose = pose_ph.clone() if pose_ph is not None else pose_ph
                points = points_ph.copy() if points_ph is not None else points_ph
                update_end = time.time()
                if(DEBUG_MODE):
                    print("update cost: ", update_end-start)
                
                if(show_depth_flag and depth is not None and image is not None and pose is not None):
                    _depth = depth.T
                    _image = np.transpose(image.numpy(),(2,1,0))
                    _image = (_image-_image.min())/(_image.max()-_image.min())
                    cloud = rosv.build_could_from_depth_image(pose, _depth, _image)
                    print("terrainpub", cloud.height, cloud.width)
                    terrainpub.publish(cloud)
                projection_end = time.time()
                if(DEBUG_MODE):
                    print("projection cost: ", projection_end-update_end)
                
                if(show_pred_flag and image is not None and pose is not None):
                    pc_img = torch.zeros_like(image[:1,...]).numpy()
                    if(points is not None):
                        try:
                            time_count = -time.time()
                            pc_img = rosv.raycastCamera.project_cloud_to_depth(pose, points, pc_img)
                            ## TODO: normalize the input
                        except Exception as e:
                            print("PC IMAGE ERROR", e)

                    pc_img = torch.tensor(pc_img).to(device)
                    _image = image.to(device)
                    _image = torch.cat([_image/255., pc_img], axis = 0) # add the pc channel
                    _image = _image[None,...]
                    #normalize
                    for i,(m,s) in enumerate(zip([0.387, 0.394, 0.404, 0.120], [0.322, 0.32, 0.30,  1.17])):
                        _image[0,i,...] = (_image[0,i,...] - m)/s

                    with Timer("forward"):
                        pred = model(_image)
                    with Timer("post forward"):
                        pred = pred[0].detach()
                        pred = nn.functional.interpolate(torch.tensor(pred).detach()[None,...], _image.shape[-2:], mode='bilinear', align_corners=True)
                        pred = pred[0][0].T
                        # pred_color = ((pred-pred.min())/(pred.max()-pred.min())*255).numpy().astype(np.uint8)
                        # im_color = cv2.applyColorMap(pred_color, cv2.COLORMAP_OCEAN)
                        cloud = rosv.build_could_from_depth_image(pose, pred, None)
                        predction_end = time.time()
                        if(DEBUG_MODE):                
                            print("prediction & projection cost: ", predction_end-projection_end)    
                            print("one loop cost: ", predction_end-start)      
                            print("-------------------------------------------------------")      
                        
                        predpub.publish(cloud)
                        pc_image_pub.publish(rosv.build_imgmsg_from_depth_image(pc_img[0].T, vmin=5, vmax=30))
                        pred_image_pub.publish(rosv.build_imgmsg_from_depth_image(pred, vmin=5, vmax=30))

"""
This ros node mode can work with a publisher as the following

```python
import rospy
from std_msgs.msg import Header
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import msgpack
import msgpack_numpy as m
import numpy as np
m.patch()
rospy.init_node('dump_publisher', anonymous=True)
pose_pub = rospy.Publisher("/semantic_filter_pose", Float64MultiArray, queue_size=1)
image_pub = rospy.Publisher("/semantic_filter_image", Float64MultiArray, queue_size=1)
depth_pub = rospy.Publisher("/semantic_filter_depth", Float64MultiArray, queue_size=1)
points_pub = rospy.Publisher("/semantic_filter_points", Float64MultiArray, queue_size=1)
with open("extract_trajectories/Reconstruct_2022-04-25-15-31-34_0/traj_0_datum_2.msgpack", "rb") as data_file:
    byte_data = data_file.read()
    data = msgpack.unpackb(byte_data)

pose_pub.publish(Float64MultiArray(data = data["pose"]['map']))

img = data["images"]["cam4"].astype(np.float64)
img_msg = Float64MultiArray(data = img.reshape(-1))
img_msg.layout.dim=[MultiArrayDimension(size=d) for d in img.shape]
image_pub.publish(img_msg)

img = data["images"]["cam4depth"].astype(np.float64)[:1,...]
img_msg = Float64MultiArray(data = img.reshape(-1))
img_msg.layout.dim=[MultiArrayDimension(size=d) for d in img.shape]
depth_pub.publish(img_msg)

points = data["pointcloud"][:,:3].astype(np.float64)
pc_msg = Float64MultiArray(data = points.reshape(-1))
pc_msg.layout.dim=[MultiArrayDimension(size=d) for d in points.shape]
points_pub.publish(pc_msg)
```
"""