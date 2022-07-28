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

import numpy as np

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Header
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import rosgraph
assert rosgraph.is_master_online()

import sys
sys.path.append("../Labelling/")
from semantic_front_end_filter.Labelling.messages.imageMessage import rgb_msg_to_image
from semantic_front_end_filter.Labelling.messages.pointcloudMessage import rospcmsg_to_pcarray
from semantic_front_end_filter.Labelling.messages.messageToVectors import msg_to_pose
import tf

# import matplotlib.pyplot as plt

# image_topic = "alphasense_driver_ros/cam4/dropped/debayered/compressed"
image_topic = "/alphasense_driver_ros/cam4/image_raw/compressed"
pointcloud_topic = "/bpearl_rear/point_cloud"
TF_BASE = "base"

rospy.init_node('ros_visulizer_replay_filter', anonymous=False)
listener = tf.TransformListener()
pose_pub = rospy.Publisher("/semantic_filter_pose", Float64MultiArray, queue_size=1)
image_pub = rospy.Publisher("/semantic_filter_image", Float64MultiArray, queue_size=1)
# depth_pub = rospy.Publisher("/semantic_filter_depth", Float64MultiArray, queue_size=1)
points_pub = rospy.Publisher("/semantic_filter_points", Float64MultiArray, queue_size=1)

def image_callback(data):
    print("image received data")
    # img = rgb_msg_to_image(data, ("debayered" in image_topic), False, ("compressed" in image_topic))
    img = rgb_msg_to_image(data, True, False, ("compressed" in image_topic))

    img = np.moveaxis(img, 2, 0)

    img_msg = Float64MultiArray(data = img.reshape(-1))
    img_msg.layout.dim=[MultiArrayDimension(size=d) for d in img.shape]
    image_pub.publish(img_msg)

points_buffer = []
def pointcloud_callback(data):
    print("pointcloud_callback")
    # tf = tf_buffer.lookup_transform_core("map", data.header.frame_id,  data.header.stamp)
    
    try:
        listener.waitForTransform("map", data.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
        (trans,rot) = listener.lookupTransform("map", data.header.frame_id, rospy.Time(0))
    except Exception as e:
        print(e)
        return
    ## The rot is in order xyzw
    pose = [*trans, *rot]
    pc_array = rospcmsg_to_pcarray(data, pose)[:,:3]
    global points_buffer
    points_buffer.append(pc_array)
    if(len(points_buffer)>5):
        points_buffer = points_buffer[-5:]
    
    pc_array = np.concatenate(points_buffer,axis=0)
    pc_msg = Float64MultiArray(data = pc_array.reshape(-1))
    pc_msg.layout.dim=[MultiArrayDimension(size=d) for d in pc_array.shape]
    points_pub.publish(pc_msg)



rospy.Subscriber(image_topic, CompressedImage, image_callback)
rospy.Subscriber(pointcloud_topic, PointCloud2, pointcloud_callback)
rate = rospy.Rate(10) # 1hz
while not rospy.is_shutdown():
    rate.sleep()
    listener.waitForTransform("map", TF_BASE, rospy.Time(0), rospy.Duration(1.0))
    (trans,rot) = listener.lookupTransform("map", TF_BASE, rospy.Time(0))
    pose = np.array([*trans, *rot]).astype(np.float64)
    pose_pub.publish(Float64MultiArray(data = pose))
