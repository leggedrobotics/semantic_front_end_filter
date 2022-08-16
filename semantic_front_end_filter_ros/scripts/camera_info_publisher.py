#!/usr/bin/python3
import rospy
from sensor_msgs.msg import Image, CameraInfo
from ruamel.yaml import YAML
import os

CAM_INFO_TOPIC = '/alphasense_driver_ros/cam4/camera_info'
CAM_IMAGE_TOPIC = '/alphasense_driver_ros/cam4/debayered'

pub = rospy.Publisher(CAM_INFO_TOPIC, CameraInfo, queue_size=10)
rospy.init_node('cam_info_pub')
cali_path = os.path.join(os.path.dirname(__file__), "../../", "semantic_front_end_filter", "Labelling", "configs", "cam4.yaml")
calibration = YAML().load(open(cali_path, 'r'))
print(calibration)
r = rospy.Rate(10) # 0.1hz


cam_info_msg = CameraInfo(
    height=calibration["image_height"],
    width=calibration["image_width"],
    distortion_model=calibration["distortion_model"],
    D = calibration["distortion_coefficients"]["data"],
    K = calibration["camera_matrix"]["data"],
    R = calibration["rectification_matrix"]["data"],
    P = calibration["projection_matrix"]["data"]
    )

def callback(data):
    cam_msg = cam_info_msg
    cam_msg.header = data.header
    pub.publish(cam_msg)
    

rospy.Subscriber(CAM_IMAGE_TOPIC, Image, callback)

if __name__ == "__main__":
    rospy.spin()
