import rospy
from std_msgs.msg import Header
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image, CompressedImage
import msgpack
import msgpack_numpy as m
import numpy as np
from cv_bridge import CvBridge

import tf

m.patch()
rospy.init_node('dump_publisher', anonymous=True)

pose_br = tf.TransformBroadcaster()
image_pub = rospy.Publisher("/alphasense_driver_ros/cam4/image_raw/compressed", Image, queue_size=1)
points_pub = rospy.Publisher("/bpearl_rear/point_cloud", PointCloud2, queue_size=1)
points_pub = rospy.Publisher("/bpearl_rear/point_cloud", PointCloud2, queue_size=1)

with open("/home/integration/git/semantic_front_end_filter/Data/Reconstruct_2022-04-24-18-35-59_0/traj_0_datum_2.msgpack", "rb") as data_file:
    byte_data = data_file.read()
    data = msgpack.unpackb(byte_data)

rate = rospy.Rate(1) # 1hz
while not rospy.is_shutdown():
    rate.sleep()
    stamp = rospy.Time.now()
    frame_id = None
    # tf
    pose_br.sendTransform((data["pose"]['map'][0:3]),
                        data["pose"]['map'][3:],
                        rospy.Time.now(),
                        "base",
                        "map")
    # image
    bridge = CvBridge()
    imgMsg = bridge.cv2_to_imgmsg(np.moveaxis(data["images"]["cam4"], 0, 2).astype(np.uint8), "bgr8")  
    image_pub.publish(imgMsg)
    #pointclouds
    pointsMsg = PointCloud2()
    cloud_arr = data['pointcloud']
    if stamp is not None:
        pointsMsg.header.stamp = stamp
    if frame_id is not None:
        pointsMsg.header.frame_id = frame_id
    pointsMsg.height = cloud_arr.shape[0]
    pointsMsg.width = cloud_arr.shape[1]
    pointsMsg.is_bigendian = False # assumption
    pointsMsg.point_step = cloud_arr.dtype.itemsize
    pointsMsg.row_step = pointsMsg.point_step*cloud_arr.shape[1]
    pointsMsg.data = cloud_arr.tostring()
    points_pub.publish(pointsMsg)
