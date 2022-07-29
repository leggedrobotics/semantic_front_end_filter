import rospy
from std_msgs.msg import Header
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image, CompressedImage
import msgpack
import msgpack_numpy as m
import numpy as np
from cv_bridge import CvBridge
import ros_numpy

import tf

m.patch()
rospy.init_node('dump_publisher', anonymous=True)

pose_br = tf.TransformBroadcaster()
image_pub = rospy.Publisher("/alphasense_driver_ros/cam4/image_raw/compressed", Image, queue_size=1)
points_pub = rospy.Publisher("/bpearl_rear/point_cloud", PointCloud2, queue_size=1)


with open("/home/integration/git/semantic_front_end_filter/Data/Reconstruct_2022-04-24-18-35-59_0/traj_0_datum_2.msgpack", "rb") as data_file:
    byte_data = data_file.read()
    data = msgpack.unpackb(byte_data)

rate = rospy.Rate(10) # 1hz
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
    imgMsg.header.stamp = stamp
    image_pub.publish(imgMsg)

    #pointclouds
    cloud_arr = data['pointcloud']
    da = np.zeros(
        cloud_arr.shape[0],
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("intensity", np.float32),
        ],
    )
    da["x"] = cloud_arr[:, 0]
    da["y"] = cloud_arr[:, 1]
    da["z"] = cloud_arr[:, 2]
    da["intensity"] = cloud_arr[:, 3]
    pc_msg = ros_numpy.msgify(PointCloud2, da)
    pc_msg.header.stamp = stamp
    pc_msg.header.frame_id = "base"

    points_pub.publish(pc_msg)


    # pointsMsg = PointCloud2()
    
    # if stamp is not None:
    #     pointsMsg.header.stamp = stamp
    # if frame_id is not None:
    #     pointsMsg.header.frame_id = frame_id
    # pointsMsg.height = cloud_arr.shape[0]
    # pointsMsg.width = cloud_arr.shape[1]
    # pointsMsg.is_bigendian = False # assumption
    # pointsMsg.point_step = cloud_arr.dtype.itemsize
    # pointsMsg.row_step = pointsMsg.point_step*cloud_arr.shape[1]
    # pointsMsg.data = cloud_arr.tostring()
    # pointsMsg.header.stamp = stamp
    # # Maybe not write!!!!!!
    # pointsMsg.header.frame_id = "base"
    # points_pub.publish(pointsMsg)
