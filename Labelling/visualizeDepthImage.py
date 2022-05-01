from ExtractDepthImage import DIFG
import tf
import rospy
import matplotlib.pyplot as plt
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# d = DIFG('/home/anqiao/catkin_ws/SA_dataset/20211007_SA_Monkey_ANYmal_Chimera/chimera_mission_2021_10_10/mission1/Recontruct_2022-04-18-19-40-09_0/GroundMap.msgpack' )
d = DIFG('/home/anqiao/semantic_front_end_filter/Labelling/Example_Files/GroundMap.msgpack')
rospy.init_node('DI_tf_listener')
image_pub = rospy.Publisher("image_topic_2",Image)
bridge = CvBridge()

listener = tf.TransformListener()
rate = rospy.Rate(3.0)
# DImage = np.ones((640, 480))
# plt.imshow(DImage)
# plt.show()
# plt.draw()
while not rospy.is_shutdown():
    try:
        # d.close()
        (trans,rot) = listener.lookupTransform('/map', '/cam0_sensor_frame', rospy.Time(0))
        euler = tf.transformations.euler_from_quaternion(rot)
        # print(euler)
        DImage = d.getDImage(transition=trans, rotation=euler)
        if(sum(sum(DImage))!=0):
            DImage[DImage==0] = DImage[DImage!=0].min()- (DImage[DImage!=0].max()-DImage[DImage!=0].min())/10
            DImage = (DImage - DImage.min())/(DImage.max()-DImage.min())*255
        # imageCV = np.zeros((DImage.shape[0], DImage.shape[1], 3))
        # imageCV[:, :, 0] = DImage*64/255
        # imageCV[:, :, 1] = DImage*200/255
        # imageCV[:, :, 2] = DImage*64/255 

        # image_message = bridge.cv2_to_imgmsg(imageCV, encoding="passthrough")
        imageCV = cv2.applyColorMap(DImage.numpy().astype(np.uint8), cv2.COLORMAP_JET)
        print("Publishing")

        # image_pub.publish(bridge.cv2_to_imgmsg(DImage.numpy().astype(np.uint8), "mono8"))
        image_pub.publish(bridge.cv2_to_imgmsg(imageCV, "bgr8"))


    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        continue
    rate.sleep()
    # plt.cla()

