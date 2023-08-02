import CSF
import numpy as np
from sensor_msgs.msg import PointCloud2
import ros_numpy

def getCSFPoints(raw_points, rigidness = 3, resolution = 0.1):

    csf = CSF.CSF()

    # prameter settings
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = resolution
    csf.params.rigidness = rigidness
    # more details about parameter: http://ramm.bnu.edu.cn/projects/CSF/download/

    csf.setPointCloud(raw_points)
    ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
    non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
    csf.do_filtering(ground, non_ground) # do actual filtering.

    return raw_points[np.array(ground)], raw_points[np.array(non_ground)]

def pc_to_msg(cloud_arr, time_stamp):
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
    pc_msg.header.stamp = time_stamp
    pc_msg.header.frame_id = "map"

    return pc_msg

