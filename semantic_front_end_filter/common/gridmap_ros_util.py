from grid_map_msgs.msg import GridMap, GridMapInfo
from std_msgs.msg import Header, Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from geometry_msgs.msg import Pose, Quaternion, Point
import numpy as np

def build_gridmap_msg(mapArrays, res, len_x, len_y, xyz, rot_quat = None, frame_id="map"):
    # gpmap = np.zeros((int(len_x/res), int(len_y/res)))
    rot_quat = (0,0,0,0) if rot_quat is None else rot_quat
    mapmsg = GridMap(
        info = GridMapInfo(
            header= Header(frame_id=frame_id),
            resolution = res,
            length_x = len_x,
            length_y = len_y,
            pose = Pose(
                position = Point(*xyz),
                orientation = Quaternion(*rot_quat)
            )),
        layers = [k for k,v in mapArrays.items()],
        basic_layers = [k for k,v in mapArrays.items()],
        data = [
            Float32MultiArray(
                layout = MultiArrayLayout(
                    dim = [
                        MultiArrayDimension(
                            label= l,
                            size = d,
                            stride = np.prod( maparr.shape[i:])
                        )for i,(d,l) in enumerate(zip(maparr.shape,
                                            ["column_index", "row_index"]))
                    ]
                ),
                data = maparr.astype(np.float32).reshape(-1).tolist()
            )
            for k,maparr in mapArrays.items()
        ]
    )
    return mapmsg