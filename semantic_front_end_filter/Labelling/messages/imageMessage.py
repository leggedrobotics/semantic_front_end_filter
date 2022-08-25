import cv2
import numpy as np
# from ruamel.yaml import YAML
import yaml
from scipy.spatial.transform import Rotation 
try:
    from .messageToVectors import msg_to_pose
    from cv_bridge import CvBridge, CvBridgeError
    from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_from_matrix, quaternion_from_matrix
except ModuleNotFoundError as ex:
    print("ImageMassage Warning: ros package fails to load")
    print(ex)
import os
class Camera:
    def __init__(self, calibration_cfg_path, cam_id, cfg):
        cali_path = None
        self.name = None

        for root, dirs, files in os.walk(calibration_cfg_path):
            for file in files:
                if cam_id in file:
                    cali_path = os.path.join(root, file)
                    self.name = cam_id

        assert cali_path is not None

        # processing options
        self.rb_swap = (cam_id in cfg['CAM_RBSWAP'])

        self.is_debayered = ('debayered' in cfg['CAM_SUFFIX'])

        # calibration data
        self.calibration = yaml.load(open(cali_path, 'r'), Loader=yaml.FullLoader)
        self.camera_matrix = np.array(self.calibration['camera_matrix']['data']).reshape([3, 3])
        self.distortion = np.array(self.calibration['distortion_coefficients']['data'])

        # ros stuff
        self.topic_id = cfg['CAM_PREFIX'] + self.name + cfg['CAM_SUFFIX']
        self.frame_key = self.calibration['camera_name']

        self.tf_base_to_sensor = None

        self.rvec = None
        self.tvec = None
        self.pose = None

        self.image_width = self.calibration['image_width']
        self.image_height = self.calibration['image_height']

    def update_static_tf(self, tf):
        rel_pose = msg_to_pose(tf)
        position = np.array(rel_pose[:3])
        quat = rel_pose[3:]
        R = np.array(quaternion_matrix(quat))

        self.tf_base_to_sensor = (position, R)

    def update_pose_from_base_pose(self, base_in_world):

        position = np.array(base_in_world[:3])
        # R_bak = np.array(quaternion_matrix(base_in_world[3:]))
        rotation = Rotation.from_quat(base_in_world[3:])
        R = np.eye(4)
        R[:3,:3] = rotation.as_matrix()
        # assert(np.sum((R_bak - R)**2) < 1e-10)


        self.pose = (np.matmul(R[:3, :3], self.tf_base_to_sensor[0]) + position, np.matmul(R, self.tf_base_to_sensor[1]))
        R = self.pose[1][:3, :3].transpose()
        self.tvec = - np.matmul(R, self.pose[0])
        self.rvec = cv2.Rodrigues(R)[0]

    def project_point(self, point_positions):
        # R = self.rotation[:3, :3].transpose()
        # T = - np.matmul(R, self.translation)
        result = cv2.projectPoints(point_positions,
                                   self.rvec,
                                   self.tvec,
                                   self.camera_matrix,
                                   self.distortion)
        return result



def getImageId(topic, candidates):
    for name in candidates:
        if (topic.find(name) != -1):
            return True, name
    return False, None


    # output = rgb image
def rgb_msg_to_image(image_msg, already_debayered=False, rb_swap=False, compressed=False):
    if compressed:
        res = CvBridge().compressed_imgmsg_to_cv2(image_msg)
    else:
        res = CvBridge().imgmsg_to_cv2(image_msg)

    if not already_debayered:
        # Debayer
        processed = cv2.cvtColor(res, cv2.COLOR_BayerGB2RGB)

        # Color correction
        # https://bitbucket.org/leggedrobotics/darpa_subt/src/8c31ec05f4bf3848bba8a9f3ae6bdd9376fa5ffc/artifacts_detection/debayer_cuda/src/debayer_cuda.cpp?at=master#lines-134
        CCM = np.array([[1.0, 0.19, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.57, 1.0]])

        if rb_swap:
            CCM[:, [2, 0]] = CCM[:, [0, 2]]
            # cv2.imshow('imagedb0',debayered) # Just for debugging

        processed = np.array(processed).astype(np.float32)
        processed = np.squeeze(np.matmul(CCM, np.expand_dims(processed, -1)))
        processed = np.clip(processed, 0.0, 255.0)
        processed /= 255.0

    else:
        processed = res
        # print(np.max(debayered))
        # cv2.imshow('image',res) # Just for debugging
        # cv2.imshow('imagedb', processed) # Just for debugging
        # cv2.waitKey(0)  # Just for debugging
        # img = np.asarray(res)

    # resize
    # orig_width = processed.shape[1]
    # width_ratio = max_width / orig_width
    # processed = cv2.resize(processed, (0, 0), fx=width_ratio, fy=width_ratio, interpolation=cv2.INTER_LINEAR)

    return processed

