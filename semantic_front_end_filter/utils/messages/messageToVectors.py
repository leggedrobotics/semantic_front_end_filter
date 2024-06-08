# This file is part of MyProject.
#
# MyProject is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MyProject is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MyProject.  If not, see <https://www.gnu.org/licenses/>.


from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_from_matrix, quaternion_from_matrix
import numpy as np

def msg_to_joint_mechanical_power(state_msg):
    pw = np.array(state_msg.joints.effort) * np.array(state_msg.joints.velocity)
    pw = np.clip(pw, 0.0, None)
    pw = np.sum(pw)
    return pw


def msg_to_joint_positions(state_msg):
    return state_msg.joints.position


def msg_to_joint_velocities(state_msg):
    return state_msg.joints.velocity

def msg_to_joint_torques(state_msg):
    return state_msg.joints.effort

def msg_to_body_lin_vel(state_msg):
    return np.array([state_msg.twist.twist.linear.x, state_msg.twist.twist.linear.y, state_msg.twist.twist.linear.z])


def msg_to_body_ang_vel(state_msg):
    return np.array([state_msg.twist.twist.angular.x, state_msg.twist.twist.angular.y, state_msg.twist.twist.angular.z])


def msg_to_grav_vec(state_msg):
    quat = (state_msg.pose.pose.orientation.x, state_msg.pose.pose.orientation.y, state_msg.pose.pose.orientation.z,
            state_msg.pose.pose.orientation.w)
    mat = quaternion_matrix(quat)
    return mat[2, :3]


def msg_to_command(command_msg):
    # # lin_dir = np.array([command_msg.twist.linear.x, command_msg.twist.linear.y])
    # # norm = np.linalg.norm(lin_dir)
    # # if norm > 0.01:
    # #     lin_dir = lin_dir / norm
    pose = [command_msg.twist.linear.x, command_msg.twist.linear.y, command_msg.twist.linear.z,
            command_msg.twist.angular.z]
    # pose = [command_msg.twist.linear.x, command_msg.twist.linear.y, command_msg.twist.angular.z]
    return pose


# def msg_to_tf(tf_msg):
#     quat = (
#         tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z,
#         tf_msg.transform.rotation.w)
#     euler = euler_from_quaternion(quat)
#     pose = [tf_msg.transform.translation.x, tf_msg.transform.translation.y, euler[2]]
#     return pose


def msg_to_pose(tf_msg):
    quat = (
        tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z,
        tf_msg.transform.rotation.w)
    euler = euler_from_quaternion(quat)
    pose = [tf_msg.transform.translation.x,
            tf_msg.transform.translation.y,
            tf_msg.transform.translation.z,
            quat[0],
            quat[1],
            quat[2],
            quat[3]]
    # mat = quaternion_matrix(quat)
    return pose


def msg_to_rotmat(tf_msg):
    quat = (
        tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z,
        tf_msg.transform.rotation.w)
    mat = quaternion_matrix(quat)
    return mat[:3, :3]