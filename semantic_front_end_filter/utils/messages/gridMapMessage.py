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


import numpy as np


class GridMapFromMessage:
    def __init__(self, msg, map_idx=1):  # msg = grid_map
        self.frame_id = msg.info.header.frame_id
        self.resolution = msg.info.resolution
        self.size = np.array([round(msg.info.length_x / self.resolution), (msg.info.length_y / self.resolution)],
                             dtype=int)

        self.position = np.array([msg.info.pose.position.x, msg.info.pose.position.y])
        self.length = self.size * self.resolution
        self.center_offset = 0.5 * self.length

        # self.map = np.asarray(msg.data[0].data, order='F')  # Caution!!: col-major

        self.map_matrix = np.asarray(msg.data[map_idx].data)  # 0: traversability
        # self.map_matrix = np.asarray(msg.data[1].data)
        self.map_matrix.resize(self.size)
        self.map_matrix = self.map_matrix.transpose()  # Caution!!: col-major

        self.nan_map = np.isnan(self.map_matrix)
        self.map_matrix[self.nan_map] = 0.0

    def getIndexFromPosition(self, position):
        index_vector = (position - self.center_offset - self.position) / self.resolution
        index_vector *= -1  # transformMapFrameToBufferOrder
        return index_vector.astype(int)

    def at(self, idx):
        return self.map_matrix[idx[0], idx[1]]

    def nanMapAt(self, idx):
        return self.nan_map[idx[0], idx[1]]

    def getLocalMap(self, position, yaw, local_map_shape):  # pose = [xyz,rpy]
        # base_position_2d = np.array([TF_POSE_REF_LIST.transform.translation.x, TF_POSE_REF_LIST.transform.translation.y])
        # pose = msg_to_tf(TF_POSE_REF_LIST)
        base_position_2d = position[:2]

        rot = np.zeros((2, 2))
        rot[0, 0] = np.cos(yaw)
        rot[0, 1] = -np.sin(yaw)
        rot[1, 0] = np.sin(yaw)
        rot[1, 1] = np.cos(yaw)

        n_x = local_map_shape[2]
        n_y = local_map_shape[3]
        dx = local_map_shape[0] / n_x
        dy = local_map_shape[1] / n_y

        map_local = np.zeros([2, n_x, n_y], dtype=np.float32)

        dx_ = dx * rot[:, 0]
        dy_ = dy * rot[:, 1]
        for x_idx in range(n_x):
            for y_idx in range(n_y):
                scan_position = base_position_2d.copy()
                scan_position -= dx_ * (x_idx - n_x / 2)
                scan_position -= dy_ * (y_idx - n_y / 2)  # Following anybotics' grid map indexing

                map_idx = self.getIndexFromPosition(scan_position)
                if abs(map_idx[0]) > self.size[0] - 1 or abs(map_idx[1]) > self.size[1] - 1:
                    map_local[0, x_idx, y_idx] = -0.0  # Elev. map
                    map_local[1, x_idx, y_idx] = 1.0  # Nan map
                else:
                    map_local[1, x_idx, y_idx] = self.nanMapAt(map_idx)
                    map_local[0, x_idx, y_idx] = self.at(map_idx) - position[2]

        # map_local = cv2.resize(map_local[0], (500, 500), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('test_elevation', (self.map_matrix - np.min(self.map_matrix))/(np.max(self.map_matrix)- np.min(self.map_matrix)))
        # cv2.imshow('test_elevation1', (map_local - np.min(map_local))/(np.max(map_local) - np.min(map_local)))
        # # print(pose[:2],",", pose[3]/np.pi * 180)
        # # print(self.position)
        # # print(self.getIndexFromPosition(pose[:2]))
        # cv2.waitKey(0)
        return map_local

    def getRaisimMap(self, raisim_world):
        map_raisim = self.map_matrix.transpose()
        map_raisim = np.flip(map_raisim)

        raisim_hmap = raisim_world.addHeightMap(
            x_scale=self.size[0] * self.resolution,
            y_scale=self.size[1] * self.resolution,
            x_samples=self.size[0],
            y_samples=self.size[1],
            x_center=self.position[0],
            y_center=self.position[1],
            heights=map_raisim.flatten()
        )

        return raisim_hmap
