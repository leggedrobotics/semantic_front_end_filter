from mimetypes import init


import numpy as np
import matplotlib.pyplot as plt
import msgpack
import trimesh
import numpy as np
import warp as wp
from warp.torch import to_torch as wp_to_torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation

import rospy
import tf

from messages.imageMessage import Camera

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

def create_plane(points):
    mesh = trimesh.Trimesh(vertices=points, faces=np.array([[0, 1, 3], [1, 2, 3]]))
    return mesh


def create_height_field(height_field, voxel_size=0.1, center_point=np.array([0, 0, 0])):
    meshes = []
    for x in range(height_field.shape[0] - 1):
        for y in range(height_field.shape[1] - 1):
            ls = []
            for x_ in range(x, x + 2):
                for y_ in range(y, y + 2):
                    ls.append([x_ * voxel_size + center_point[0], y_ * voxel_size + center_point[1], height_field[x_, y_]])
                    
            st = ls[1]
            ls[1] = ls[0]
            ls[0] = st
            
            if np.isnan( np.array([ls])).any():
                continue
                
            meshes.append(create_plane(np.array([ls])[0]))

    return trimesh.util.concatenate([meshes])

wp.init()

@wp.kernel
def raycast_generic(
    mesh: wp.uint64,
    ray_origin: wp.array(dtype=wp.vec3),
    ray_dir: wp.array(dtype=wp.vec3),
    height: wp.array(dtype=wp.float32),
):

    tid = wp.tid()

    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index

    # ray cast against the mesh wp.vec3(0.0, 0.0, -1.0)
    if wp.mesh_query_ray(mesh, ray_origin[tid], ray_dir[tid], 1.0e6, t, u, v, sign, n, f):
        height[tid] = t


@wp.kernel
def raycast_normal(
    mesh: wp.uint64,
    ray_origin: wp.array(dtype=wp.vec3),
    height: wp.array(dtype=wp.float32),
):

    tid = wp.tid()

    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index
    ray_dir = wp.vec3(0.0, 0.0, -1.0)
    # ray cast against the mesh wp.vec3(0.0, 0.0, -1.0)
    if wp.mesh_query_ray(mesh, ray_origin[tid], ray_dir, 1.0e6, t, u, v, sign, n, f):
        height[tid] = t




class DIFG:
    def __init__(self, ground_map_path, camera_calibration_path = None, cfg = None):
        with open(ground_map_path, "rb") as data_file:
            data = data_file.read()
            ground_dict = msgpack.unpackb(data)

            ground_dict["yRealRange"], ground_dict["xRealRange"]
        
        height_field = np.array( ground_dict["GPMap"] )
        center_point = np.array( [ground_dict["xRealRange"][0],ground_dict["yRealRange"][0],0])
        
        mesh = create_height_field( height_field, ground_dict["res"], center_point)

        # Raycasting a single image
        m2 = mesh.as_open3d
                        
        vertices = np.array(m2.vertices).astype(np.float32)
        triangles = np.array(m2.triangles).astype(np.uint32)

        self.wp_device = "cuda" if str(device).find("cuda") != -1 else "cpu"
        self.wp_mesh = wp.Mesh(
            points=wp.array(np.asarray(vertices), dtype=wp.vec3, device=self.wp_device),
            indices=wp.array(triangles.astype(np.int32), dtype=int, device=self.wp_device),
        )

        
        
        if(camera_calibration_path != None):
            self.camera = Camera(camera_calibration_path,0,cfg)
            K = self.camera.camera_matrix
        else:
            fx,fy,cx,cy = 300,300,320,240
            K = np.array( [[fx, 0, cx],[0, fy, cy], [0,0,1]] )
        self.W,self.H = 640,480
        pixel_cor = np.mgrid[0:self.W,0:self.H]
        pixel_cor_hom = np.concatenate( [ pixel_cor, np.ones_like(pixel_cor[None,0,:,:])], axis=0 )

        ray_dir = (np.linalg.inv(K) @ (pixel_cor_hom.reshape(3,-1))).T
        self.ray_dir = ray_dir/ np.linalg.norm(ray_dir, axis=1)[:,None]
        # https://i.stack.imgur.com/AGwu9.jpg  
        

    # def showMesh(self):
    #     pixel_cor = np.mgrid[vertices.min(axis=0)[0]:vertices.max(axis=0)[0]:0.1,
    #     vertices.min(axis=0)[1]:vertices.max(axis=0)[1]:0.1]
    #     vec = torch.from_numpy(self.pixel_cor).to(device).T.reshape( (-1,2))
    #     vec = torch.cat( [vec, torch.zeros_like(vec[:,0][:,None])] ,dim=1)
    #     vec[:,2] = 10

    #     vec = vec.type(torch.float32)
    #     res = self.get_distance(vec)
    #     res /= res.max()
    #     dis = res.reshape((self.pixel_cor.shape[2], self.pixel_cor.shape[1])).cpu().numpy()
    #     Image.fromarray(np.uint8( dis*255))

    def get_distance(self, start_points, directions=None):
            """Raycasts environment mesh.
            If no direction is provide raycasts z down (0,0,-1) for all rays.

            Args:
                start_points (torch.Tensor): origin of the rays
                directions (torch.Tensor): optionally directions of rays
            Returns:
                [torch.Tensor]: ray lengts in meter
            """
            ray_origin = wp.types.array(
                ptr=start_points.data_ptr(),
                dtype=wp.vec3,
                length=start_points.shape[0],
                copy=False,
                owner=False,
                requires_grad=False,
                device=start_points.device.type,
            )
            ray_origin.tensor = start_points
            n = len(start_points)
            distances = wp.zeros(n, dtype=wp.float32, device=self.wp_device)

            if directions is not None:
                ray_dir = wp.types.array(
                    ptr=directions.data_ptr(),
                    dtype=wp.vec3,
                    length=directions.shape[0],
                    copy=False,
                    owner=False,
                    requires_grad=False,
                    device=directions.device.type,
                )
                ray_dir.tensor = directions
                kernel = raycast_generic
                inputs = [self.wp_mesh.id, ray_origin, ray_dir, distances]
            else:
                kernel = raycast_normal
                inputs = [self.wp_mesh.id, ray_origin, distances]

            wp.launch(kernel=kernel, dim=n, inputs=inputs, device=self.wp_device)
            wp.synchronize()
            distances = wp_to_torch(distances)
            return distances
        
    def getDImage(self, transition, rotation):
        H_map_cam = np.eye(4)

        # H_map_cam[:3,3] =  np.array( [[410.51799114,  82.08771105,  -1.37993082]]) # position over one of the trianlge vertices
        H_map_cam[:3,3] =  np.array( [transition])

        H_map_cam[:3,:3] = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix() # looking down
        H_map_cam[:3,:3] = Rotation.from_euler('yz', [0, 180], degrees=True).as_matrix() @ H_map_cam[:3,:3]
        # H_map_cam[:3,:3] = R.from_euler('xyz', [-90.059, 30, -120.744], degrees=True).as_matrix() # looking down


        R = torch.from_numpy( H_map_cam ).to(device)[:3,:3]
        directions = torch.from_numpy( self.ray_dir )
        directions = directions.to(device).clone()
        directions = (directions @ R).type(torch.float32)

        start_points = torch.from_numpy( H_map_cam[:3,3] ).to(device)
        start_points = start_points[None,:].repeat(self.ray_dir.shape[0],1).type(torch.float32)

        dis = self.get_distance(start_points, directions)

        dis = dis.reshape(self.W,self.H).cpu()
        # To get nice colors
        dis[dis==0] = dis[dis!=0].min()- (dis[dis!=0].max()-dis[dis!=0].min())/5
        dis = dis/dis.max()*255
        plt.imshow(dis.T)
        plt.show()

def main():
    d = DIFG('./Example_Files/GroundMap.msgpack')
    d.getDImage(transition=[119.193, 429.133, -1], rotation=[-90, 0, -90])    

if __name__ == '__main__':
    main()