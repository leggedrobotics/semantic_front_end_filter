"""
Chenyu 2022-06-07
visulization of the network using elevation_mapping_cupy
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "elevation_mapping_cupy", "elevation_mapping_cupy", "script"))
import cupy as xp
from elevation_mapping_cupy.parameter import Parameter
from elevation_mapping_cupy.elevation_mapping import ElevationMap
import numpy as np
from scipy.spatial.transform import Rotation as Rotation

# Load modules for the GFT
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "Labelling"))
from GroundfromTrajs import GFT


from pointcloudUtils import RaycastCamera


# Define a function to draw a marker of the robot on plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def plt_add_robot_patch(ax, pos):
    """
    pos is the form data["pose"]["map"]: (x,y,z,rx,ry,rz,rw)
    """
    xyth = [*pos[:2], 
        Rotation.from_quat(pos[3:]).as_euler('xyz',degrees=False)[2]]

    robot_l = 0.4
    robot_w = 0.3
    robot = patches.Polygon([[-robot_l,-robot_w], [robot_l,-robot_w], 
                             [robot_l+0.5, 0], [robot_l,robot_w], 
                             [-robot_l,robot_w]], color = "r", fill = True)
    robot_center = patches.Circle((0,0),0.2, color = "w",fill = True)
    t2 = mpl.transforms.Affine2D().rotate(xyth[2]).translate(xyth[0],xyth[1])+ ax.transData
    robot.set_transform(t2) 
    robot_center.set_transform(t2)
    pat_r = ax.add_patch(robot)
    pat_c = ax.add_patch(robot_center)
    return [pat_r, pat_c]


def get_xy_from_elevation(elevation):
    cell_n = elevation.cell_n
    resolution = elevation.resolution
    pos = elevation.center
    x = ((xp.arange(cell_n-1,1,-1)+0.5) - 0.5 * cell_n )*resolution + pos[0]
    y = ((xp.arange(cell_n-1,1,-1)+0.5) - 0.5 * cell_n )*resolution + pos[1]
    x,y = xp.meshgrid(x,y, indexing="ij")
    x,y = xp.asnumpy(x.reshape(-1)), xp.asnumpy(y.reshape(-1))
    return x,y

def plt_add_elevation_map(ax, elevation, data, vmin, vmax, colorbar = True):
    x,y = get_xy_from_elevation(elevation)
    sc = ax.scatter(x = x, y = y, 
            c = data.reshape(-1),s = 2, vmin = vmin, vmax = vmax)
    if colorbar:
        plt.colorbar(sc,ax = ax)
    return [sc,]

# def plt_add_traj_map(ax, elevation, gft, vmin, vmax):
#     x,y = get_xy_from_elevation(elevation)
#     data = [gft.getHeight(a,b, "GPMap")[0] for a,b in zip(x,y)]
#     sc = ax.scatter(x = x, y = y, 
#             c = data,s = 2, vmin = vmin, vmax = vmax)
#     plt.colorbar(sc,ax = ax)

class WorldViewElevationMap:
    """
    A convinent warper for ElevationMap
    """
    def __init__(self, resolution, map_length, init_with_initialize_map = True):
        self.param = Parameter(
            use_chainer=False, weight_file=os.path.join(os.path.dirname(__file__), "../elevation_mapping_cupy/elevation_mapping_cupy/config/weights.dat"), 
                plugin_config_file=os.path.join(os.path.dirname(__file__), "../elevation_mapping_cupy/elevation_mapping_cupy/config/plugin_config.yaml"),
            resolution = resolution, map_length=map_length, 
            min_valid_distance = 0, enable_visibility_cleanup = False, enable_edge_sharpen=False,
            enable_overlap_clearance = False,
            # max_height_range=100, ramped_height_range_c=10000,
            # initial_variance= 1000, initialized_variance= 1000, max_variance=1000, sensor_noise_factor=0,
            # mahalanobis_thresh = 1000,
            dilation_size_initialize = 0.
        )
        self.init_with_initialize_map = init_with_initialize_map
        self.reset()
        
    @property
    def cell_n(self):
        return self.elevation.cell_n
    
    @property
    def resolution(self):
        return self.elevation.resolution

    @property
    def center(self):
        return self.elevation.center

    def reset(self):
        self.elevation = ElevationMap(self.param)
        self.is_init = self.init_with_initialize_map
    
    def move_to_and_input(self, pos, points):
        """
        Move to the pos(x,y,z) and input the points(in world frame)
        points is numpy array
        """
        points = xp.array(points)
        self.elevation.move_to(pos)
        R = xp.eye(3)
        t = xp.zeros(3)
        if(not self.is_init):
            print("input called")
            self.elevation.input(points, R, t, 0, 0)
        else:
            print("initialize_map called")
            self.elevation.initialize_map(points.copy())
            self.is_init = False
    
    def get_elevation_map(self):
        """
        Output the elevation map, in the form of numpy array
        """
        data = np.zeros((self.elevation.cell_n - 2, self.elevation.cell_n - 2), dtype=np.float32)
        self.elevation.get_map_with_name_ref("elevation", data)
        return xp.asnumpy(data)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # for loading model
    from train import *
    import msgpack
    import msgpack_numpy as m

    m.patch()
    sys.argv = ["train.py","--data_path", "/media/chenyu/T7/Data/extract_trajectories_003_augment", 
                "--bs", "1", 
            "--slim_dataset", "True"]
    args = parse_args()
    # from rosvis import *
    checkpoint_path = "checkpoints/share/2022-05-14-00-19-41/UnetAdaptiveBins_best.pt"
    model = models.UnetAdaptiveBins.build(n_bins=args.modelconfig.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                            norm=args.modelconfig.norm, use_adabins=True)
    model,_,_ = model_io.load_checkpoint(checkpoint_path ,model) 
    model.cuda()

    ## Loader
    data_loader = DepthDataLoader(args, 'online_eval').data
    # sample = next(iter(data_loader))
    data_iter = iter(data_loader)
    sample = next(data_iter)
    rawdatapath = sample["path"][0].replace("extract_trajectories_003_augment", "extract_trajectories_002")
    ground_map_filepath = os.path.join(os.path.dirname(rawdatapath), "GroundMap.msgpack")

    data_loader.dataset.filenames[0]
    filenames = [f'/media/chenyu/T7/Data/extract_trajectories_003_augment/WithPointCloudReconstruct_2022-03-26-22-28-54_0/traj_0_datum_{i}.msgpack' for i in range(100)]
    filenames = list(filter(lambda f: os.path.exists(f),filenames))
    data_loader.dataset.filenames = filenames
    data_iter = iter(data_loader)



    gft = GFT(GroundMapFile = ground_map_filepath)
    camera = RaycastCamera("../Labelling/configs")
    elevation = WorldViewElevationMap(resolution = 0.1, map_length = 10, init_with_initialize_map = True)
    elevation_pred_fusion = WorldViewElevationMap(resolution = 0.1, map_length = 10, init_with_initialize_map = False)
    elevation_pc_fusion = WorldViewElevationMap(resolution = 0.1, map_length = 10, init_with_initialize_map = False)


    fig, axs = plt.subplots(3,3, figsize = (21,15))
    [ax.get_xaxis().set_animated(True) for ax in axs.reshape(-1)]
    [ax.get_yaxis().set_animated(True) for ax in axs.reshape(-1)]
    need_color_bar = True
    
    def animate(i):
        global data_iter, need_color_bar
        try:
            sample = next(data_iter)
        except Exception as e:
            print(e)
            data_iter = iter(data_loader)
            sample = next(data_iter)
        rawdatapath = sample["path"][0].replace("extract_trajectories_003_augment", "extract_trajectories_002")
        with open(rawdatapath, "rb") as data_file:
            byte_data = data_file.read()
            rawdata = msgpack.unpackb(byte_data)

        ## Prepare data
        pose = rawdata["pose"]["map"]
        print(pose[:3])
        pc_points = rawdata["pointcloud"][:,:3]
        pos = rawdata["pose"]["map"][:3].copy()
        elevation.move_to_and_input(pos, pc_points)
        x,y = get_xy_from_elevation(elevation)
        data_traj = np.array([gft.getHeight(a,b, "GPMap")[0] for a,b in zip(x,y)])

        elevation.reset()
        elevation.move_to_and_input(pos, pc_points)
        elevation_pc_fusion.move_to_and_input(pos, pc_points)
        map_pc = elevation.get_elevation_map()
        map_pc_fusion = elevation_pc_fusion.get_elevation_map()


        elevation.reset()
        _, pred = model(sample["image"].cuda())
        depth = sample["depth"][0][0].T
        pred = pred[0].detach()#.numpy()
        pred = nn.functional.interpolate(torch.tensor(pred).detach()[None,...], torch.tensor(depth.T).shape[-2:], mode='bilinear', align_corners=True)
        pred = pred[0][0].T
        

        pose = rawdata["pose"]["map"] # xyzw
        pts = camera.project_depth_to_cloud(pose, pred)

        height_mask = pts[:,2] < pose[2]
        pred_points = pts[height_mask].cpu()

        elevation.move_to_and_input(pos, pred_points)
        elevation_pred_fusion.move_to_and_input(pos, pred_points)
        map_pred = elevation.get_elevation_map()
        map_pred_fusion = elevation_pred_fusion.get_elevation_map()


        elems = []
        axs[0,0].set_title("pc point clouds")
        sc = axs[0,0].scatter(x = pc_points[:,0], y = pc_points[:,1], c = pc_points[:,2],
                            s = 1, vmin = pos[2]-1.5, vmax = pos[2]+0.5)
        if need_color_bar: plt.colorbar(sc, ax = axs[0,0])
        elems.append(sc)
        axs[0,0].set_xlim((pos[0]-5, pos[0]+5))
        axs[0,0].set_ylim((pos[1]-5, pos[1]+5))
        elems += plt_add_robot_patch(axs[0,0], pose)

        axs[0,1].set_title("pc points elevation map")
        elems += plt_add_elevation_map(axs[0,1], elevation, map_pc, vmin = pos[2]-1.5, vmax = pos[2]+0.5,colorbar = need_color_bar)
        elems += plt_add_robot_patch(axs[0,1], pose)
        axs[0,1].set_xlim((pos[0]-5, pos[0]+5))
        axs[0,1].set_ylim((pos[1]-5, pos[1]+5))

        axs[0,2].set_title("pc points fusion map")
        elems += plt_add_elevation_map(axs[0,2], elevation_pc_fusion, map_pc_fusion, 
            vmin = pos[2]-1.5, vmax = pos[2]+0.5,colorbar = need_color_bar)
        elems += plt_add_robot_patch(axs[0,2], pose)
        axs[0,2].set_xlim((pos[0]-5, pos[0]+5))
        axs[0,2].set_ylim((pos[1]-5, pos[1]+5))


        axs[1,0].set_title("pred point clouds")
        sc = axs[1,0].scatter(x = pred_points[:,0], y = pred_points[:,1], c = pred_points[:,2],
                            s = 1, vmin = pos[2]-1.5, vmax = pos[2]+0.5)
        if need_color_bar: plt.colorbar(sc, ax = axs[0,0])
        elems.append(sc)
        axs[1,0].set_xlim((pos[0]-5, pos[0]+5))
        axs[1,0].set_ylim((pos[1]-5, pos[1]+5))
        elems += plt_add_robot_patch(axs[1,0], pose)

        axs[1,1].set_title("pc points elevation map")
        elems += plt_add_elevation_map(axs[1,1], elevation, map_pred, vmin = pos[2]-1.5, vmax = pos[2]+0.5, colorbar = need_color_bar)
        elems += plt_add_robot_patch(axs[1,1], pose)
        axs[1,1].set_xlim((pos[0]-5, pos[0]+5))
        axs[1,1].set_ylim((pos[1]-5, pos[1]+5))

        axs[1,2].set_title("pred fusion map")
        elems += plt_add_elevation_map(axs[1,2], elevation_pred_fusion, map_pred_fusion, 
            vmin = pos[2]-1.5, vmax = pos[2]+0.5,colorbar = need_color_bar)
        elems += plt_add_robot_patch(axs[1,2], pose)
        axs[1,2].set_xlim((pos[0]-5, pos[0]+5))
        axs[1,2].set_ylim((pos[1]-5, pos[1]+5))

        x,y = get_xy_from_elevation(elevation)
        data_traj = np.array([gft.getHeight(a,b, "GPMap")[0] for a,b in zip(x,y)])
        data_error = np.array(map_pred.reshape(-1) - data_traj)
        data_error_fusion = np.array(map_pred_fusion.reshape(-1) - data_traj)

        axs[2,0].set_title("traj elevation map")
        elems += plt_add_elevation_map(axs[2,0], elevation, data_traj, vmin = pos[2]-1.5, vmax = pos[2]+0.5, colorbar = need_color_bar)
        elems += plt_add_robot_patch(axs[2,0], pose)
        axs[2,0].set_xlim((pos[0]-5, pos[0]+5))
        axs[2,0].set_ylim((pos[1]-5, pos[1]+5))


        axs[2,1].set_title("Error")
        elems += plt_add_elevation_map(axs[2,1], elevation, data_error, vmin = -0.5, vmax = 0.5, colorbar = need_color_bar)
        elems += plt_add_robot_patch(axs[2,0], pose)
        axs[2,1].set_xlim((pos[0]-5, pos[0]+5))
        axs[2,1].set_ylim((pos[1]-5, pos[1]+5))

        axs[2,2].set_title("fusion Error")
        elems += plt_add_elevation_map(axs[2,2], elevation, data_error_fusion, vmin = -0.5, vmax = 0.5, colorbar = need_color_bar)
        elems += plt_add_robot_patch(axs[2,0], pose)
        axs[2,2].set_xlim((pos[0]-5, pos[0]+5))
        axs[2,2].set_ylim((pos[1]-5, pos[1]+5))


        # elems = []
        # elems+= plt_add_elevation_map(ax, elevation, data_traj, vmin = pos[2]-1.5, vmax = pos[2]+0.5,colorbar = False)
        # elems+= plt_add_robot_patch(ax, pose)
        # ax.set_xlim((pos[0]-5, pos[0]+5))
        # ax.set_ylim((pos[1]-5, pos[1]+5))
        # elems += [ax.get_yticklabels(), ax.get_xticklabels()]
        # elems += [ax.xaxis, ax.yaxis]
        need_color_bar = False
        return elems

    ani = animation.FuncAnimation(
        fig, animate, frames = 10, # init_func=init,
        interval=200, repeat_delay=1000, blit=True)
    # writervideo = animation.FFMpegWriter(fps=60)

    ani.save('visulization/elevation_animation.mp4')
    plt.show()

