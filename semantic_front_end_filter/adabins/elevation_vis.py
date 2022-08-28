"""
Chenyu 2022-06-07
visulization of the network using elevation_mapping_cupy
"""
import os
import sys
from semantic_front_end_filter import SEMANTIC_FRONT_END_FILTER_ROOT_PATH
sys.path.append(os.path.join(os.path.dirname(SEMANTIC_FRONT_END_FILTER_ROOT_PATH), "elevation_mapping_cupy", "elevation_mapping_cupy", "script"))
import cupy as xp
from elevation_mapping_cupy.parameter import Parameter
from elevation_mapping_cupy.elevation_mapping import ElevationMap
import numpy as np
from scipy.spatial.transform import Rotation
from ruamel.yaml import YAML
import yaml
# from dacite import from_dict

# Load modules for the GFT

from semantic_front_end_filter.Labelling.GroundfromTrajs import GFT
from .pointcloudUtils import RaycastCamera


# Define a function to draw a marker of the robot on plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _robot_patch(pos):
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
    R = np.array([[np.cos(xyth[2]), -np.sin(xyth[2])],
                  [np.sin(xyth[2]),  np.cos(xyth[2])]])
    t = np.array([xyth[:2]])
    robot.set_xy( (R@ robot.get_xy().T ).T+t )
    robot_center.center =  (robot_center.center[0] + t[0][0], robot_center.center[1] + t[0][1])
    # t2 = mpl.transforms.Affine2D().rotate(xyth[2]).translate(xyth[0],xyth[1])#+ ax.transData
    # robot.set_transform(t2) 
    # robot_center.set_transform(t2)
    return (robot, robot_center)

def plt_add_robot_patch(ax, pos):
    (robot, robot_center) = _robot_patch(pos)
    pat_r = ax.add_patch(robot)
    pat_c = ax.add_patch(robot_center)
    return [pat_r, pat_c]

def plt_set_robot_patch(pat_r, pat_c, pos):
    """
    pos is the form data["pose"]["map"]: (x,y,z,rx,ry,rz,rw)
    """
    (robot, robot_center) = _robot_patch(pos)
    pat_r.set_xy(robot.get_xy())
    pat_c.set(center = robot_center.center)
    

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

def plt_set_elevation_map(sc, xy, data):
    x,y = xy
    offsets = np.stack([x,y]).T
    sc.set_offsets(offsets)
    sc.set_array(data.reshape(-1))
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
        """
        arg: init_with_initialize_map: (True, False, None, "nearest", "linear", "cubic")
        """
        # self.param = Parameter(
        #     use_chainer=False, weight_file=os.path.join(os.path.dirname(__file__), "../../elevation_mapping_cupy/elevation_mapping_cupy/config/weights.dat"), 
        #         plugin_config_file=os.path.join(os.path.dirname(__file__), "../../elevation_mapping_cupy/elevation_mapping_cupy/config/plugin_config.yaml"),
        #     resolution = resolution, map_length=map_length, 
        #     min_valid_distance = 0, enable_visibility_cleanup = False, enable_edge_sharpen=False,
        #     enable_overlap_clearance = False,
        #     # max_height_range=100, ramped_height_range_c=10000,
        #     # initial_variance= 1000, initialized_variance= 1000, max_variance=1000, sensor_noise_factor=0,
        #     # mahalanobis_thresh = 1000,
        #     dilation_size_initialize = 0.
        # )
        with open(os.path.join(os.path.dirname(__file__),"cfgs/elevation_mapping_cupy.yaml"), 'rb') as f:
            conf = yaml.safe_load(f.read())
        self.param = Parameter()
        for key, value in conf['elevation_mapping'].items():
            self.param.set_value(key, value)
        self.param.resolution = resolution
        self.param.map_length = map_length
        self.param.weight_file=os.path.join(os.path.dirname(__file__), "../../elevation_mapping_cupy/elevation_mapping_cupy/config/weights.dat")
        self.param.plugin_config_file=os.path.join(os.path.dirname(__file__), "../../elevation_mapping_cupy/elevation_mapping_cupy/config/plugin_config.yaml")
        p = dict(enable_overlap_clearance = False, max_height_range = 10, ramped_height_range_c = 10)
        for key, value in p.items():
            self.param.set_value(key, value)

        self.init_with_initialize_map = "cubic" if init_with_initialize_map == True else init_with_initialize_map
        self.init_with_initialize_map = False if self.init_with_initialize_map is None else self.init_with_initialize_map
        assert (self.init_with_initialize_map==False or self.init_with_initialize_map in ["nearest", "linear", "cubic"])
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
        R = xp.eye(3,dtype = float)
        t = xp.array(pos).astype(float)
        points = xp.array(points)# change the frame of points, translate them into pos's frame
        self.elevation.move_to(pos)
        if(not self.is_init):
            points -= t
            self.elevation.input(points, R, t, 0, 0)
        else:
            print("initialize_map called")
            self.elevation.initialize_map(points.copy(), self.is_init)
            self.is_init = False
    
    def get_elevation_map(self):
        """
        Output the elevation map, in the form of numpy array
        """
        data = np.zeros((self.elevation.cell_n - 2, self.elevation.cell_n - 2), dtype=np.float32)
        self.elevation.get_map_with_name_ref("elevation", data)
        return xp.asnumpy(data)


def load_param_from_path(data_path):
    model_cfg = YAML().load(open(os.path.join(data_path, "ModelConfig.yaml"), 'r'))
    return model_cfg

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # for loading model
    from train import *
    import msgpack
    import msgpack_numpy as m

    m.patch()
    sys.argv = ["train.py","--data_path", "/media/anqiao/Semantic/Data/extract_trajectories_003_augment", 
                "--bs", "1", 
            "--slim_dataset", "True"]
    # parser = ArgumentParser()
    # parser.add_argument("--model_path", default="/home/anqiao/tmp/semantic_front_end_filter/adabins/checkpoints/2022-08-03-16-26-08/UnetAdaptiveBins_latest.pt")
   
    args.model_path="/home/anqiao/tmp/semantic_front_end_filter/adabins/checkpoints/2022-08-03-16-26-08/UnetAdaptiveBins_latest.pt" # parser.add_argument("--dataset_path", default="/home/anqiao/tmp/semantic_front_end_filter/Labelling/extract_trajectories")   
    args.data_path = "/media/anqiao/Semantic/Data/extract_trajectories_003_augment"
    args.slim_dataset = "True"
    # parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    # parser.add_argument("--name", default="UnetAdaptiveBins")
    # parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    # parser.add_argument("--root", default=".", type=str,
    #                     help="Root folder to save data in")
    # parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")
    # parser.add_argument("--tqdm", default=False, action="store_true", help="show tqdm progress bar")

    # parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    # parser.add_argument("--tags", default='', type=str, help="Wandb tags, seperate by `,`")     
    # args = parse_args(parser)
    model_cfg = load_param_from_path(os.path.dirname(args.model_path))

    # from rosvis import *
    # checkpoint_path = "checkpoints/share/2022-05-14-00-19-41/UnetAdaptiveBins_best.pt"
    # args = parse_args(parser)
    checkpoint_path = "/home/anqiao/tmp/semantic_front_end_filter/adabins/checkpoints/2022-08-03-16-26-08/UnetAdaptiveBins_latest.pt"
    # model = models.UnetAdaptiveBins.build(n_bins=args.modelconfig.n_bins, min_val=args.min_depth, max_val=args.max_depth,
    #                                         norm=args.modelconfig.norm, use_adabins=True)
    # model,_,_ = model_io.load_checkpoint(checkpoint_path ,model)
    model = models.UnetAdaptiveBins.build(n_bins=args.modelconfig.n_bins, input_channel = 4, min_val=args.min_depth, max_val=args.max_depth,
                                            norm=args.modelconfig.norm, use_adabins=model_cfg["use_adabins"], deactivate_bn = model_cfg["deactivate_bn"], skip_connection = model_cfg["skip_connection"])
    model = model_io.load_checkpoint(args.model_path ,model)[0] 
    model.cuda()

    ## Loader
    data_loader = DepthDataLoader(args, 'online_eval').data
    # sample = next(iter(data_loader))
    data_iter = iter(data_loader)
    sample = next(data_iter)
    rawdatapath = sample["path"][0].replace("extract_trajectories_003_augment", "extract_trajectories_003/extract_trajectories")
    ground_map_filepath = os.path.join(os.path.dirname(rawdatapath), "GroundMap.msgpack")

    data_loader.dataset.filenames[0]
    filenames = [f'/media/anqiao/Semantic/Data/extract_trajectories_003_augment/WithPointCloudReconstruct_2022-03-26-22-28-54_0/traj_0_datum_{i}.msgpack' for i in range(50)]
    filenames = list(filter(lambda f: os.path.exists(f),filenames))
    data_loader.dataset.filenames = filenames
    data_iter = iter(data_loader)



    gft = GFT(GroundMapFile = ground_map_filepath)
    # camera = RaycastCamera("../Labelling/configs")
    camera = RaycastCamera("/home/anqiao/tmp/semantic_front_end_filter/anymal_c_subt_semantic_front_end_filter/config/calibrations/alphasense")
    elevation = WorldViewElevationMap(resolution = 0.1, map_length = 10, init_with_initialize_map = True)
    elevation_pred_fusion = WorldViewElevationMap(resolution = 0.1, map_length = 10, init_with_initialize_map = False)
    elevation_pc_fusion = WorldViewElevationMap(resolution = 0.1, map_length = 10, init_with_initialize_map = False)


    class AnimateElevation:
        def __init__(self, data_loader):
            
            self.fig, self.axs = plt.subplots(3,3, figsize = (21,15))
            [ax.get_xaxis().set_animated(True) for ax in self.axs.reshape(-1)]
            [ax.get_yaxis().set_animated(True) for ax in self.axs.reshape(-1)]
            self.data_loader = data_loader
            self.data_iter = iter(self.data_loader)
            self.stream = self.data_stream()
            self.need_color_bar = True

        def setup_plot(self):
            try:
                data = next(self.stream)
            except Exception as e:
                self.stream = self.data_stream()
                data = next(self.stream)
            pose = data["pose"]
            pc_points = data["pc_points"]
            pred_points = data["pred_points"]
            map_pc = data["map_pc"]
            map_pc_fusion = data["map_pc_fusion"]
            map_pred = data["map_pred"]
            map_pred_fusion = data["map_pred_fusion"]
            (x,y) = data["xy"]
            data_traj = data["data_traj"]
            data_error = data["data_error"]
            data_error_fusion = data["data_error_fusion"]

            self.elems = []
            self.axs[0,0].set_title("pc point clouds")
            self.ax00sc = self.axs[0,0].scatter(x = pc_points[:,0], y = pc_points[:,1], c = pc_points[:,2],
                                s = 1, vmin = pose[2]-1.5, vmax = pose[2]+0.5)
            if self.need_color_bar: plt.colorbar(self.ax00sc, ax = self.axs[0,0])
            self.elems.append(self.ax00sc)
            self.axs[0,0].set_xlim((pose[0]-5, pose[0]+5))
            self.axs[0,0].set_ylim((pose[1]-5, pose[1]+5))
            self.ax00ptch = plt_add_robot_patch(self.axs[0,0], pose)
            self.elems += self.ax00ptch

            self.axs[0,1].set_title("pc points elevation map")
            self.ax01sc =  plt_add_elevation_map(self.axs[0,1], elevation, map_pc, vmin = pose[2]-1.5, vmax = pose[2]+0.5,colorbar = self.need_color_bar)
            self.ax01ptch = plt_add_robot_patch(self.axs[0,1], pose)
            self.elems += self.ax01sc
            self.elems += self.ax01ptch
            self.axs[0,1].set_xlim((pose[0]-5, pose[0]+5))
            self.axs[0,1].set_ylim((pose[1]-5, pose[1]+5))

            self.axs[0,2].set_title("pc points fusion map")
            self.ax02sc = plt_add_elevation_map(self.axs[0,2], elevation_pc_fusion, map_pc_fusion, 
                vmin = pose[2]-1.5, vmax = pose[2]+0.5,colorbar = self.need_color_bar)
            self.ax02ptch= plt_add_robot_patch(self.axs[0,2], pose)
            self.elems += self.ax02sc
            self.elems += self.ax02ptch
            self.axs[0,2].set_xlim((pose[0]-5, pose[0]+5))
            self.axs[0,2].set_ylim((pose[1]-5, pose[1]+5))

            self.axs[1,0].set_title("pred point clouds")
            self.ax10sc = self.axs[1,0].scatter(x = pred_points[:,0], y = pred_points[:,1], c = pred_points[:,2],
                                s = 1, vmin = pose[2]-1.5, vmax = pose[2]+0.5)
            if self.need_color_bar: plt.colorbar(self.ax10sc, ax = self.axs[1,0])
            self.elems.append(self.ax10sc)
            self.axs[1,0].set_xlim((pose[0]-5, pose[0]+5))
            self.axs[1,0].set_ylim((pose[1]-5, pose[1]+5))
            self.ax10ptch = plt_add_robot_patch(self.axs[1,0], pose)
            self.elems += self.ax10ptch

            self.axs[1,1].set_title("pc points elevation map")
            self.ax11sc = plt_add_elevation_map(self.axs[1,1], elevation, map_pred, vmin = pose[2]-1.5, vmax = pose[2]+0.5, colorbar = self.need_color_bar)
            self.ax11ptch = plt_add_robot_patch(self.axs[1,1], pose)
            self.elems+= self.ax11sc
            self.elems+= self.ax11ptch
            self.axs[1,1].set_xlim((pose[0]-5, pose[0]+5))
            self.axs[1,1].set_ylim((pose[1]-5, pose[1]+5))

            self.axs[1,2].set_title("pred fusion map")
            self.ax12sc = plt_add_elevation_map(self.axs[1,2], elevation_pred_fusion, map_pred_fusion, 
                vmin = pose[2]-1.5, vmax = pose[2]+0.5,colorbar = self.need_color_bar)
            self.ax12ptch = plt_add_robot_patch(self.axs[1,2], pose)
            self.elems += self.ax12sc
            self.elems += self.ax12ptch
            self.axs[1,2].set_xlim((pose[0]-5, pose[0]+5))
            self.axs[1,2].set_ylim((pose[1]-5, pose[1]+5))

               
            self.axs[2,0].set_title("traj elevation map")
            self.ax20sc = plt_add_elevation_map(self.axs[2,0], elevation, data_traj, vmin = pose[2]-1.5, vmax = pose[2]+0.5, colorbar = self.need_color_bar)
            self.ax20ptch = plt_add_robot_patch(self.axs[2,0], pose)
            self.elems += self.ax20sc
            self.elems += self.ax20ptch
            self.axs[2,0].set_xlim((pose[0]-5, pose[0]+5))
            self.axs[2,0].set_ylim((pose[1]-5, pose[1]+5))


            self.axs[2,1].set_title("Error")
            self.ax21sc = plt_add_elevation_map(self.axs[2,1], elevation, data_error, vmin = -0.5, vmax = 0.5, colorbar = self.need_color_bar)
            self.ax21ptch = plt_add_robot_patch(self.axs[2,1], pose)
            self.elems += self.ax21sc
            self.elems += self.ax21ptch
            self.axs[2,1].set_xlim((pose[0]-5, pose[0]+5))
            self.axs[2,1].set_ylim((pose[1]-5, pose[1]+5))

            self.axs[2,2].set_title("fusion Error")
            self.ax22sc = plt_add_elevation_map(self.axs[2,2], elevation, data_error_fusion, vmin = -0.5, vmax = 0.5, colorbar = self.need_color_bar)
            self.ax22ptch = plt_add_robot_patch(self.axs[2,2], pose)
            self.elems += self.ax22sc
            self.elems += self.ax22ptch
            self.axs[2,2].set_xlim((pose[0]-5, pose[0]+5))
            self.axs[2,2].set_ylim((pose[1]-5, pose[1]+5))

            self.need_color_bar = False
            return self.elems

        def update_plot(self, i):
            try:
                data = next(self.stream)
            except Exception as e:
                self.stream = self.data_stream()
                data = next(self.stream)
            pose = data["pose"]
            pc_points = data["pc_points"]
            pred_points = data["pred_points"]
            map_pc = data["map_pc"]
            map_pc_fusion = data["map_pc_fusion"]
            map_pred = data["map_pred"]
            map_pred_fusion = data["map_pred_fusion"]
            (x,y) = data["xy"]
            data_traj = data["data_traj"]
            data_error = data["data_error"]
            data_error_fusion = data["data_error_fusion"]

            self.ax00sc.set_offsets(pc_points[:,:2])
            self.ax00sc.set_array(pc_points[:,2])
            plt_set_robot_patch(*self.ax00ptch, pose)

            plt_set_elevation_map(*self.ax01sc, (x,y), map_pc)
            plt_set_robot_patch(*self.ax01ptch, pose)

            plt_set_elevation_map(*self.ax02sc, (x,y), map_pc_fusion)
            plt_set_robot_patch(*self.ax02ptch, pose)


            self.ax10sc.set_offsets(pred_points[:,:2])
            self.ax10sc.set_array(pred_points[:,2])
            plt_set_robot_patch(*self.ax10ptch, pose)

            plt_set_elevation_map(*self.ax11sc, (x,y), map_pred)
            plt_set_robot_patch(*self.ax11ptch, pose)
            
            plt_set_elevation_map(*self.ax12sc, (x,y), map_pred_fusion)
            plt_set_robot_patch(*self.ax12ptch, pose)
            

            plt_set_elevation_map(*self.ax20sc, (x,y), data_traj)
            plt_set_robot_patch(*self.ax20ptch, pose)

            plt_set_elevation_map(*self.ax21sc, (x,y), data_error)
            plt_set_robot_patch(*self.ax21ptch, pose)      

            plt_set_elevation_map(*self.ax22sc, (x,y), data_error_fusion)
            plt_set_robot_patch(*self.ax22ptch, pose)            


            [ax.set_xlim((pose[0]-5, pose[0]+5)) for ax in self.axs.reshape(-1)]
            [ax.set_ylim((pose[1]-5, pose[1]+5)) for ax in self.axs.reshape(-1)]

            return self.elems

        def data_stream(self):
            try:
                sample = next(self.data_iter)
            except Exception as e:
                print(e)
                self.data_iter = iter(self.data_loader)
                sample = next(self.data_iter)
            rawdatapath = sample["path"][0].replace("extract_trajectories_003_augment", "extract_trajectories_003/extract_trajectories")
            with open(rawdatapath, "rb") as data_file:
                byte_data = data_file.read()
                rawdata = msgpack.unpackb(byte_data)

            ## Prepare data
            pose = rawdata["pose"]["map"]
            print(pose[:3])
            pc_points = rawdata["pointcloud"][:,:3]
            pos = rawdata["pose"]["map"][:3].copy()
            elevation.move_to_and_input(pos, pc_points)
            
            elevation.reset()
            elevation.move_to_and_input(pos, pc_points)
            elevation_pc_fusion.move_to_and_input(pos, pc_points)
            map_pc = elevation.get_elevation_map()
            map_pc_fusion = elevation_pc_fusion.get_elevation_map()

            elevation.reset()
            pred = model(sample["image"].cuda())
            depth = sample["depth"][0][0].T
            pred = pred[0].detach()#.numpy()
            pred = nn.functional.interpolate(torch.tensor(pred).detach()[None,...], torch.tensor(depth.T).shape[-2:], mode='bilinear', align_corners=True)
            pred = pred[0][0].T
            
            pose = rawdata["pose"]["map"] # xyzw
            pts = camera.project_depth_to_cloud(torch.Tensor(pose), pred)

            height_mask = pts[:,2] < pose[2]
            pred_points = pts[height_mask].cpu()

            elevation.move_to_and_input(pos, pred_points)
            elevation_pred_fusion.move_to_and_input(pos, pred_points)
            map_pred = elevation.get_elevation_map()
            map_pred_fusion = elevation_pred_fusion.get_elevation_map()

            x,y = get_xy_from_elevation(elevation)
            data_traj = np.array([gft.getHeight(a,b, "GPMap")[0] for a,b in zip(x,y)])
            data_error = np.array(map_pred.reshape(-1) - data_traj)
            data_error_fusion = np.array(map_pred_fusion.reshape(-1) - data_traj)

            yield {
                "pose": pose,
                "pc_points": pc_points,
                "pred_points": pred_points,
                "map_pc": map_pc,
                "map_pc_fusion": map_pc_fusion,
                "map_pred": map_pred,
                "map_pred_fusion": map_pred_fusion,
                "xy": (x,y), 
                "data_traj": data_traj,
                "data_error": data_error,
                "data_error_fusion": data_error_fusion
            }


    animator = AnimateElevation(data_loader)

    ani = animation.FuncAnimation(
        animator.fig, func = animator.update_plot, frames = len(data_loader), 
        init_func=animator.setup_plot,
        interval=200, repeat_delay=1000, blit=True)
    # writervideo = animation.FFMpegWriter(fps=60)

    ani.save('visulization/elevation_animation.mp4')

    # plt.show()

