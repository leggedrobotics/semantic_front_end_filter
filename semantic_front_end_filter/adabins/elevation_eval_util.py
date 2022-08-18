"""
Chenyu 
Utilities for the evaluation of elevation map.
"""

import msgpack
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import rotate

class ElevationMapEvaluator:
    """
    This class holds the information of a ground truth, 
    compute error given an elevation map,
    and count the mean and variance of elevation map errors
    """
    def __init__(self, ground_map_path, elev_map_param):
        """
        arg ground_map_path:    The path to the ground map msgpack, 
                generated from Labelling/GroundfromTrajs.py
        arg elev_map_param:     The parameter dataclass for the elevation map
                for example elevation_mapping_cupy.parameter.Parameter
        """
        with open(ground_map_path, "rb") as data_file:
            data = data_file.read()
            ground_dict = msgpack.unpackb(data)
            print("load ground dict, y real range: ",ground_dict["yRealRange"], 
                                    "x real range: ", ground_dict["xRealRange"])
            self.ground_dict = ground_dict
            self.gpmap = np.array(ground_dict["GPMap"])
        # Interpolate function for self.gpmap, reference: semantic_front_end_filter/Labelling/ExtractDepthImage.py
        center_point = np.array([ground_dict["xRealRange"][0],ground_dict["yRealRange"][0],0]) # this should be a misnomer, it is the upper left corner
        # some side notes about RealRange and res: The (RealRange[1]-RealRange[0])/res and gpmap size do not match
        #   The reason of this problem is the floor operation "math.floor(ContactArray[:, 0].min())" before dividing res
        #   However, this is not a big issue, as the RealRange[0] and gparray[0][0] is matched in construct and use
        #   Just ignore the RealRange[1]
        x = np.arange(self.gpmap.shape[0]) * ground_dict["res"] + center_point[0]
        y = np.arange(self.gpmap.shape[1]) * ground_dict["res"] + center_point[1]
        self.gpmap_interp = RegularGridInterpolator((x, y), self.gpmap, bounds_error = False)   

        self.elev_map_param = elev_map_param
        self.reset_error_count()


    def reset_error_count(self):
        """
        Init(reset) Data arrays for the counted error
        """
        # Some parameters for the elevation map
        self.map_length = self.elev_map_param.map_length
        self.resolution = self.elev_map_param.resolution
        # this cell_n is sticked to the defn in elevation_cupy
        self.cell_n = int(round(self.map_length / self.resolution)) + 2

        self.error_sum = np.zeros([self.cell_n-2, self.cell_n-2], dtype=np.float32)
        self.error_count = np.zeros([self.cell_n-2, self.cell_n-2], dtype = int)
        self.error_list = [] # record all errors. This might be too memory expensive

    
    def get_gpmap_at_xy(self, xy):
        """translate and crop the gpmap to get a elev_map of the size (cell_n-2) x (cell_n-2)
        arg xy, r: the transition and z-axis angle of the robot
        """
        # Get the gpmap in the elevmap frame(transition x,y, no rotation)
        x = ((np.arange(self.cell_n-1,1,-1)+0.5) - 0.5 * self.cell_n )*self.resolution + xy[0]
        y = ((np.arange(self.cell_n-1,1,-1)+0.5) - 0.5 * self.cell_n )*self.resolution + xy[1]
        x,y = np.meshgrid(x,y, indexing="ij")
        x,y = x.reshape(-1), y.reshape(-1)
        xypoints = np.vstack((x,y)).T
        elevmap_gt = self.gpmap_interp(xypoints).reshape(self.cell_n-2, self.cell_n-2)
        return elevmap_gt

    def compute_error_against_gpmap(self, elevmap, xy, r=0):
        """compute the rmse error between elevmap and self.gpmap, record the error
            The returned and recorded error will be rotated into the robot frame, by r. r can be set_to 0 if not intended to rotate it
        arg elevmap: 2d np array, the elevation map around the robot, oriented in the world frame
        arg xy, r: the transition and z-axis angle of the robot, r is in rad
        """
        
        elevmap_gt = self.get_gpmap_at_xy(xy)
        error = elevmap - elevmap_gt

        # update the error_sum error count
        mask = ~np.isnan(error)
        error = rotate(error, angle=-r/np.pi*180, reshape=False, order = 0, mode='constant', cval = np.nan)
        self.error_sum[mask] += abs(error[mask])
        self.error_count[mask] += 1
        self.error_list.append(error)

        return error

        
if __name__ == "__main__":

    # test with a naive elevation baseline: elevation with foot contact trajs
    import matplotlib.pyplot as plt
    
    import sys,os
    from semantic_front_end_filter import SEMANTIC_FRONT_END_FILTER_ROOT_PATH
    from semantic_front_end_filter.adabins.elevation_vis import WorldViewElevationMap
    from semantic_front_end_filter.Labelling.GroundfromTrajs import GFT

    from elevation_mapping_cupy.parameter import Parameter

    ## Generate the input for elevationmap_cupy from FeetTrajs
    target_pos = np.array([130,425])

    FeetTrajs_filepath = os.path.join(SEMANTIC_FRONT_END_FILTER_ROOT_PATH, "Labelling/Example_Files/FeetTrajs.msgpack")
    gft = GFT(FeetTrajsFile = FeetTrajs_filepath, InitializeGP=False)
    foot_holds = {k : np.array(gft.getContactPoints(v)[0]) for k,v in gft.FeetTrajs.items()} # A dict of the contact points of each foot
    foot_holds_array = np.vstack(foot_holds.values())
    foot_holds_array = foot_holds_array[np.sum((foot_holds_array[:,:2] - target_pos)**2, axis = 1)<10**2]
    print("foot_holds_array shape:", foot_holds_array.shape)

    ## use elevation_map to fuse contact points, "init_with_initialize_map" can be None, "nearest", "linear", "cubic")
    elevation = WorldViewElevationMap(resolution = 0.1, map_length = 10, init_with_initialize_map = None)
    elevation.reset()
    elevation.move_to_and_input([*target_pos,-4.46], foot_holds_array)
    elev_map_foot_holds = elevation.get_elevation_map()
    print("EMPYT foot hold elev map?",np.isnan(elev_map_foot_holds).all())

    ## Initiate the elevation map evaluater
    ground_map_path = os.path.join(SEMANTIC_FRONT_END_FILTER_ROOT_PATH, "Labelling/Example_Files/GroundMap.msgpack")
    param = elevation.param # use the parameter same as the elevation_map
    evaluator = ElevationMapEvaluator(ground_map_path, param)

    #compute error
    error = evaluator.compute_error_against_gpmap(elev_map_foot_holds, target_pos, np.pi/6)
    ## If not to rotate, pass the robot angle argument as r
    # error = evaluator.compute_error_against_gpmap(elev_map_foot_holds, target_pos)

    elev_gt = evaluator.get_gpmap_at_xy(target_pos)
    print(abs(error)[~np.isnan(error)].mean())
    print(elev_gt[~np.isnan(elev_gt)].mean())
    print(elev_map_foot_holds[~np.isnan(elev_map_foot_holds)].mean())
    fig, axs = plt.subplots(1,2,figsize=(20,10))
    axs[0].imshow(elev_map_foot_holds,vmin = -5, vmax = -4)
    axs[1].imshow(error, vmin = -0.5, vmax = 0.5)
    plt.show()

    # # Then checkout the error summmary values for later use
    # evaluator.error_count
    # evaluator.error_sum
    # evaluator.error_list