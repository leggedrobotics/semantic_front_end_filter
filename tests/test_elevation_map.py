# test to generate an elevation map from contact trajectories
def test_elevation_map():
    import sys,os
    from semantic_front_end_filter import SEMANTIC_FRONT_END_FILTER_ROOT_PATH
    from semantic_front_end_filter.adabins.elevation_vis import WorldViewElevationMap
    from semantic_front_end_filter.Labelling.GroundfromTrajs import GFT
    from elevation_mapping_cupy.parameter import Parameter
    import numpy as np

    ## Generate the input for elevationmap_cupy from FeetTrajs
    target_pos = np.array([130,425])

    FeetTrajs_filepath = os.path.join(SEMANTIC_FRONT_END_FILTER_ROOT_PATH, "Labelling/Example_Files/FeetTrajs.msgpack")
    gft = GFT(FeetTrajsFile = FeetTrajs_filepath, InitializeGP=False)
    foot_holds = {k : np.array(gft.getContactPoints(v)[0]) for k,v in gft.FeetTrajs.items()} # A dict of the contact points of each foot
    foot_holds_array = np.vstack(list(foot_holds.values()))
    foot_holds_array = foot_holds_array[np.sum((foot_holds_array[:,:2] - target_pos)**2, axis = 1)<10**2]


    ## use elevation_map to fuse contact points, "init_with_initialize_map" can be None, "nearest", "linear", "cubic")
    for init_method in [False, "nearest", "linear", "cubic"]:
        elevation = WorldViewElevationMap(resolution = 0.1, map_length = 10, init_with_initialize_map = init_method)
        elevation.reset()
        elevation.move_to_and_input([*target_pos,-4.46], foot_holds_array)
        elev_map_foot_holds = elevation.get_elevation_map()
        print("Number of not_nan points for ", init_method, (~np.isnan(elev_map_foot_holds)).sum())
        assert( not (np.isnan(elev_map_foot_holds)).all())
