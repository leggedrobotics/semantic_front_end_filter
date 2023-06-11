import os 
import semantic_front_end_filter
from semantic_front_end_filter.utils.labelling.GroundfromTrajs import saveLocalMaps


spf_path = semantic_front_end_filter.__path__[0]
bag_path = "/media/anqiao/ycyBigDrive/Data/Italy/reconstruct_Italy/Reconstruct_2022-07-18-20-34-01_0.bag"
out_dir = "/media/anqiao/ycyBigDrive/Data/extract_trajectories_007_Italy_test/extract_trajectories"
tmp_out_dir = out_dir + "_tmp"
bag_name = bag_path.split('/')[-1].split('.')[0]
# # Extract feet trajectories
# os.system("python "+spf_path+"/utils/labelling/extractFeetTrajsFromRosbag.py --bag_path "+bag_path+" --cfg_path "+spf_path+"/cfgs/data_extraction_SA.yaml --outdir "+tmp_out_dir)

# Build local Gaussian maps
# feet_trajs_path = outdir+'/'+bag_name+'/FeetTrajs.msgpack'
# saveLocalMaps(feet_traj = feet_trajs_path, save_path = feet_trajs_path.rsplit('/', 1)[0])

# Build msgpacks for training
# os.system("python "+spf_path+"/utils/labelling/extractPointCloudFromRosbag.py --bag_path "+bag_path+" --cfg_path "+spf_path+"/cfgs/data_extraction_SA.yaml --out_dir "+tmp_out_dir)

#################################################
# Before proceeding to the next step, please do the same operation to all the rosbags you want to process in the tmp_out_dir
#################################################

# Add self-supervised depth estimation label
print("python "+spf_path+"/utils/labelling/ExtractLocalDepthImage.py --mpas_path "+tmp_out_dir+"/"+bag_name+"/localGroundMaps/ --cfg_path "+spf_path+"/cfgs/data_extraction_SA.yaml --source_path "+tmp_out_dir+" --target_path "+out_dir)
os.system("python "+spf_path+"/utils/labelling/ExtractLocalDepthImage.py --maps_path "+tmp_out_dir+"/"+bag_name+"/localGroundMaps/ --cfg_path "+spf_path+"/cfgs/data_extraction_SA.yaml --source_path "+tmp_out_dir+" --target_path "+out_dir)

# Finally, label the data as you want.
# We provide our dataset on 