# please change the root_dir, the path to extractFeetTrajsFromRosbag.py and the outdir, calibration path in data_extraction_SA.yaml

# For Italy
# import os
# root_dir = '/media/anqiao/Semantic/Data/Italy/day3/'
# outdir='/media/anqiao/Semantic/Data/extract_trajectories_006_Italy/extract_trajectories/'

# for date_file in os.listdir(root_dir):
#     if("mission" in date_file):
#         for mission_file in os.listdir(root_dir + '/' + date_file):
#             if mission_file.split('_')[-1] == '2':
#                 bag_file = [root_dir + '/' + date_file + '/' + mission_file + '/' + bag for bag in os.listdir(root_dir+'/'+date_file+'/'+mission_file) if bag.split('_')[0]=='Reconstruct'][0]
#                 print(bag_file)
#                 # os.system('python /home/anqiao/tmp/semantic_front_end_filter/Labelling/extractFeetTrajsFromRosbag.py --bag_path=' + bag_file)
#                 os.system('python /home/anqiao/tmp/Extract/semantic_front_end_filter/Labelling/extractPointCloudFromRosbag.py --bag_path=' + bag_file + ' --out_dir=' + outdir)

# # For South Africa
import os
root_dir = '/media/anqiao/Semantic/Data/20211007_SA_Monkey_ANYmal_Chimera'
outdir='/media/anqiao/Semantic/Data/extract_trajectories_006_SA/extract_trajectories/'

for date_file in os.listdir(root_dir):
    if("chimera" in date_file):
        for mission_file in os.listdir(root_dir + '/' + date_file):
            if mission_file.split('_')[-1] in ['locomotion', 'locomotino']:
                bag_file = [root_dir + '/' + date_file + '/' + mission_file + '/' + bag for bag in os.listdir(root_dir+'/'+date_file+'/'+mission_file) if bag.split('_')[-1]=='0.bag'][0]
                os.system('python /home/anqiao/tmp/Extract/semantic_front_end_filter/Labelling/extractPointCloudFromRosbag.py --bag_path=' + bag_file + ' --out_dir=' + outdir)

# For Zurich
# import os
# root_dir = '/media/anqiao/Semantic/Data/20220810_ANYmal_Cerberus_Hoeng_High_Grass_Bumping/Reconstruct/'
# outdir='/media/anqiao/Semantic/Data/extract_trajectories_006_Zurich/extract_trajectories/'

# for bag_file in os.listdir(root_dir):
#     os.system('python /home/anqiao/tmp/Extract/semantic_front_end_filter/Labelling/extractPointCloudFromRosbag.py --bag_path=' + root_dir + '/' + bag_file + ' --out_dir=' + outdir)
