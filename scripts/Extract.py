# please change the root_dir, the path to extractFeetTrajsFromRosbag.py and the outdir, calibration path in data_extraction_SA.yaml

import os
root_dir = '/media/anqiao/Semantic/Data/Italy/day3/'

for date_file in os.listdir(root_dir):
    for mission_file in os.listdir(root_dir + '/' + date_file):
        if mission_file.split('_')[-1] == '2':
            bag_file = [root_dir + '/' + date_file + '/' + mission_file + '/' + bag for bag in os.listdir(root_dir+'/'+date_file+'/'+mission_file) if bag.split('_')[0]=='Reconstruct'][0]
            print(bag_file)
            # os.system('python /home/anqiao/tmp/semantic_front_end_filter/Labelling/extractFeetTrajsFromRosbag.py --bag_path=' + bag_file)
            os.system('python /home/anqiao/tmp/semantic_front_end_filter/Labelling/extractPointCloudFromRosbag.py --bag_path=' + bag_file)
