import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--file_path", default="/home/anqiao/semantic_front_end_filter/Labelling/Example_Files/")
parser.add_argument(
    "--checkpoints_dir", default="/home/anqiao/tmp/semantic_front_end_filter/adabins/checkpoints")
parser.add_argument(
    "--data_path", default="/media/anqiao/Semantic/Data/extract_trajectories_007_Italy_Anomaly_clean/extract_trajectories")
parser.add_argument(
    "--keyword", default="2023-03-05")

args = parser.parse_args()
file_path = args.file_path
checkpoints_dir = args.checkpoints_dir

with open(file_path + '/Evaluate_Table.csv', 'w+') as f:
    print('Creating file')
    # f.write(', Grassland, , , High_Grass, , , Forest, \n')
    f.write('GL_RMSE,GL_REL,GL_log10,HG_RMSE,HG_REL,HG_log10,F_RMSE,F_REL,F_log10,')
    f.write('GL_ASS,GL_AO,GL_SSS,HG_ASS,HG_AO,HG_SSS,F_ASS,F_AO,F_SSS\n')

key_word = '2023-03-05'
for checkpoint in os.listdir(checkpoints_dir):
    if key_word in checkpoint:
        os.system("python /home/anqiao/tmp/semantic_front_end_filter/semantic_front_end_filter/Labelling/evalIoU_cuda.py --models " + checkpoints_dir + '/' + checkpoint + "/UnetAdaptiveBins_latest.pt --save_path " + file_path + 'Evaluate_Table.csv --dataset_path '+args.data_path)