# Semantic Pointcloud Filter
[[Project page]](https://sites.google.com/leggedrobotics.com/semantic-pointcloud-filter?usp=sharing)
[[Paper]](https://arxiv.org/abs/2305.07995)
[[Data]](https://drive.google.com/file/d/1MjcNVJ2iwSdw3h5Z6QahUG4deMm8cr8H/view?usp=sharing)
[[Video]](https://youtu.be/y56zwSrnJTg)
## Installation

### Dependencies
create and activate conda environment

```bash
conda env create -f environment.yaml
conda activate spf_venv
```
To visulize the grid map created by Gaussian Process, you also need to install the [msgpack-c](https://github.com/msgpack/msgpack-c/tree/cpp_master). And add the [grid_map](https://github.com/ANYbotics/grid_map) packages into your catkin workspace. 
### SPF

```bash
cd semantic_front_end_filter
pip install -e .
```
To generate your own dataset, add rospackage `semantic_front_end_filter_ros` in this repo to your catkin worksapce. 

## Getting started
### Dataset Generation
1. Use `replay_and_record.launch` to get the re-record rosbag.
2. Follow the comments in `scripts/build_dataset.py` to generate the dataset.  
You can also download the data from [here](https://drive.google.com/drive/folders/1tRlrYeos8YdGmtDGacB-2Bt_fNqFKyHx), which we build on data collected from Perugia, Italy.

For the details of how to build a map from the robot feet trajectories, see [here](https://github.com/leggedrobotics/semantic_front_end_filter/tree/main/semantic_front_end_filter/utils/labelling).
### Train model 

```bash
python semantic_front_end_filter/scripts/train.py --data_path <path-to-data-folder>
```
To validate the trained model, run
```bash
python semantic_front_end_filter/scripts/eval.py --model <path-to-model-folder> --outdir <path-to-save-the-eveluation-plot> --data_path <path-to-data-folder>
```
<!-- **notice**: In this branch, the skip connection is removed, since the consistency is required when we predict the depth of teh whole picture instaed of only some points. And also for convinent the preparation of the image like cropping and flipping is commented. Because after cropping the calibration of camera is changed and the prejection will be very complex. Also since the view of robot is our task is nearly the same, so maybe its not that necerssry to do these pre process -->

## Running rosnode
Our trained model can be downloaded [here](https://drive.google.com/drive/folders/1Lx5QfLrfS0vk_88-UAJolm3D_ovZh5wS). Please remember to download the whole folder.

After setting up the repo, change the model_path in semantic_front_end_filter_ros/scripts/deploy_foward.py. Then you can simply run this file as a ros node, listening to camera image and pointclouds, and publishing filtered pointclouds.

The reconstructed validation trajectory that we commonly use is [here](https://drive.google.com/drive/folders/1m1XzdB_q6GBZjpP_csMFxQ3IIILvXtjO?usp=sharing). You can directly play the bag and it will contain necessary pointcloud, image, and tf messages.





