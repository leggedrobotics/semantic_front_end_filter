# Install

### create and activate conda environment

```bash
conda env create -f environment.yaml
conda activate spf_venv
```

### SPF

```bash
cd semantic_front_end_filter
pip install -e .
```

# semantic_front_end_filter
**notice**: In this branch, the skip connection is removed, since the consistency is required when we predict the depth of teh whole picture instaed of only some points. And also for convinent the preparation of the image like cropping and flipping is commented. Because after cropping the calibration of camera is changed and the prejection will be very complex. Also since the view of robot is our task is nearly the same, so maybe its not that necerssry to do these pre process
## Setup

TODOTODO.....

4. install the semantic_front_end_filter package

```bash
### in your working env
### In the root of semantic_front_end_filter package
pip3 install -e .
```


## Running rosnode
The model can be downloaded [here](https://drive.google.com/drive/folders/1fSOTvCp4IA6ey2NAhyHrBxCyJ9_pRd1N). Please remember to download the whole folder.

After setting up the repo, change the model_path, image_topic in semantic_front_end_filter_ros/scripts/deploy_foward.py. Then you can simply run this file as a ros node, listening to camera image and pointclouds, and publishing filtered pointclouds.

The reconstructed validation trajectory that we commonly use is [here](https://drive.google.com/drive/folders/1m1XzdB_q6GBZjpP_csMFxQ3IIILvXtjO?usp=sharing). You can directly play the bag and it will contain necessary pointcloud, image, and tf messages.



## Train

```bash
python3 -m semantic_front_end_filter.adabins.train
```

> consult semantic_front_end_filter/adabins/cfg.py for the command line configurations.


## Rviz Replay

First get the trajectories replayed

```bash
# semantic_front_end_filter/bin
bash launch_replayandrecord.sh
```

Then Run the following two scripts with roscore running

```bash
# In two seperated terminals
rosrun semantic_front_end_filter_ros rosvis.py
rosrun semantic_front_end_filter_ros rosvis_replay.py
```

### rosvis_replay.py 

This file filters the rostopics in replay stack to the topics required by rosviz.py

- front camera
  - subscribe: `/alphasense_driver_ros/cam4/dropped/debayered/compressed`
      - (in forest dataset might be `/alphasense_driver_ros/cam4/dropped/debayered/slow`)
  - broadcast: `/semantic_filter_image`
- point cloud
  - subscribe: `/bpearl_front/point_cloud`
  - broadcast: `/semantic_filter_points`
- pose (pose in map frame)
  - subscribe: `/tf`
  - broadcast: `/semantic_filter_pose`

### rosvis.py

This file loads a adabin model, listen on the topics of images, pose, depth, and point cloud. Broadcast the result of the model as pointclouds

#### subscribe

- `semantic_filter_pose`:  `Float64MultiArray`
    -  pose of the robot x,y,z,rx,ry,rz,rw
- `semantic_filter_depth`:  `Float64MultiArray`
    - An ground truth depth array of the shape like 
- `semantic_filter_image`:  `Float64MultiArray`
    - An ground image array of the shape like [3 540 720]
- `semantic_filter_points`:  `Float64MultiArray`
    -  A point cloud array of the shape [N 3], where N is the number of points

#### Broadcast

- `terrain_pc`:  `PointCloud2`
    - pointcloud of the ground truth depth label
- `pred_pc`:  `PointCloud2`
    - pointcloud of the prediction
- `pointcloud`:  `PointCloud2`
    - pointcloud of the received pointcloud

