# semantic_front_end_filter

## Setup

TODOTODO.....

4. install the semantic_front_end_filter package

```bash
### in your working env
### In the root of semantic_front_end_filter package
pip3 install -e .
```


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

