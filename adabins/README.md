# AdaBins

> Migrated from [shariqfarooq123/AdaBins](https://github.com/shariqfarooq123/AdaBins.git)

## Training on euler

### Dataset

Dataset is uploaded on `/cluster/work/riner/users/PLR-2022/semantic_frontend_filter/extract_trajectories.tar`

### Environment

The code can run in euler official installation of python, plus a local package msgpack_numpy

**Load Environment Module**

```bash
# on euler: Execute this every time logging in
env2lmod
module load gcc/8.2.0 python/3.8.5
```

**Install msgpack_numpy**

```bash
# After loading python/3.8.5 on euler
python3 -m pip install --user msgpack-numpy
python3 -m pip install --user geffnet
python3 -m pip install --user simple-parsing
```

this will install the package `msgpack-numpy` to your `$HOME/.local/lib/python3.8/site-packages` 

### Run jobs

**script for copying dataset to $TMPDIR**

Save the following script into a file "run_train_adabins.sh"

```bash
#!/usr/bin/env bash
tar -xf extract_trajectories.tar -C $TMPDIR
cd semantic_front_end_filter/adabins/
python3 train.py  "$@"
```

> The `"$@"` means passing the command arguments calling the bash inside.



**submit jobs onto cluster**

```bash
bsub -n 32 \
-R "rusage[mem=1000,ngpus_excl_p=1]" \
-W 04:00 \
-R "select[gpu_mtotal0>=9000]" \
-R "rusage[scratch=200]" \
-R "select[gpu_driver>=470]" \
bash ./run_train_adabins.sh "$@"
```

- `-W` specifies the time
- `-n` specifies the number of cores
- `mem=xxx` the total memory is `n*mem`
- `scratch=200` is the space in `$TMPDIR`


### Command line arguments

Please have a look at `cfg.py` to see what command lines are avaliable

Some command line that often used are:
- "bs": batch size
- "validate_every"
- "pc_image_label_W"
- "traj_label_W"
- "normalize_output_mean"
- "normalize_output_std"

## Evaluation on local Machine

1. scp checkpoints from euler to local machine

### Visulize input and output of the model

Run `vismodel.py`, a simple script to load a data from test loader, run it through model and plot.

## Rviz Replay

Run the following two scripts with roscore running

```bash
python3 rosvis_replay.py 
python3 rosvis.py
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


