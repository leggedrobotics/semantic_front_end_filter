# Construct Map from Robot Feet Trajectories

## Installation

### Depandencies
Visualize session depends on `msgpack-c`.

-Install the [msgpack-c](https://github.com/msgpack/msgpack-c/tree/cpp_master) follow this link https://github.com/msgpack/msgpack-c/tree/cpp_master.

### Add rospackages to workspace

-Add the [grid_map](https://github.com/ANYbotics/grid_map) packages to your catkin workspace follow this link https://github.com/ANYbotics/grid_map.


-Add rospackage `anymal_c_subt_semantic_front_end_filter` in this repo to your catkin worksapce.

## Dataset generation

1. Use `replayandrecord.launch` to get the re-record rosbag.
2. Run `python3 extractFeetTrajsFromRosbag.py`
3. Run `python3 extractPointCloudFromRosbag.py`. 
   1. It automatically calls `GroundfromTrajs.py` `ExtractDepthImage.py`
   2. It outputs a `msgpack` for each tuple of point clould, images, depth image, pos, and other information.


## Usage & Examples
### Get grid map from a rosbag

1. Use replay_and_record.launch to get the re-record rosbag. 
   
   ```bash
   roslaunch anymal_c_subt_semantic_front_end_filter replay_and_record.launch bagfile:='...' dumped_rosparameters:='...' path_map_darpa_tf_file:=~/SADocker/anymal_docker/anymal-darpa/map_darpa_transformation.txt robot:=chimera
   ```

2. Fill the data_extraction_SA.yaml with the re-recorded rosbag path and output path. Then run the extractFeetTrajsFromRosbag.py to get the file FeetTrajs.msgpack. This File contains a dict of four feet trajcetories. 
   
    ```bash
   python extractFeetTrajsFromRosbag.py
   ```

3. Use getGroundFromTrajs.py to generate the a grid map and save it in a file named GroundMap.msgpack. This file contains the basice information of the reconstructed grid map, like shape, height and confidence of the height.

    ```bash
   python getGroundFromTrajs.py
   ```

4. Use replay_with_map.launch to visualize the gird_map and robot in rviz.
   ```bash
   roslaunch anymal_c_subt_semantic_front_end_filter replay_with_map.launch bagfile:='...' dumped_rosparameters:='...' path_map_darpa_tf_file:=~/SADocker/anymal_docker/anymal-darpa/map_darpa_transformation.txt robot:=chimera map_file:='...'
   ```

### Get Point Cloud from a rosbag

Similar to `extractFeetTrajsFromRosbag.py`, run `python3 extractPointCloudFromRosbag.py`. It outputs a `msgpack` for each tuple of point clould, images, pos, and etc information.

### Use of Grid Map
GetGroundfromTrajs.py provides a class **GFT**, which allows you to generate a grid map from FeetTrajs.msgpack or load a grid map from GroundMap.msgpack. 

You can instantiate class **GFT** with one and only one of these two files. Then you can use multi API to access or visualize the grid map. For example, 

```python
# Get a Grid Map by Gaussian Process. 
# Since the Gaussian Process will consume some time(about 30s), if you only want to use a sparse grid map, you can also set InitializeGP = False and fit with Gaussian Process late by GFT::initializeGPMap().
gft = GFT(FeetTrajsFile='./Examples/FeetTrajs.msgpack', InitializeGP = True)
gft.save('./Examples/', GPMap=True)

# Load Grid Map File
gftload = GFT(GroundMapFile='./Examples/GroundMap.msgpack')

# Get Height
print(gftload.getHeight(0, 0))

xlist = np.zeros(3)
ylist = np.zeros(3)
print(gftload.getHeight(xlist, ylist))

# Get the whole map and confidence
GPMap = gftload.getGPMap()

# Visualize
gftload.visualizeGPMap()

```

## Definations

**Sparse Grid Map**: A grid map constructed only by the filling the grid map with foot contacts.

**Gaussian Process Map**: A grid map constructed by fitting Gaussian Process on Sparse Grid Map.