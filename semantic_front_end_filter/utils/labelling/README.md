# Construct Map from Robot Feet Trajectories

## Installation

### Depandencies
Visualize session depends on `msgpack-c`.

-Install the [msgpack-c](https://github.com/msgpack/msgpack-c/tree/cpp_master) follow this link https://github.com/msgpack/msgpack-c/tree/cpp_master.

### Add rospackages to workspace

-Add the [grid_map](https://github.com/ANYbotics/grid_map) packages to your catkin workspace follow this link https://github.com/ANYbotics/grid_map.


-Add rospackage `anymal_c_subt_semantic_front_end_filter` in this repo to your catkin worksapce.

## Map generation

1. Use `replayandrecord.launch` to get the re-record rosbag.
2. Run `python3 extractFeetTrajsFromRosbag.py`
3. Run `python3 GroundfromTrajs.py`. 


## Usage & Examples
### Get grid map from a rosbag

1. Use replay_and_record.launch to get the re-record rosbag. 
   
   ```bash
   roslaunch semantic_front_end_filter_ros replay_and_record.launch bagfile:='...' output_file:='...' 
   ```

2. Fill the data_extraction_SA.yaml with the re-recorded rosbag path and output path. Then run the extractFeetTrajsFromRosbag.py to get the file FeetTrajs.msgpack. This File contains a dict of four feet trajcetories. 
   
    ```bash
   python extractFeetTrajsFromRosbag.py
   ```

3. Use getGroundFromTrajs.py to generate local grid maps and save them in the folder localGroundMaps. Each msgpack in this folder contains the basice information of the locally reconstructed grid map, like shape, height and confidence of the height.

    ```bash
   python getGroundFromTrajs.py
   ```

4. Use replay_with_map.launch to visualize the gird_map and robot in rviz.
   ```bash
   roslaunch semantic_front_end_filter_ros replay_with_map.launch bagfile:='...' maps_path:='...' map_num:="..."
   ```


### Use of Grid Map
GetGroundfromTrajs.py provides a class **GFT**, which allows you to generate a grid map from FeetTrajs.msgpack or load a grid map from GroundMap.msgpack. 

You can instantiate class **GFT** with one and only one of these two files. Then you can use multi API to access or visualize the grid map. For example, 

```python
## Get a Grid Map by Gaussian Process. 
# Since the Gaussian Process will consume some time(about 30s), if you only want to use a sparse grid map, you can also set InitializeGP = False and fit with Gaussian Process later by GPT::initializeGPMap().
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

## To avoid the drift from long distance travel, you can choose only build local maps based on local footholds, which is also used in this project.

saveLocalMaps(feet_traj='./Examples/FeetTrajs.msgpack', save_path = './Examples/')
```

## Definations

**Sparse Grid Map**: A grid map constructed only by the filling the grid map with foot contacts.

**Gaussian Process Map**: A grid map constructed by fitting Gaussian Process on Sparse Grid Map.