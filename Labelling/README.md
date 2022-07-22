# Dataset Gerneration (for Itatly dataset)

## Installation

### Depandencies

-[msgpack-c](https://github.com/msgpack/msgpack-c/tree/cpp_master) on branch cpp_master

-[anymal_docker](https://github.com/leggedrobotics/anymal_docker/tree/feature/21.11) on branch feature/21.11. When install this docker by instructions, you will also be asked to install [anymal_rsl](https://bitbucket.org/leggedrobotics/anymal_rsl/src/master/). Remember to install the corresponding dependencies for anymal_rsl.

### Add these rospackages to workspace
-[alphasense_rsl](https://bitbucket.org/leggedrobotics/alphasense_rsl/src/299f263b37f95b774eaec6afae4b9b7d905809cc/?at=feature%2Freorganization) on branch feature/reorganization

-Add the [grid_map](https://github.com/ANYbotics/grid_map) on branch master

-For Italy dataset, add rospackage `semantic_frontend_filter` from this repo.

## Dataset generation

1. Following the instruction in the next seesion to replay and re-record the rosbags.
2. Follow the commet in /Labelling/Extract.py to change some paths. Then run it. It will search the re-reorded bags in each mission and out put a dataset with msgpacks.  
   
## Replay Rosbag

1. Revise the configration file /semantic_frontend_filter/config/2022-07-08-09-05-19_anymal-cerberus-lpc_mission.yaml. Make sure the following values of parameters are correct.
   
  
   ```bash
         use_sim_time = true
         ...
         replay_basic_rsl:
         nodes:
           anymal_tf_publisher:
             enabled: false
             name: anymal_tf_publisher
           inspection_payload_robot_state_publisher:
             enabled: false
             name: inspection_payload_robot_state_publisher
           localization_manager:
             enabled: false
             name: localization_manager
   ```



2. Rvise the launch file `catkin_ws/src/alphasense_rsl/image_proc_cuda_ros/launch/image_proc_cuda_node.launch`. Also change the calibration_path for each camera.
   ```xml
   <arg name="run_undistortion"                     default="true"/>

   <arg name="input_topic3"                         default="alphasense_driver_ros/cam3"/>
   <arg name="transport3"                           default="raw"/> 
   <arg name="output_topic3"                        default="alphasense_driver_ros/cam3/debayered"/>
   <arg name="needs_rotation_cam3"                  default="true"/>
   <arg name="calibration_cam3"                     default="$(find image_proc_cuda)/config/alphasense_calib_example.yaml"/>
   <arg name="color_calibration_cam3"               default="$(find image_proc_cuda)/config/alphasense_color_calib_example.yaml"/>

   <arg name="input_topic4"                         default="alphasense_driver_ros/cam4"/>
   <arg name="transport4"                           default="raw"/> 
   <arg name="output_topic4"                        default="alphasense_driver_ros/cam4/debayered"/>
   <arg name="needs_rotation_cam4"                  default="true"/>
   <arg name="calibration_cam4"                     default="$(find image_proc_cuda)/config/alphasense_calib_example.yaml"/>
   <arg name="color_calibration_cam4"               default="$(find image_proc_cuda)/config/alphasense_color_calib_example.yaml"/>

   <arg name="input_topic5"                         default="alphasense_driver_ros/cam5"/>
   <arg name="transport5"                           default="raw"/> 
   <arg name="output_topic5"                        default="alphasense_driver_ros/cam5/debayered"/>
   <arg name="needs_rotation_cam5"                  default="false"/>
   <arg name="calibration_cam5"                     default="$(find image_proc_cuda)/config/alphasense_calib_example.yaml"/>
   <arg name="color_calibration_cam5"               default="$(find image_proc_cuda)/config/alphasense_color_calib_example.yaml"/>
   ```

3. To replay and re-record the rosbags use
   ```bash
   roslaunch semantic_frontend_filter replay_and_record_Italy.launch bagfile:="'1.bag' '2.bag' ..." output_file=:" "

   ```
4. To replay with the reconstructed map use 
   ```bash
   roslaunch semantic_frontend_filter replay_with_map_Italy.launch bagfile:="'1.bag' '2.bag' ..." map_file:="GroundMap.msgpack"
   ```


## Use of Grid Map
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

## Known issuses

For image undistortion you need to acces cuda within docker anymal-gpu:21.11. If cannot, add `--rm --gpus all` to `RUN_COMMAND` in `anymal_docker/anymal/bin/run.sh`.

```bash
RUN_COMMAND="docker run \
  --rm --gpus all \
  --volume=$XSOCK:$XSOCK:rw \
  ...
  ...
```