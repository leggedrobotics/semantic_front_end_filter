import msgpack
import msgpack_numpy as m
import numpy as np
from pendulum import time
from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_from_matrix, quaternion_from_matrix
from ruamel.yaml import YAML
from ExtractDepthImage import DIFG
import os
import matplotlib.pyplot as plt
from messages.imageMessage import Camera

m.patch()

def loadmap(path, i):
    if not os.path.exists(path+'/localGroundMap_{:03d}.msgpack'.format(i)):
        map_data = {}
    else:
        with open (path+'/localGroundMap_{:03d}.msgpack'.format(i), 'rb') as data_file:
            map_data = data_file.read()
            map_data = msgpack.unpackb(map_data)
    return map_data

cfg_path = '/home/anqiao/tmp/semantic_front_end_filter/semantic_front_end_filter/Labelling/data_extraction_SA.yaml'
maps_path = '/home/anqiao/catkin_ws/SA_dataset/mountpoint/Data/extract_trajectories_006_Italy/extract_trajectories/Reconstruct_2022-07-18-20-34-01_0/localGroundMaps/'
# map1_path = '/home/anqiao/catkin_ws/SA_dataset/mountpoint/Data/extract_trajectories_006_Italy/extract_trajectories/Reconstruct_2022-07-18-20-34-01_0/localGroundMaps/localGroundMap_000.msgpack'
calibration_path = "/home/anqiao/tmp/semantic_front_end_filter/anymal_c_subt_semantic_front_end_filter/config/calibrations/alphasense"
source_path = '/home/anqiao/catkin_ws/SA_dataset/mountpoint/Data/extract_trajectories_006_Italy_Anomaly/extract_trajectories/'
target_path = '/home/anqiao/catkin_ws/SA_dataset/mountpoint/Data/extract_trajectories_007_Italy/extract_trajectories/'

cfg = YAML().load(open(cfg_path, 'r'))

camera = Camera(calibration_path, 'cam4', cfg)
camera.tf_base_to_sensor = (np.array([-0.407, -0.004, 0.231]), 
                            np.array([[-9.99654060e-01,  2.59955028e-02,  3.99930812e-03,  0.        ],
                                    [ -2.59955028e-02, -9.99662058e-01,  5.19910056e-05,  0.        ],
                                    [ 3.99930812e-03, -5.19910056e-05,  9.99992001e-01,  0.        ],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]]))

# currentMapIndex = 0
# currentMap = loadmap(maps_path, currentMapIndex)
# nextMap = loadmap(maps_path, currentMapIndex+1)
# depth_img_cam = DIFG(maps_path+'/localGroundMap_{:03d}.msgpack'.format(currentMapIndex), calibration_path, 'cam4', cfg)
# nextMapTimespan = nextMap['timespan']

for re in os.listdir(source_path):
    print(re)
    maps_path = target_path + '/'+ re + '/localGroundMaps/'
    currentMapIndex = 0
    IsLastMap = False
    currentMap = loadmap(maps_path, currentMapIndex)
    nextMap = loadmap(maps_path, currentMapIndex+1)
    depth_img_cam = DIFG(maps_path+'/localGroundMap_{:03d}.msgpack'.format(currentMapIndex), calibration_path, 'cam4', cfg)
    nextMapTimespan = nextMap['timespan']
    for file in sorted(os.listdir(source_path+'/'+re), key=lambda x : (int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0]))):
        print(file)
        with open (source_path+'/'+re + '/' +file , 'rb') as data_file:
            datumData = data_file.read()
            datumData = msgpack.unpackb(datumData)
            
        euler = euler_from_quaternion(datumData['pose'][3:])
        position = datumData['pose'][:3]
        timestamp = datumData['time']
        if (timestamp>nextMap['timespan'][0] and IsLastMap == False):
            print("Changing to next map.")
            # Update current and next map
            currentMap = nextMap
            currentMapIndex += 1
            nextMap = loadmap(maps_path, currentMapIndex+1)
            if len(nextMap)==0:
                nextMap = currentMap
                currentMapIndex -= 1
                IsLastMap = True
                print("Reach the last map!")
            nextMapTimespan = nextMap['timespan']
            # Update camera 
            depth_img_cam = DIFG(maps_path+'/localGroundMap_{:03d}.msgpack'.format(currentMapIndex), calibration_path, 'cam4', cfg)
            
        camera.update_pose_from_base_pose(datumData['pose'])
        d_img, v_img = depth_img_cam.getDImage(transition=camera.pose[0], rotation=camera.pose[1][:3, :3], rotation_is_matrix=True)
        datumData['depth_var'] = np.concatenate([d_img[None, ...], v_img[None, ...]], axis = 0)

        with open(target_path + '/' + re + '/' +file, "wb") as outfile:
            file_dat = msgpack.packb(datumData)
            outfile.write(file_dat)
    # plt.figure()
    # plt.imshow(d_img)
    # plt.show()