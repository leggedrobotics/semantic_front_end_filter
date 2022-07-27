cd ../anymal_c_subt_semantic_front_end_filter
export ROS_PACKAGE_PATH=$(pwd):$ROS_PACKAGE_PATH


roslaunch anymal_c_subt_semantic_front_end_filter replayandrecord.launch \
    bagfile:=/Data/20211007_SA_Monkey_ANYmal_Chimera/chimera_mission_2021_10_10/mission3_locomotion/2021-10-10-17-51-18_anymal-chimera_mission-001.bag\
    dumped_rosparameters:=/Data/20211007_SA_Monkey_ANYmal_Chimera/chimera_mission_2021_10_10/mission3_locomotion/2021-10-10-17-51-18_anymal-chimera-lpc_mission.yaml
    # path_map_darpa_tf_file:=/Data/20211007_SA_Monkey_ANYmal_Chimera/chimera_mission_2021_10_08/mission3_locomotion/2021-10-07-11-06-35_map_darpa_transformation.txt

