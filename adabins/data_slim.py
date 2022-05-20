"""
Chenyu Yang 
Call script to preprocess the dataset. 
E.g. project raw point clouds onto pc_images
"""
import os
import torch
import numpy as np
import msgpack
import msgpack_numpy as m
m.patch()

def filter_msg_pack(input_path, output_path):
    with open(input_path, "rb") as data_file:
        byte_data = data_file.read()
        data = msgpack.unpackb(byte_data)
    depth_gt = np.moveaxis(data["images"]["cam4depth"],0,2)

    pc_image = np.zeros_like(depth_gt[:,:,:1])
    pos = data["pose"]["map"][:3]
    pc = data["pointcloud"]
    pc_distance = np.sqrt(np.sum((pc[:,:3] - pos)**2, axis = 1))

    imgshape = pc_image.shape[:-1] 
    pc_proj_mask = pc[:, 10] > 0.5 # the point is on the graph
    pc_proj_loc = pc[:, 11:13] # the x,y pos of point on image
    pc_proj_mask = (pc_proj_mask & (pc_proj_loc[:, 0]<imgshape[1])
                                & (pc_proj_loc[:, 0]>=0)
                                &  (pc_proj_loc[:, 1]<imgshape[0])
                                &  (pc_proj_loc[:, 1]>=0))
    pc_proj_loc = pc_proj_loc[pc_proj_mask].astype(np.int32)
    pc_distance = pc_distance[pc_proj_mask]
    pc_image[pc_proj_loc[:,1], pc_proj_loc[:,0], 0] = pc_distance


    with open(output_path, "wb") as out_file:
        file_dat = msgpack.packb({
            "image":data["images"]["cam4"],
            "pc_image":pc_image,
            "depth_var":data["images"]["cam4depth"]
        })
        out_file.write(file_dat)


if __name__ == "__main__":
    data_path = "/media/chenyu/Semantic/Data/extract_trajectories_003/"
    target_path = "/media/chenyu/Semantic/Data/extract_trajectories_003_slim/"
    count = 0
    for root, dirs, files in os.walk(data_path):
        target_root = root.replace(data_path, target_path)
        try: 
            os.mkdir(target_root)
        except FileExistsError as e:
            pass
        for file in files:
            if file.startswith('traj') and file.endswith('.msgpack'):
                file_path = os.path.join(root,file)
                target_file_path = os.path.join(target_root,file)
                count+=1
                print("count:", count)
                if(os.path.exists(target_file_path)): continue
                filter_msg_pack(file_path, target_file_path)
    