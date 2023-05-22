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
import cv2 as cv
from semantic_front_end_filter.adabins.pointcloudUtils import RaycastCamera

m.patch()

def filter_msg_pack(input_path, output_path):
    with open(input_path, "rb") as data_file:
        byte_data = data_file.read()
        data = msgpack.unpackb(byte_data)
    depth_gt = np.moveaxis(data["images"]["cam4depth"],0,2)
    
    # pc_img = torch.zeros_like(torch.Tensor(data["images"]["cam4"][:1, ...])).to(device).float()
    pc_img = torch.zeros(1, 540, 720).to(device)
    pose = torch.Tensor(data["pose"]["map"]).to(device)
    points = torch.Tensor(data["pointcloud"][:,:3]).to(device)
    pc_img = raycastCamera.project_cloud_to_depth(pose, points, pc_img)
    pc_image = pc_img.squeeze(0)[..., None].cpu().numpy()
    # old method
    # pc_image = np.zeros_like(depth_gt[:,:,:1])
    # pos = data["pose"]["map"][:3]
    # pc = data["pointcloud"]
    # pc_distance = np.sqrt(np.sum((pc[:,:3] - pos)**2, axis = 1))

    # imgshape = pc_image.shape[:-1] 
    # pc_proj_mask = pc[:, 10] > 0 # the point is on the graph
    # pc_proj_loc = pc[:, 11:13] # the x,y pos of point on image
    # pc_proj_mask = (pc_proj_mask & (pc_proj_loc[:, 0]<imgshape[1])
    #                             & (pc_proj_loc[:, 0]>=0)
    #                             &  (pc_proj_loc[:, 1]<imgshape[0])
    #                             &  (pc_proj_loc[:, 1]>=0))
    # pc_proj_loc = pc_proj_loc[pc_proj_mask].astype(np.int32)
    # pc_distance = pc_distance[pc_proj_mask]
    # pc_image[pc_proj_loc[:,1], pc_proj_loc[:,0], 0] = pc_distance
    
    if(np.sum(data["images"]["cam4depth"][0,:,:]>1e-9)<10 ):
        print("%s is filtered out as it only have %d nonzero value"%(input_path, 
            np.sum(data["images"]["cam4depth"][0,:,:]>1e-9)) )
        return

    with open(output_path, "wb") as out_file:
        file_dat = msgpack.packb({
            "time": data['time'],
            "pose":data["pose"]["map"],
            "image":data["images"]["cam4"],
            "pc_image":pc_image,
            "depth_var":data["images"]["cam4depth"]
        })
        out_file.write(file_dat)


def dilation(pc_image, w):
    """
    pc_image: 3-dim image
    """
    # Inpaint with dilation
    kernel = np.ones((w, w), np.float32)
    dst_conv_mean = cv.filter2D(pc_image, -1, kernel)[:, :, np.newaxis]
    
    mask = np.zeros_like(pc_image)
    mask[(dst_conv_mean>1e-9) & (pc_image<1e-9)] = 1
    mask = np.uint8(mask)
    pc = cv.inpaint(pc_image,mask,2,cv.INPAINT_NS)[:, :, np.newaxis]
    return pc

def dilate_pc(input_path, output_path):
    """
    Augment the pc image in a slim version dataset by dilating it. 
    The new dilated pc images will be stacked behind the first pc image
    """
    with open(input_path, "rb") as data_file:
        byte_data = data_file.read()
        data = msgpack.unpackb(byte_data)

    pc_image = data["pc_image"]
    
    pc_images = [pc_image]+[dilation(pc_image, w) for w in [5,10]]
    import matplotlib.pyplot as plt
    data = data.copy()
    with open(output_path, "wb") as out_file:
        data["pc_image"] = np.concatenate(pc_images, axis = 2)
        out_file.write(msgpack.packb(data))


if __name__ == "__main__":
    # For filter_msg_pack
    data_path = "/media/anqiao/Semantic/Data/extract_trajectories_006_SA/extract_trajectories"
    target_path = "/media/anqiao/Semantic/Data/extract_trajectories_006_SA_slim/extract_trajectories"
    # For dilate_pc
    # data_path = "/media/chenyu/T7/Data/extract_trajectories_006_Zurich/extract_trajectories"
    # target_path = "/media/chenyu/T7/Data/extract_trajectories_006_Zurich_augment/extract_trajectories"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raycastCamera = RaycastCamera("/home/anqiao/tmp/semantic_front_end_filter/anymal_c_subt_semantic_front_end_filter/config/calibrations/alphasense", device)    
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
                if(os.path.exists(target_file_path)): 
                    print(target_file_path, "already exist")
                    continue
                ## choose what to do by commenting out corresponding lines
                filter_msg_pack(file_path, target_file_path)
                # dilate_pc(file_path, target_file_path)
    