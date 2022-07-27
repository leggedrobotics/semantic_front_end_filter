#!/usr/bin/env python

import os
from ruamel.yaml import YAML
from argparse import ArgumentParser
import struct

import cv2

import numpy as np
import matplotlib.pyplot as plt  # Just for debug.
import matplotlib

import msgpack
import msgpack_numpy as m
import math
from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_from_matrix, quaternion_from_matrix

m.patch()

class RosVisulizer:
    def __init__(self, topic):
        import rospy
        from sensor_msgs import point_cloud2
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header
        import rosgraph
        assert rosgraph.is_master_online()
        self.pub = rospy.Publisher(topic, PointCloud2, queue_size=1)
        # if(rospy.)
        rospy.init_node('ros_visulizer', anonymous=True)
        
        
    def buildPoint(self, x,y,z,r,g,b,a=None):
        if(np.array([r,g,b]).max()<1.01):
            r = int(r * 255.0)
            g = int(g * 255.0)
            b = int(b * 255.0)
            a = 255 if a is None else int(a * 255.0)
        else:
            r = int(r)
            g = int(g)
            b = int(b)
            a = 255 if a is None else int(a)
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        return [x, y, z, rgb]

    def publish_point_cloud(self, pc, img=None):
        """
        the Pc's field is x,y,z, i, r,g,b, [camflag, projx, projy]x3
        thanks to example at https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
        """
        from sensor_msgs import point_cloud2
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header

        header = Header()
        header.frame_id = "map"
        
        fields = [
          PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          # PointField('rgb', 12, PointField.UINT32, 1),
          PointField('rgba', 12, PointField.UINT32, 1),
        ]

        pc = np.array(pc)
        pc[:, :3] -= pc[:, :3].mean(axis=0)
        # pc[:, 4:7] /=256.
        if img is None:
            colors = pc[:, 4:7] /256.
        else: # get colors by projecting points to img
            colors = np.zeros_like(pc[:, 4:7])
            imgshape = img.shape
            print("imgshape :",imgshape)
            assert(len(imgshape)==3 and (imgshape[0] in [1,3]))
            tmp_fxy = pc[:, 7+ 1*3: 7 + (1+1)*3]
            flag, px, py = tmp_fxy[:,0], tmp_fxy[:,1], tmp_fxy[:,2]
            flag = flag>0.5 # turn 0-1 into boolean
            flag = flag & (0<=px) & (px< imgshape[2]) & (0<=py) & (py< imgshape[1])
            print("flag.sum :",flag.sum())
            px[~flag] = 0
            py[~flag] = 0
            px = px.astype(np.int32)
            py = py.astype(np.int32)
            feats = img[:, py, px]
            feats = feats.T
            colors = np.where(flag[:,None], feats, colors)
            print("colors.max :",colors.max())
        cloud = point_cloud2.create_cloud(header, fields, 
            [self.buildPoint(*p[:3], *c) for p,c in zip(pc, colors)])

        self.pub.publish(cloud)
        
## SHOW FULL IMG
def showImages(images):

    N = len(images)

    fig, axs = plt.subplots(1, N, figsize=(20, 6))
    # labels = list(images.keys())
    # labels.sort()

    for i, (label, image) in enumerate(images.items()):
        image_size = str(image.shape)
        if 'map' in label:
            image = image[0]
        else:
            image = np.moveaxis(image, 0, 2)
        if(image.shape[2]==2): # depth maps and covariance map
            image=image[:,:,0]
        else:
            image = image/256
        axs[i].imshow(image)
        axs[i].set_title(label + image_size, fontsize=8)
        axs[i].xaxis.set_ticklabels([])
        axs[i].axes.yaxis.set_ticklabels([])
    return fig

## SHOW PATCHES
def showPatches(patches):
    rows = 5
    N = max(2, math.ceil(len(patches) / rows))

    fig, axs = plt.subplots(N, rows, figsize=(10, 6))
    # labels = list(images.keys())
    # labels.sort()

    row = 0
    col = 0

    for i in range(N * rows):
        if i < len(patches):
            image = patches[i]
            image_size = str(image.shape)
            image = np.moveaxis(image, 0, 2)
            image = image/256.

            axs[row, col].imshow(image)
            # position_str = np.array2string(position, precision=2, suppress_small=True)
            # axs[row, col].set_title(position_str, fontsize=7)

        axs[row, col].xaxis.set_ticklabels([])
        axs[row, col].axes.yaxis.set_ticklabels([])

        col += 1
        if col == 5:
            col = 0
            row += 1

    return fig

def showPointClouds(pc):
    from mpl_toolkits.mplot3d import Axes3D
    #xyz i rgb
    pc = np.array(pc)
    colors = pc[:, 4:7]/255.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:,0], pc[:,1], pc[:,2], c = colors)
    return fig

def showPointCloudsOnGraph(pc, images):
    N = len(images)
    fig, axs = plt.subplots(1, N, figsize=(20, 6))
    
    pc = np.array(pc)
    imgs = [img.copy() for img in images]
    ### Color the points differently to verify that the projection is indeed correct.
    print("pc max and min", pc[:,:3].min(axis=0), pc[:,:3].max(axis=0))
    # pc = pc[pc[:,2]<-0.5]
    for p in pc:
        # s = max(0, min(1,(p[2] +1.55)/(1+1.55)))
        # s = max(0, min(1, (p[0]+p[1] -7.5)/1))
        # c = [int(255-255*s), int(255*s),  int(255-255*s) ]
        c = [0,255,0]
        for i in range(N):
            assert (abs(p[7+3*i])<1e-6 or abs(p[7+3*i]-1)<1e-6)
            if(p[7+3*i]>0.5):
                if(0<=int(p[7+3*i+1])<imgs[i].shape[2] and  
                   0<= int(p[7+3*i+2])<imgs[i].shape[1]):
                    imgs[i][:,int(p[7+3*i+2]), int(p[7+3*i+1])] = c

    for i, (image) in enumerate(imgs):
        image_size = str(image.shape)
        image = np.moveaxis(image, 0, 2)
        image = image/256.
        axs[i].imshow(image)
        axs[i].set_title(image_size, fontsize=8)
        axs[i].xaxis.set_ticklabels([])
        axs[i].axes.yaxis.set_ticklabels([])
    return fig

def showPointCloudswithColor(pc, images):
    N = len(images)
    fig, axs = plt.subplots(1, N, figsize=(20, 6))
    
    print(len(pc))
    pc = np.array(pc)
    imgs = [img.copy() for img in images]
    masks = [img.copy() for img in images]
    masks = [mask.fill(False) for mask in masks]
    ### Color the points differently to verify that the projection is indeed correct.
    print("pc max and min", pc[:,:3].min(axis=0), pc[:,:3].max(axis=0))
    # pc = pc[pc[:,2]<-0.5]
    for p in pc:
        # s = max(0, min(1,(p[2] +1.55)/(1+1.55)))
        # s = max(0, min(1, (p[0]+p[1] -7.5)/1))
        # c = [int(255-255*s), int(255*s),  int(255-255*s) ]
        c = [0,255,0]
        for i in range(N):
            assert (abs(p[7+3*i])<1e-6 or abs(p[7+3*i]-1)<1e-6)
            if(p[7+3*i]>0.5):
                if(0<=int(p[7+3*i+1])<imgs[i].shape[2] and  
                   0<= int(p[7+3*i+2])<imgs[i].shape[1]):
                    masks[i][:, int(p[7+3*i+2]), int(p[7+3*i+1])] = True

    for i, (image) in enumerate(imgs):
        image = image(masks[i])
        image_size = str(image.shape)
        image = np.moveaxis(image, 0, 2)
        image = image/256.
        axs[i].imshow(image)
        axs[i].set_title(image_size, fontsize=8)
        axs[i].xaxis.set_ticklabels([])
        axs[i].axes.yaxis.set_ticklabels([])
    return fig

def main():
    import time

    rosv_ok = False
    try:
        rosv = RosVisulizer("pointcloud")
        rosv_ok = True
    except Exception as e:
        print("Ros visulizer failed", e)
    
    parser = ArgumentParser()
    parser.add_argument('--dir', required=True, help='Directory to visualize.')
    parser.add_argument('--hint', required=False, default=None, help='interested file name')

    args = parser.parse_args()

    # Get file names.
    file_names_found = [os.path.join(root, file) for root, dirs, files in os.walk(args.dir) for file in files if
                  file.endswith('.msgpack')]

    if args.hint is not None:
        file_names = [file for file in file_names_found if args.hint in file]
    else:
        file_names = file_names_found
    file_names.sort()

    print('Number of files found: ' + str(len(file_names)))

    for i in range(0, len(file_names), 1):

        # plt.cla()
        with open(file_names[i], "rb") as data_file:
            byte_data = data_file.read()
        data = msgpack.unpackb(byte_data)

        print("pose",data["pose"])

        img_keys = list(data['images'].keys())
        img_keys.sort()

        img_dict = {}
        for key in img_keys:
            img_dict[key] = data['images'][key]


        fig0 = showImages(img_dict)

        pc = data['pointcloud']
        fig2 = showPointClouds(pc)

        fig3 =showPointCloudsOnGraph(pc, [img_dict[k] for k in ["cam3","cam4","cam5"]])
        if(rosv_ok):
            # rosv.publish_point_cloud(data['pointcloud'])
            img = img_dict["cam4depth"].copy()
            img[img>10] = 0
            flags = (img!=0)
            img[flags] = img[flags]/ img.max()
            img = (img*255).astype(np.uint8)
            print("max,min :",img.max(),img.min())
            im_color = cv2.applyColorMap(img[0], cv2.COLORMAP_JET)
            im_color = np.moveaxis(im_color, 2, 0)
            print("im_color :",im_color.shape)
            
            rosv.publish_point_cloud(data['pointcloud'], im_color)

        plt.show()

        
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close(fig0)
        # plt.close(fig1)

if __name__ == '__main__':
    main()
