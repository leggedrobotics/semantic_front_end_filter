#!/usr/bin/env python
# coding: utf-8

# from train import *
from email.base64mime import header_length
from math import degrees
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.transforms import Bbox
from ruamel.yaml import YAML
import cv2
import msgpack
import msgpack_numpy as m
from semantic_front_end_filter.adabins.pointcloudUtils import RaycastCamera
from semantic_front_end_filter.adabins.train import *
from semantic_front_end_filter.adabins.elevation_vis import WorldViewElevationMap, plt_add_robot_patch
from scipy.spatial.transform import Rotation as Rotation
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
# from semantic_front_end_filter_ros.scripts.test_foward import RosVisulizer
m.patch()


# def full_extent(ax, pad=0.0):
#     """Get the full extent of an axes, including axes labels, tick labels, and
#     titles."""
#     # For text objects, we need to draw the figure first, otherwise the extents
#     # are undefined.
#     ax.figure.canvas.draw()
#     items = ax.get_xticklabels() + ax.get_yticklabels() 
# #    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
#     items += [ax, ax.title]
#     bbox = Bbox.union([item.get_window_extent() for item in items])

#     return bbox.expanded(1.0 + pad, 1.0 + pad)

def vis_one(loader = "test", figname=""):
    global test_loader_iter, train_loader_iter
    if(loader=="test"):
        if(test_loader_iter is None):
            data_loader = DepthDataLoader(args, 'online_eval').data 
            data_loader_iter = iter(data_loader)
            test_loader_iter = data_loader_iter
        else:
            data_loader_iter = test_loader_iter
    elif(loader=="train"):
        if(train_loader_iter is None):
            data_loader = DepthDataLoader(args, 'train').data
            data_loader_iter = iter(data_loader)
            train_loader_iter = data_loader_iter
        else:
            data_loader_iter = train_loader_iter
    sample = next(data_loader_iter)

    inputimg = np.moveaxis(sample["image"][0][:3,:,:].numpy(),0,2)
    inputpc = np.moveaxis(sample["image"][0][3:,:,:].numpy(),0,2)

    # inputimg = np.moveaxis(sample["image"][0].numpy(),0,2)
    inputimg.max(), inputimg.min()
    inputimg = (inputimg-inputimg.min())/(inputimg.max()- inputimg.min())
    fig, axs = plt.subplots(5, 4,figsize=(20, 20))
    if(axs.ndim==1):
        axs = axs[None,...]
    # axs[0,0].imshow(inputimg[:,:,:3])
    axs[0,0].imshow(cv2.cvtColor(inputimg[:,:,:3], cv2.COLOR_BGR2RGB))
    axs[0,0].set_title("Input")
    fig.suptitle(sample["path"])

    print(sample["depth"].shape)
    depth = sample["depth"][0][0].numpy()
    print(depth.shape)
    axs[0,1].imshow(depth,vmin = 0, vmax=40)
    axs[0,1].set_title("traj label")

    pc_img = sample["pc_image"][0][0].numpy()
    print(pc_img.shape)
    axs[0,2].imshow(pc_img,vmin = 0, vmax=40)
    axs[0,2].set_title("pc label")

    pc_diff = pc_img - depth
    pc_diff[depth<1e-9] = 0
    pc_diff[pc_img<1e-9] = 0
    # axs[0,3].imshow(pc_diff,vmin = -5, vmax=5)
    axs[0,3].imshow(inputpc,vmin = 0, vmax=40)
    # axs[0,3].set_title("pc - traj")
    axs[0,3].set_title("Input pc")
    for i, (model, name) in enumerate(zip(model_list,names_list)):
        # bins, images = model(sample["image"][:,:3,...])
        print(i)
        input_ = sample["image"]
        # input_[:, :, :, :] = 0
        if(model.use_adabins):
            bins, images = model(input_)
        else:
            images = model(input_)

        pred = images[0].detach().numpy()

        plot_ind = 4+4*i
        axs[plot_ind//4, plot_ind%4].imshow(pred[0],vmin = 0, vmax=40)
        axs[plot_ind//4, plot_ind%4].set_title(f"{name}prediction")

        plot_ind = 5+4*i
        # pred = nn.functional.interpolate(torch.tensor(pred)[None,...], torch.tensor(depth).shape[-2:], mode='bilinear', align_corners=True)
        pred = nn.functional.interpolate(torch.tensor(pred)[None,...], torch.tensor(depth).shape[-2:])
        pred = pred[0][0].numpy()
        print("pred shape:", pred.shape)
        diff = pred- depth
        print("diff shape:", diff.shape)
        print("depth shape:", depth.shape)
        mask_traj = depth>1e-9
        diff[~mask_traj] = None
        axs[plot_ind//4, plot_ind%4].imshow(diff,vmin = -5, vmax=5)
        axs[plot_ind//4, plot_ind%4].set_title("Square Err %.1f"%np.sum(diff**2))
        divider = make_axes_locatable(axs[plot_ind//4, plot_ind%4])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cax = cax, mappable = axs[plot_ind//4, plot_ind%4].images[0])

        plot_ind = 6+4*i
        pcdiff = pred- pc_img
        mask_pc = pc_img>1e-9
        pcdiff[~mask_pc] = 0
        axs[plot_ind//4, plot_ind%4].imshow(pcdiff,vmin = -5, vmax=5)
        axs[plot_ind//4, plot_ind%4].set_title("Square Err to pc%.1f"%np.sum(pcdiff**2))

        plot_ind = 7+4*i
        axs[plot_ind//4, plot_ind%4].plot(pred[mask_pc].reshape(-1), pcdiff[mask_pc].reshape(-1), "x",ms=1,alpha = 0.2,label = "pc_img_err")
        axs[plot_ind//4, plot_ind%4].plot(pred[mask_traj].reshape(-1), diff[mask_traj].reshape(-1), "x",ms=1,alpha = 0.2, label = "traj_err")
        axs[plot_ind//4, plot_ind%4].set_title("err vs distance")
        axs[plot_ind//4, plot_ind%4].set_xlabel("depth_prediction")
        axs[plot_ind//4, plot_ind%4].set_ylabel("err")
        axs[plot_ind//4, plot_ind%4].set_ylim((-20,20))
        axs[plot_ind//4, plot_ind%4].legend()

    # # extent = full_extent(axs[0,0]).transformed(fig.dpi_scale_trans.inverted())
    # # fig.savefig('%sinput_.png'%figname, bbox_inches=extent)
    # plt.imsave('%sinput_.png'%figname, inputimg[:,:,:3])
    # extent = full_extent(axs[0,1]).transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig('%strajlabel_.png'%figname, bbox_inches=extent)
    # extent = full_extent(axs[0,2]).transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig('%spclabel_.png'%figname, bbox_inches=extent)
    # extent = full_extent(axs[1,0]).transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig('%sprediction_.png'%figname, bbox_inches=extent)

        # axs[plot_ind//3, plot_ind%3].colorbar()
    print("pred range:",pred.max(), pred.min())
    
def vis_network_structure():
    data_loader = DepthDataLoader(args, 'train').data
    writer = SummaryWriter('.visulization/tmpvis')
    sample = next(iter(data_loader))
    model = model_list[0]
    writer.add_graph(model, sample["image"])
    writer.close()

"""
Load the parameters from the 
"""
def load_param_from_path(data_path):
    model_cfg = YAML().load(open(os.path.join(data_path, "ModelConfig.yaml"), 'r'))
    return model_cfg

def robot_patch(pos):
    """
    pos is the form data["pose"]["map"]: (x,y,z,rx,ry,rz,rw)
    """
    xyth = [*pos[:2], 
        Rotation.from_quat(pos[3:]).as_euler('xyz',degrees=False)[2]]

    robot_l = 4
    robot_w = 3
    robot = patches.Polygon([[-robot_l,-robot_w], [robot_l,-robot_w], 
                             [robot_l+0.5, 0], [robot_l,robot_w], 
                             [-robot_l,robot_w]], color = "r", fill = True)
    robot_center = patches.Circle((0,0),0.2, color = "w",fill = True)
    R = np.array([[np.cos(xyth[2]), -np.sin(xyth[2])],
                  [np.sin(xyth[2]),  np.cos(xyth[2])]])
    t = np.array([xyth[:2]])
    robot.set_xy( (R@ robot.get_xy().T ).T+t )
    robot_center.center =  (robot_center.center[0] + t[0][0], robot_center.center[1] + t[0][1])
    # t2 = mpl.transforms.Affine2D().rotate(xyth[2]).translate(xyth[0],xyth[1])#+ ax.transData
    # robot.set_transform(t2) 
    # robot_center.set_transform(t2)
    return (robot, robot_center)
def saveOnePic(pack_path, path):
    with open(pack_path, "rb") as data_file:
        bytedata = data_file.read()
        data = msgpack.unpackb(bytedata)
    # transform to tensor to accelerate 
    image_array = data["images"]["cam4"]
    pose_array = data["pose"]["map"]
    image = torch.Tensor(image_array).to(device)
    points = torch.Tensor(data["pointcloud"][:, :3]).to(device)
    pose = torch.Tensor(pose_array).to(device)
    
    # get pc image
    pc_img = torch.zeros_like(image[:1, ...]).to(device).float()
    pc_img = raycastCamera.project_cloud_to_depth(
                    pose.cpu(), points, pc_img)
    # get prediction
    input = torch.cat([image/255., pc_img],axis=0)
    input = input[None, ...]
    for i, (m, s) in enumerate(zip([0.387, 0.394, 0.404, 0.120], [0.322, 0.32, 0.30,  1.17])):
        input[0, i, ...] = (input[0, i, ...] - m)/s
    pred = model(input)[0][0]
    pred_for_show = pred.clone()
    # filter with input pc
    pred [(pc_img[0]==0)] = np.nan
    pred = pred.T

    # get elevation from prediction
    pts = raycastCamera.project_depth_to_cloud(pose.cpu(), pred)
    pts = pts[~torch.isnan(pts[:,0])]
    # height_mask = (pts[:,2] < pose[2]) & (pose[0]-5 < pts[:,0]) & (pts[:,0] < pose[0]+5) & (pose[1]-5 < pts[:,1]) & (pts[:,1] < pose[1]+5)
    pred_points = pts.detach().cpu().numpy()
    elevation_pred_fusion.move_to_and_input(pose_array[0:3], pred_points)
    map_pred_fusion = elevation_pred_fusion.get_elevation_map()

    # get elevation map from only pc
    # mask = (data["pointcloud"][:, 0]<pose_array[0]+10) & (data["pointcloud"][:, 0]>pose_array[0]-10) & (data["pointcloud"][:, 1]<pose_array[1]+10) & (data["pointcloud"][:, 1]>pose_array[1]-10) & (data["pointcloud"][:, 2]<pose_array[2])
    test_points = data["pointcloud"][:, :3].copy()
    test_points[:, 2] = 0
    # elevation_pc_fusion.move_to_and_input(pose_array[0:3], data["pointcloud"][:, :3])
    elevation_pc_fusion.move_to_and_input(pose_array[0:3], test_points)
    map_pc_fusion = elevation_pc_fusion.get_elevation_map()
    print(map_pc_fusion[~np.isnan(map_pc_fusion)].mean())
    
    ## Plot
    fig, axs = plt.subplots(2, 3,figsize=(20, 20))
    fig.suptitle(pack_path)
    axs[0, 0].imshow(cv2.cvtColor(np.moveaxis(image_array, 0, 2)/255,  cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("input image")

    axs[0, 1].imshow(pc_img[0].cpu().numpy(), vmin = 0, vmax=40)
    axs[0, 1].set_title("input pc && pc label")

    axs[0, 2].imshow(data['images']['cam4depth'][0], vmin = 0, vmax=40)
    axs[0, 2].set_title("trajectory label")

    axs[1, 0].imshow(pred_for_show.cpu().detach().numpy())
    axs[1, 0].set_title("prediction")

    axs[1, 1].imshow(map_pred_fusion)
    center = elevation_pc_fusion.param.map_length/elevation_pc_fusion.param.resolution/2
    xyth = [center, center, Rotation.from_quat(pose_array[3:]).as_euler('xyz',degrees=True)[2]]
    width = 5
    # axs[1, 1].add_patch(Rectangle((xyth[0] - width/2, xyth[1] - 2*width/2), width, 2*width, angle = -xyth[2]))
    axs[1, 1].annotate("", xytext=(center, center), xy=(np.sin(np.deg2rad(xyth[2]))*width+center, np.cos(np.deg2rad(xyth[2]))*width+center), arrowprops=dict(headlength=40, headwidth=20))
    axs[1, 1].set_title("elevation map with prediction")

    axs[1, 2].imshow(map_pc_fusion)
    axs[1, 2].annotate("", xytext=(center, center), xy=(np.sin(np.deg2rad(xyth[2]))*width+center, np.cos(np.deg2rad(xyth[2]))*width+center), arrowprops=dict(headlength=40, headwidth=20))
    axs[1, 2].set_title("elevation map with only pc")

    # plt.show()
    plt.savefig(path)
    print(path)
    plt.close(fig)

def getKey(msg_path):
    return (int(msg_path.split('_')[1]), int((msg_path.split('.')[0]).split('_')[-1]))

def visOneTraj(traj_path, model, args):
    outdir = args.outdir + "/" + traj_path.split('/')[-1]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for i, pack in enumerate(sorted([f for f in os.listdir(traj_path) if 'traj' in f], key = getKey)):
        if('traj' in pack):
            pack_path = traj_path + '/' + pack
            print(pack_path)
            saveOnePic(pack_path, path=outdir+'/%03d.jpg'%i)

    os.system("ffmpeg -framerate 2 -pattern_type glob -i '" +outdir+"/*.jpg' -c:v libx264 -pix_fmt yuv420p "+outdir+"/out.mp4")
 
if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raycastCamera = RaycastCamera("/home/anqiao/tmp/semantic_front_end_filter/anymal_c_subt_semantic_front_end_filter/config/calibrations/alphasense", device)
    elevation_pred_fusion = WorldViewElevationMap(resolution = 0.1, map_length = 10, init_with_initialize_map = False)
    elevation_pc_fusion = WorldViewElevationMap(resolution = 0.1, map_length = 10, init_with_initialize_map = False)

    parser = ArgumentParser()
    parser.add_argument("--model_path", default="/home/anqiao/tmp/semantic_front_end_filter/adabins/checkpoints/2022-08-03-16-26-08/UnetAdaptiveBins_latest.pt")
    # parser.add_argument("--model_path", default="/media/anqiao/Semantic/Models/2022-08-03-16-26-08/UnetAdaptiveBins_latest.pt")
    parser.add_argument("--dataset_path", default="/media/anqiao/Semantic/Data/extract_trajectories_006_Italy/extract_trajectories")
    # parser.add_argument("--dataset_path", default="/home/anqiao/tmp/semantic_front_end_filter/Labelling/extract_trajectories")
    # parser.add_argument("--outdir", default="visulization/results")
    args = parse_args(parser)
    args.outdir = args.model_path.rsplit("/", 1)[0] + "/videos"
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    model_cfg = load_param_from_path(os.path.dirname(args.model_path))
    # model = None
    model = models.UnetAdaptiveBins.build(input_channel = 4,**model_cfg)
    model = model_io.load_checkpoint(args.model_path ,model)[0]
    model.to(device)
    for traj in os.listdir(args.dataset_path):
        if "Re" in traj:
            traj_path = os.path.join(args.dataset_path, traj)
            visOneTraj(traj_path, model, args)
        # os.system("ffmpeg -framerate 1 -pattern_type glob -i '" +args.outdir+"/*.jpg' -c:v libx264 -pix_fmt yuv420p "+args.outdir+"/out.mp4")
        # if "Reconstruct" in dir:
        #     print(os.path.join(root, dir))

    # for i in range(20):
    #     vis_one("train", figname=os.path.join(args.outdir, "%d"%i))
    #     plt.savefig(os.path.join(args.outdir, "%d.jpg"%i))
        # plt.show()
    # vis_network_structure()


    