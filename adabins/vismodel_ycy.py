#!/usr/bin/env python
# coding: utf-8

from train import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.transforms import Bbox

test_loader_iter = None
train_loader_iter = None

<<<<<<< HEAD
def vis_one(loader = "train"):
=======
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
>>>>>>> 1b77de00b41a8d609f348a2e583f019931f579ec
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
    # inputimg = np.moveaxis(sample["image"][0].numpy(),0,2)
    inputimg.max(), inputimg.min()
    inputimg = (inputimg-inputimg.min())/(inputimg.max()- inputimg.min())
    fig, axs = plt.subplots(5, 4,figsize=(20, 20))
    if(axs.ndim==1):
        axs = axs[None,...]
    axs[0,0].imshow(inputimg[:,:,:3])
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
    axs[0,3].imshow(pc_diff,vmin = -5, vmax=5)
    axs[0,3].set_title("pc - traj")

    for i, (model, name) in enumerate(zip(model_list,names_list)):
        # bins, images = model(sample["image"][:,:3,...])
        images = model(sample["image"])
        pred = images[0].detach().numpy()

        plot_ind = 4+4*i
        axs[plot_ind//4, plot_ind%4].imshow(pred[0],vmin = 0, vmax=40)
        axs[plot_ind//4, plot_ind%4].set_title(f"{name}prediction")

        plot_ind = 5+4*i
        pred = nn.functional.interpolate(torch.tensor(pred)[None,...], torch.tensor(depth).shape[-2:], mode='bilinear', align_corners=True)
        pred = pred[0][0].numpy()
        print("pred shape:", pred.shape)
        diff = pred- depth
        print("diff shape:", diff.shape)
        print("depth shape:", depth.shape)
        mask_traj = depth>1e-9
        diff[~mask_traj] = 0
        axs[plot_ind//4, plot_ind%4].imshow(diff,vmin = -5, vmax=5)
        axs[plot_ind//4, plot_ind%4].set_title("Square Err %.1f"%np.sum(diff**2))
        divider = make_axes_locatable(axs[plot_ind//4, plot_ind%4])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cax = cax, mappable = axs[plot_ind//4, plot_ind%4].images[0])

        plot_ind = 6+4*i
        pcdiff = pred- pc_img
        mask_pc = (pc_img>1e-9)&(depth<1e-9)
        pcdiff[~mask_pc] = 0
        axs[plot_ind//4, plot_ind%4].imshow(pcdiff,vmin = -5, vmax=5)
        axs[plot_ind//4, plot_ind%4].set_title("Square Err to pc%.1f"%np.sum(pcdiff**2))

        plot_ind = 7+4*i
        axs[plot_ind//4, plot_ind%4].plot(pc_img[mask_pc].reshape(-1), pcdiff[mask_pc].reshape(-1), "x",ms=1,alpha = 0.2,label = "pc_img_err")
        axs[plot_ind//4, plot_ind%4].plot(depth[mask_traj].reshape(-1), diff[mask_traj].reshape(-1), "x",ms=1,alpha = 0.2, label = "traj_err")
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


if __name__=="__main__":
    parser.add_argument("--models", default="")
    parser.add_argument("--names", default="")
    parser.add_argument("--outdir", default="visulization/results")
    args = parse_args()
    args.data_path = "/media/anqiao/Semantic/Data/extract_trajectories_004_forRepeater/"

    try:
        # checkpoint_paths = sys.argv[1:]
        checkpoint_paths = args.models.split(" ")
        print(checkpoint_paths)
    except Exception as e:
        print(e)
        print("Usage: python vismodel checkpoint_path")
    
    
    model_list = [models.UnetAdaptiveBins.build(n_bins=args.modelconfig.n_bins, use_adabins=False, min_val=args.min_depth, max_val=args.max_depth,
                                            norm=args.modelconfig.norm) for i in checkpoint_paths]
    names_list = args.names.split(" ")
    loads = [model_io.load_checkpoint(checkpoint_path ,model) for checkpoint_path, model in zip(checkpoint_paths, model_list)]
    # model,opt,epoch = model_io.load_checkpoint(checkpoint_path ,model)
    model_list = [l[0] for l in loads]

    for i in range(20):
<<<<<<< HEAD
        vis_one("train")
=======
        vis_one("test", figname=os.path.join(args.outdir, "%d"%i))
>>>>>>> 1b77de00b41a8d609f348a2e583f019931f579ec
        plt.savefig(os.path.join(args.outdir, "%d.jpg"%i))
        # plt.show()
    # vis_network_structure()