#!/usr/bin/env python
# coding: utf-8

from train import *
import matplotlib.pyplot as plt
<<<<<<< HEAD
args = parse_args("@args_train.txt")
args.data_path = "/media/anqiao/Semantic/Data/extract_trajectories_001/"

test_loader = DepthDataLoader(args, 'online_eval').data
model = models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                          norm=args.norm)
model,opt,epoch = model_io.load_checkpoint("/media/anqiao/Semantic/Data/checkpoints/UnetAdaptiveBins_30-Apr_16-45-nodebs3-tep25-lr0.000357-wd0.1-a1bc793e-5a8d-4bfb-8ad2-2ffb39586476_best.pt" ,model)
=======
from torch.utils.tensorboard import SummaryWriter

test_loader = None
train_loader = None
>>>>>>> b31f5ae7bd617731e01f153391b6a4ffc40a7234

def vis_one(loader = "test"):
    if(loader=="test"):
        data_loader = DepthDataLoader(args, 'online_eval').data if test_loader is None else test_loader
    elif(loader=="train"):
        data_loader = DepthDataLoader(args, 'train').data if train_loader is None else train_loader
    sample = next(iter(data_loader))

    inputimg = np.moveaxis(sample["image"][0][:3,:,:].numpy(),0,2)
    # inputimg = np.moveaxis(sample["image"][0].numpy(),0,2)
    inputimg.max(), inputimg.min()
    inputimg = (inputimg-inputimg.min())/(inputimg.max()- inputimg.min())
    fig, axs = plt.subplots(3, 4,figsize=(20, 15))
    if(axs.ndim==1):
        axs = axs[None,...]
    axs[0,0].imshow(inputimg)
    axs[0,0].set_title("Input")
    fig.suptitle(sample["path"])

    print(sample["depth"].shape)
    depth = sample["depth"][0][0].numpy()
    print(depth.shape)
    axs[0,1].imshow(depth)
    axs[0,1].set_title("Label")

    pc_img = sample["pc_image"][0][0].numpy()
    print(pc_img.shape)
    axs[0,2].imshow(pc_img)
    axs[0,2].set_title("pc_label")

    pc_diff = pc_img - depth
    pc_diff[depth<1e-9] = 0
    pc_diff[pc_img<1e-9] = 0
    axs[0,3].imshow(pc_diff,vmin = -5, vmax=5)
    axs[0,3].set_title("pc - traj")

    for i, (model, name) in enumerate(zip(model_list,names_list)):
        # bins, images = model(sample["image"][:,:3,...])
        bins, images = model(sample["image"])
        pred = images[0].detach().numpy()
        plot_ind = 4+2*i
        axs[plot_ind//4, plot_ind%4].imshow(pred[0])
        axs[plot_ind//4, plot_ind%4].set_title(f"pred_model_{name}")
        plot_ind = 5+2*i
        pred = nn.functional.interpolate(torch.tensor(pred)[None,...], torch.tensor(depth).shape[-2:], mode='bilinear', align_corners=True)
        pred = pred[0][0].numpy()
        print("pred shape:", pred.shape)
        diff = pred- depth
        print("diff shape:", diff.shape)
        print("depth shape:", depth.shape)
        mask = depth>1e-9
        diff[~mask] = 0
        axs[plot_ind//4, plot_ind%4].imshow(diff,vmin = -5, vmax=5)
        axs[plot_ind//4, plot_ind%4].set_title("Square Err %.1f"%np.sum(diff**2))
        plt.colorbar(mappable = axs[plot_ind//4, plot_ind%4].images[0])
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
    args = parse_args()
    args.data_path = "/media/chenyu/T7/Data/extract_trajectories_003_slim/"

    try:
        # checkpoint_paths = sys.argv[1:]
        checkpoint_paths = args.models.split(" ")
        print(checkpoint_paths)
    except Exception as e:
        print(e)
        print("Usage: python vismodel checkpoint_path")
    
    
    model_list = [models.UnetAdaptiveBins.build(n_bins=args.modelconfig.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                            norm=args.modelconfig.norm) for i in checkpoint_paths]
    names_list = args.names.split(" ")
    loads = [model_io.load_checkpoint(checkpoint_path ,model) for checkpoint_path, model in zip(checkpoint_paths, model_list)]
    # model,opt,epoch = model_io.load_checkpoint(checkpoint_path ,model)
    model_list = [l[0] for l in loads]

    vis_one("train")
    plt.show()
    # vis_network_structure()