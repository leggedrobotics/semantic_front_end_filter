#!/usr/bin/env python
# coding: utf-8

from train import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

test_loader_iter = None
train_loader_iter = None

def vis_one(loader = "test"):
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
    fig, axs = plt.subplots(3, 4,figsize=(20, 15))
    if(axs.ndim==1):
        axs = axs[None,...]
    axs[0,0].imshow(inputimg)
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
        bins, images = model(sample["image"])
        pred = images[0].detach().numpy()
        plot_ind = 4+2*i
        axs[plot_ind//4, plot_ind%4].imshow(pred[0],vmin = 0, vmax=40)
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
    parser.add_argument("--outdir", default="visulization/results")
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

    for i in range(20):
        vis_one("test")
        plt.savefig(os.path.join(args.outdir, "%d.jpg"%i))
    # plt.show()
    # vis_network_structure()