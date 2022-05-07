#!/usr/bin/env python
# coding: utf-8

from train import *
import matplotlib.pyplot as plt

test_loader = None
train_loader = None

def vis_one(loader = "test"):
    if(loader=="test"):
        data_loader = DepthDataLoader(args, 'online_eval').data if test_loader is None else test_loader
    elif(loader=="train"):
        data_loader = DepthDataLoader(args, 'train').data if train_loader is None else train_loader
    sample = next(iter(data_loader))

    inputimg = np.moveaxis(sample["image"][0].numpy(),0,2)
    inputimg.max(), inputimg.min()
    inputimg = (inputimg-inputimg.min())/(inputimg.max()- inputimg.min())
    fig, axs = plt.subplots(3, 3,figsize=(20, 9))
    axs[0,0].imshow(inputimg)
    axs[0,0].set_title("Input")

    print(sample["depth"].shape)
    depth = sample["depth"][0][0].numpy()
    print(depth.shape)
    axs[0,1].imshow(depth)
    axs[0,1].set_title("Label")

    for i, model in enumerate(model_list):
        bins, images = model(sample["image"])
        pred = images[0].detach().numpy()
        plot_ind = 2+i
        axs[plot_ind//3, plot_ind%3].imshow(pred[0])
        axs[plot_ind//3, plot_ind%3].set_title(f"Pred_model{i}")
    print("pred range:",pred.max(), pred.min())
    



if __name__=="__main__":

    args = parse_args("@args_train.txt")
    args.data_path = "/media/chenyu/T7/Data/extract_trajectories/"

    try:
        checkpoint_paths = sys.argv[1:]
    except Exception as e:
        print("Usage: python vismodel checkpoint_path")
    
    
    model_list = [models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                            norm=args.norm) for i in checkpoint_paths]
    loads = [model_io.load_checkpoint(checkpoint_path ,model) for checkpoint_path, model in zip(checkpoint_paths, model_list)]
    # model,opt,epoch = model_io.load_checkpoint(checkpoint_path ,model)
    model_list = [l[0] for l in loads]

    vis_one("train")
    plt.show()