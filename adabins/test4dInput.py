#!/usr/bin/env python
# coding: utf-8

from train import *
import matplotlib.pyplot as plt
import geffnet
args = parse_args("@args_train.txt")
args.data_path = "/media/anqiao/Semantic/Data/extract_trajectories_001/"

test_loader = DepthDataLoader(args, 'online_eval').data
model = models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                          norm=args.norm)
model,opt,epoch = model_io.load_checkpoint("./checkpoints/AdaBins_kitti.pt" ,model)

orginal_first_layer_weight = model.encoder.original_model.conv_stem.weight
model.encoder.original_model.conv_stem = geffnet.conv2d_layers.Conv2dSame(4, 48, kernel_size=(3, 3), stride=(2, 2), bias=False)
with torch.no_grad():
    model.encoder.original_model.conv_stem.weight[:, 0:3, :, :] = orginal_first_layer_weight
    # model.encoder.original_model.conv_stem.weight[:, 3, :, :] = torch.zeros([48, 3, 3])

    
sample = next(iter(test_loader))


inputimg = np.moveaxis(sample["image"][0].numpy(),0,2)
inputimg.max(), inputimg.min()
inputimg = (inputimg-inputimg.min())/(inputimg.max()- inputimg.min())
fig, axs = plt.subplots(1, 3,figsize=(20, 6))
axs[0].imshow(inputimg)
axs[0].set_title("Input")

depth = sample["depth"][0].numpy()
axs[1].imshow(depth)
axs[1].set_title("Label")

FDimage = torch.zeros([1, 4, 540, 720])
FDimage[:, 0:3, :, :] = sample["image"]
FDimage[:, 3, :, :] = sample["image"][:, 0, :, :] 
bins, images = model(FDimage)
pred = images[0].detach().numpy()
axs[2].imshow(pred[0])
axs[2].set_title("Pred")
print("pred range:",pred.max(), pred.min())
plt.show()

