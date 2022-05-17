#!/usr/bin/env python
# coding: utf-8

from termios import VMIN
from train import *
import matplotlib.pyplot as plt
args = parse_args("@args_train.txt")
args.data_path = "/media/anqiao/Semantic/Data/extract_trajectories/"

test_loader = DepthDataLoader(args, 'online_eval').data
model = models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                          norm=args.norm)
model,opt,epoch = model_io.load_checkpoint("/media/anqiao/Semantic/Data/checkpoints/UnetAdaptiveBins_13-May_08-25-ReduceBinsto20_best.pt" ,model)


sample = next(iter(test_loader))


inputimg = np.moveaxis(sample["image"][0][0:3, :, :].numpy(),0,2)
inputimg.max(), inputimg.min()
inputimg = (inputimg-inputimg.min())/(inputimg.max()- inputimg.min())
fig, axs = plt.subplots(2, 2,figsize=(20, 6))
axs[0, 0].imshow(inputimg)
axs[0, 0].set_title("Input")

depth = sample["depth"][0].numpy()
axs[0, 1].imshow(depth)
axs[0, 1].set_title("Label")

bins, images = model(sample["image"])
pred = images[0].detach().numpy()
axs[1, 0].imshow(pred[0])
axs[1, 0].set_title("Pred")
print("pred range:",pred.max(), pred.min())


import cv2
pred_resized = cv2.resize(pred[0], (depth.shape[1], depth.shape[0]))
diff = pred_resized - depth[:, :, 0]
im = axs[1, 1].imshow(diff, cmap = 'plasma', vmin = -10, vmax = 10)
plt.colorbar(im, ax = axs[1, 1])
axs[1, 1].set_title("Difference")

plt.show()
