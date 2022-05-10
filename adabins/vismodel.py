#!/usr/bin/env python
# coding: utf-8

from train import *
import matplotlib.pyplot as plt
args = parse_args("@args_train.txt")
args.data_path = "/media/anqiao/Semantic/Data/extract_trajectories_001/"

test_loader = DepthDataLoader(args, 'online_eval').data
model = models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                          norm=args.norm)
model,opt,epoch = model_io.load_checkpoint("/media/anqiao/Semantic/Data/checkpoints/UnetAdaptiveBins_30-Apr_16-45-nodebs3-tep25-lr0.000357-wd0.1-a1bc793e-5a8d-4bfb-8ad2-2ffb39586476_best.pt" ,model)


sample = next(iter(test_loader))


inputimg = np.moveaxis(sample["image"][0].numpy(),0,2)
inputimg.max(), inputimg.min()
inputimg = (inputimg-inputimg.min())/(inputimg.max()- inputimg.min())
fig, axs = plt.subplots(1, 4,figsize=(20, 6))
axs[0].imshow(inputimg)
axs[0].set_title("Input")

depth = sample["depth"][0].numpy()
axs[1].imshow(depth)
axs[1].set_title("Label")

bins, images = model(sample["image"])
pred = images[0].detach().numpy()
axs[2].imshow(pred[0])
axs[2].set_title("Pred")

import cv2
pred_resized = cv2.resize(pred[0], (depth.shape[1], depth.shape[0]))
diff = pred_resized - depth[:, :, 0]
im = axs[3].imshow(diff, cmap = 'plasma')
plt.colorbar(im)


print("pred range:",pred.max(), pred.min())
plt.show()

