#!/usr/bin/env python
# coding: utf-8

from train import *
import matplotlib.pyplot as plt
args = parse_args("@args_train.txt")
args.data_path = "/media/anqiao/Semantic/Data/extract_trajectories/"

test_loader = DepthDataLoader(args, 'online_eval').data
model = models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                          norm=args.norm)
model,opt,epoch = model_io.load_checkpoint("./checkpoints/UnetAdaptiveBins_09-May_22-30-nodebs6-tep25-lr0.000357-wd0.1-920d65a0-231f-4df2-a415-056d6a051ea6_best.pt" ,model)


sample = next(iter(test_loader))


inputimg = np.moveaxis(sample["image"][0][0:3, :, :].numpy(),0,2)
inputimg.max(), inputimg.min()
inputimg = (inputimg-inputimg.min())/(inputimg.max()- inputimg.min())
fig, axs = plt.subplots(1, 3,figsize=(20, 6))
axs[0].imshow(inputimg)
axs[0].set_title("Input")

depth = sample["depth"][0].numpy()
axs[1].imshow(depth)
axs[1].set_title("Label")

bins, images = model(sample["image"])
pred = images[0].detach().numpy()
axs[2].imshow(pred[0])
axs[2].set_title("Pred")
print("pred range:",pred.max(), pred.min())
plt.show()

