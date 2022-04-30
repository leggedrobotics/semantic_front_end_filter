#!/usr/bin/env python
# coding: utf-8

from train import *
import matplotlib.pyplot as plt
args = parse_args("@args_train.txt")
args.data_path = "/media/chenyu/T7/Data/extract_trajectories/"

test_loader = DepthDataLoader(args, 'online_eval').data
model = models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                          norm=args.norm)
model,opt,epoch = model_io.load_checkpoint("./checkpoints/UnetAdaptiveBins_29-Apr_23-48-nodebs3-tep25-lr0.000357-wd0.1-7f9878ca-111a-414c-a3ed-f49ebb65f87e_latest.pt" ,model)


sample = next(iter(test_loader))


inputimg = np.moveaxis(sample["image"][0].numpy(),0,2)
inputimg.max(), inputimg.min()
inputimg = (inputimg-inputimg.min())/(inputimg.max()- inputimg.min())
plt.figure()
plt.imshow(inputimg)

depth = sample["depth"][0].numpy()
plt.figure()
plt.imshow(depth)

bins, images = model(sample["image"])
pred = images[0].detach().numpy()
plt.figure()
plt.imshow(pred[0])
print("pred range:",pred.max(), pred.min())
plt.show()

