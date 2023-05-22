import matplotlib.pyplot as plt
import numpy as np 
import cv2
import msgpack
import msgpack_numpy as m
from skimage.color import label2rgb

m.patch()

def getImage(file_path , traj = 0, datum = 0):
    f = open("{}/traj_{}_datum_{}.msgpack".format(file_path, traj, datum), "rb")
    bdata = f.read()
    data = msgpack.unpackb(bdata)
    return cv2.cvtColor(np.moveaxis(data["image"], 0, 2)/255, cv2.COLOR_BGR2RGB)

def getDatum(file_path , traj = 0, datum = 0):
    f = open("{}/traj_{}_datum_{}.msgpack".format(file_path, traj, datum), "rb")
    bdata = f.read()
    data = msgpack.unpackb(bdata)
    return data

# import torch
# image = getImage('/home/anqiao/catkin_ws/SA_dataset/extract_trajectories_test/extract_trajectories/Reconstruct_2022-07-18-20-34-01_0')
# # plt.imshow(image)
# inputImage = torch.Tensor(image).to('cuda')
# inputImage = torch.moveaxis(inputImage, 2, 0)[None, ...]
# print(inputImage.shape)
# inputGradient = [gra[None,None,:,:] for gra in torch.gradient(inputImage[0,0], dim=[0,1])]

# # Gray scale image
# inputImageGray = (torch.sum(inputImage, axis=1)/3)[None, ...]
# grad = torch.gradient(inputImageGray[0, 0], dim=(0, 1))
# plt.subplot(1, 2, 1)
# plt.imshow(grad[0].cpu().numpy())
# # plt.show()

# # Gaussian filter
# kernel = torch.Tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])[None, None, ...].to('cuda')
# inputImageGaussian = torch.nn.functional.conv2d(inputImageGray, kernel, padding=1)
# # plt.imshow(inputImageGaussian.cpu().numpy()[0][0])
# grad_gaussian = torch.gradient(inputImageGaussian[0, 0], dim=(0, 1))
# plt.subplot(1, 2, 2)
# plt.imshow(grad_gaussian[0].cpu().numpy())
# plt.show()

from skimage import segmentation, exposure
import cv2
from skimage.filters import rank
from skimage.morphology import disk
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
from skimage import data
from skimage.color import rgb2gray






# image = cv2.imread('/home/anqiao/Desktop/forest.png')/255
# image = image[:, :, 0]
# image = img_as_ubyte(data.eagle())

# print(image.shape)
image = getImage('/home/anqiao/catkin_ws/SA_dataset/mountpoint/Data/extract_trajectories_006_Italy_slim/extract_trajectories/Reconstruct_2022-07-19-19-02-15_0', 8, 3)
# print(image.shape)
print(exposure.is_low_contrast(image))
# image = exposure.rescale_intensity(image)
slic = segmentation.slic(image, n_segments=300, compactness=0.01, sigma=1)

denoised = rgb2gray(image)

denoised = rank.median(denoised, disk(2))
markers = rank.gradient(denoised, disk(3)) < 10
markers = ndi.label(markers)[0]
gradient = rank.gradient(denoised, disk(2))
labels = segmentation.watershed(gradient, markers)


plt.subplot(2,3,1)
plt.imshow(image)

plt.subplot(2,3,2)
plt.imshow(segmentation.mark_boundaries(image, slic))

plt.subplot(2,3,3)
plt.imshow(label2rgb(slic,
                     image,
                     kind = 'avg'))

plt.subplot(2, 3, 4)
plt.imshow(gradient)

plt.subplot(2, 3, 5)
plt.imshow(markers)

plt.subplot(2, 3, 6)
plt.imshow(image)
plt.imshow(labels, alpha=0.3)
print(labels)
# plt.imshow(image)
plt.show()