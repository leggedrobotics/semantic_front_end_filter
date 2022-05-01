import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import msgpack
import msgpack_numpy as m

m.patch()

class AnymalImageDataset(Dataset):
    def __init__(self, msgpack_dir: str, scale: float = 1.0):
        expnames = glob(os.path.join(msgpack_dir, '*'))
        names = [n for en in expnames for n in glob(os.path.join(msgpack_dir, en, '*datum*.msgpack'))]
        self.images = []
        self.depth = []
        self.scale = scale
        for data_name in names:
            with open(data_name, "rb") as data_file:
                byte_data = data_file.read()
                data = msgpack.unpackb(byte_data)
                
                if("cam4" in data["images"].keys() and "cam4depth" in data["images"].keys() ):
                    self.images.append(np.moveaxis(data["images"]["cam4"], 0, 2))
                    self.depth.append(np.array(data["images"]["cam4depth"])[0])

    def __len__(self):
        return len(self.images)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    def __getitem__(self, idx):

        # img = Image.fromarray((self.images[idx]).astype(np.uint8))
        # depth = Image.fromarray((self.depth[idx]*255).astype(np.uint8))
        # assert img.size[:2] == depth.size[:2], \
        #     f'Image and depth should be the same size, but are {img.size} and {depth.size}'

        # img = self.preprocess(img, self.scale, is_mask=False)
        # depth = self.preprocess(depth, self.scale, is_mask=True)

        img = self.images[idx]/255
        img = np.moveaxis(img, 2, 0)
        depth = self.depth[idx]

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'depth': torch.as_tensor(depth.copy()).float().contiguous()
        }

