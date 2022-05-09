# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications

import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import msgpack
import msgpack_numpy as m

m.patch()

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class DepthDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and perform/report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        # if mode == 'online_eval':
        #     with open(args.filenames_file_eval, 'r') as f:
        #         self.filenames = f.readlines()
        # else:
        #     with open(args.filenames_file, 'r') as f:
        #         self.filenames = f.readlines()
        self.filenames = []
        import os
        if("TMPDIR" in os.environ.keys()):
            args.data_path = os.path.join(os.environ["TMPDIR"], args.data_path)
        print("data_path",args.data_path)
        for root, dirs, files in os.walk(args.data_path):
            for file in files:
                if file.startswith('traj') and file.endswith('.msgpack'):
                    # print("loading file: %s"%file, end =" ")
                    sample_path = os.path.join(root,file)
                    # with open(sample_path, "rb") as data_file:
                    #     byte_data = data_file.read()
                    #     data = msgpack.unpackb(byte_data)
                    # if("images" in data.keys()):
                    self.filenames.append(sample_path)
                        # print("success")
                    # else:
                        # print("empty")
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
        random.Random(0).shuffle(self.filenames)
        self.filenames = self.filenames[:-100] if self.mode == 'train' else self.filenames[-100:]

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]

        # focal = float(sample_path.split()[2])
        focal = 0

        if self.mode == 'train':
            # image_path = os.path.join(self.args.data_path, remove_leading_slash(sample_path.split()[3]))
            # depth_path = os.path.join(self.args.gt_path, remove_leading_slash(sample_path.split()[4]))
            # image = Image.open(image_path)
            # depth_gt = Image.open(depth_path)
            with open(sample_path, "rb") as data_file:
                byte_data = data_file.read()
                data = msgpack.unpackb(byte_data)
            image = Image.fromarray(np.moveaxis(data["images"]["cam4"].astype(np.uint8), 0, 2))
            depth_gt = np.moveaxis(data["images"]["cam4depth"],0,2)


            pc_image = np.zeros_like(depth_gt)
            pos = data["pose"]["map"][:3]
            pc = data["pointcloud"]
            pc_distance = np.sqrt(np.sum((pc[:,:3] - pos)**2, axis = 1))

            imgshape = pc_image.shape[:-1] 
            pc_proj_mask = pc[:, 10] > 0.5 # the point is on the graph
            pc_proj_loc = pc[:, 11:13] # the x,y pos of point on image
            pc_proj_mask = (pc_proj_mask & (pc_proj_loc[:, 0]<imgshape[1])
                                        & (pc_proj_loc[:, 0]>=0)
                                        &  (pc_proj_loc[:, 1]<imgshape[0])
                                        &  (pc_proj_loc[:, 1]>=0))
            pc_proj_loc = pc_proj_loc[pc_proj_mask].astype(np.int32)
            pc_distance = pc_distance[pc_proj_mask]
            pc_image[pc_proj_loc[:,1], pc_proj_loc[:,0], 0] = pc_distance

            # if self.args.do_kb_crop is True:
            #     height = image.height
            #     width = image.width
            #     top_margin = int(height - 352)
            #     left_margin = int((width - 1216) / 2)
            #     depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            #     image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # # To avoid blank boundaries due to pixel registration
            # if self.args.dataset == 'nyu':
            #     depth_gt = depth_gt.crop((43, 45, 608, 472))
            #     image = image.crop((43, 45, 608, 472))

            # if self.args.do_random_rotate is True:
            #     random_angle = (random.random() - 0.5) * 2 * self.args.degree
            #     image = self.rotate_image(image, random_angle)
            #     depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            pc_image = np.asarray(pc_image, dtype=np.float32)
            
            # print("shapes", image.shape, depth_gt.shape)
            # depth_gt = np.expand_dims(depth_gt, axis=2)


            # if self.args.dataset == 'nyu':
            #     depth_gt = depth_gt / 1000.0
            # else:
            #     depth_gt = depth_gt / 256.0
            image, depth_gt, pc_image = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width, pc_image)
            image, depth_gt, pc_image = self.train_preprocess(image, depth_gt, pc_image)
            image = np.concatenate((image, pc_image[:, :, 0:1]), axis=2)
            depth_gt_mean = depth_gt[:, :, 0:1]
            depth_gt_variance = depth_gt[:, :, 1:]
            pc_image = pc_image[:, :, 0:1]
            sample = {'image': image.copy(), 'depth': depth_gt_mean.copy(), 'pc_image': pc_image.copy(), 'focal': focal, 'depth_variance': depth_gt_variance.copy()}

        else:
            # if self.mode == 'online_eval':
            #     data_path = self.args.data_path_eval
            # else:
            #     data_path = self.args.data_path

            # image_path = os.path.join(data_path, remove_leading_slash(sample_path.split()[0]))
            # image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            with open(sample_path, "rb") as data_file:
                byte_data = data_file.read()
                data = msgpack.unpackb(byte_data)
            image = Image.fromarray(np.moveaxis(data["images"]["cam4"].astype(np.uint8), 0, 2))
            depth_gt = np.moveaxis(data["images"]["cam4depth"],0,2)
            image = np.asarray(image, dtype=np.float32) / 255.0
            
            #  pc image
            pc_image = np.zeros_like(depth_gt)
            pos = data["pose"]["map"][:3]
            pc = data["pointcloud"]
            pc_distance = np.sqrt(np.sum((pc[:,:3] - pos)**2, axis = 1))

            imgshape = pc_image.shape[:-1] 
            pc_proj_mask = pc[:, 10] > 0.5 # the point is on the graph
            pc_proj_loc = pc[:, 11:13] # the x,y pos of point on image
            pc_proj_mask = (pc_proj_mask & (pc_proj_loc[:, 0]<imgshape[1])
                                        & (pc_proj_loc[:, 0]>=0)
                                        &  (pc_proj_loc[:, 1]<imgshape[0])
                                        &  (pc_proj_loc[:, 1]>=0))
            pc_proj_loc = pc_proj_loc[pc_proj_mask].astype(np.int32)
            pc_distance = pc_distance[pc_proj_mask]
            pc_image[pc_proj_loc[:,1], pc_proj_loc[:,0], 0] = pc_distance

            image = np.concatenate((image, pc_image[:, :, 0:1]), axis=2)
            depth_gt_mean = depth_gt[:, :, 0:1]
            depth_gt_variance = depth_gt[:, :, 1:]
            pc_image = pc_image[:, :, 0:1]
            pc_image = np.asarray(pc_image, dtype=np.float32)

            if self.mode == 'online_eval':
                # gt_path = self.args.gt_path_eval
                # depth_path = os.path.join(gt_path, remove_leading_slash(sample_path.split()[1]))
                # has_valid_depth = False
                # try:
                    # depth_gt = Image.open(depth_path)
                    has_valid_depth = True
                # except IOError:
                    # depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                # if has_valid_depth:
                    depth_gt_mean = np.asarray(depth_gt_mean, dtype=np.float32)
                    # depth_gt = np.expand_dims(depth_gt, axis=2)
                    # if self.args.dataset == 'nyu':
                    #     depth_gt = depth_gt / 1000.0
                    # else:
                    #     depth_gt = depth_gt / 256.0

            if self.args.do_kb_crop is True:
                # height = image.shape[0]
                # width = image.shape[1]
                # top_margin = int(height - 352)
                # left_margin = int((width - 1216) / 2)
                # image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                # if self.mode == 'online_eval' and has_valid_depth:
                # depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

                # image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
                # image, depth_gt = self.train_preprocess(image, depth_gt)
                image,depth_gt = image,depth_gt
            if self.mode == 'online_eval':
                sample = {'image': image.copy(), 'depth': depth_gt_mean.copy(), 'focal': focal, 'has_valid_depth': has_valid_depth,
                        #   'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1]}
                          'image_path': sample_path, 'depth_path': sample_path, 'depth_variance': depth_gt_variance.copy()}
            else:
                sample = {'image': image.copy(), 'focal': focal}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width, *args):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        # the args are other images to be cropped to the same size
        retargs = [i[y:y + height, x:x + width, :] for i in args]
        return img, depth, *retargs

    def train_preprocess(self, image, depth_gt, *args):
        # Random flipping
        do_flip = random.random()
        retargs = args
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
            retargs = [(i[:, ::-1, :]).copy() for i in args]
        
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt, *retargs

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.387, 0.394, 0.404, 0.120], std=[0.322, 0.32, 0.30,  1.17])

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        depth_variance = sample['depth_variance']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            depth_variance = self.to_tensor(depth_variance)
            pc_image = sample['pc_image']
            pc_image = self.to_tensor(pc_image)
            return {'image': image, 'depth': depth, "pc_image":pc_image, 'focal': focal, "depth_variance": depth_variance}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth,
                    'image_path': sample['image_path'], 'depth_path': sample['depth_path'], "depth_variance": depth_variance}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
