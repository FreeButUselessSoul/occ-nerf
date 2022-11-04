import os

import torch
import numpy as np
from tqdm import tqdm
import imageio

from dataloader.with_colmap import resize_imgs
from torchvision import models
from torch.nn import functional as F
from utils.comp_ray_dir import comp_ray_dir_cam
from utils.pose_utils import center_poses
from utils.lie_group_helper import convert3x4_4x4
from utils.vgg import Vgg19

def load_imgs(image_dir, num_img_to_load, start, end, skip, load_sorted, load_img):
    img_names = np.array(sorted(os.listdir(image_dir)))  # all image names

    # down sample frames in temporal domain
    if end == -1:
        img_names = img_names[start::skip]
    else:
        img_names = img_names[start:end:skip]

    if not load_sorted:
        np.random.shuffle(img_names)

    # load images after down sampled
    if num_img_to_load > len(img_names):
        print('Asked for {0:6d} images but only {1:6d} available. Exit.'.format(num_img_to_load, len(img_names)))
        exit()
    elif num_img_to_load == -1:
        print('Loading all available {0:6d} images'.format(len(img_names)))
    else:
        print('Loading {0:6d} images out of {1:6d} images.'.format(num_img_to_load, len(img_names)))
        img_names = img_names[:num_img_to_load]

    img_paths = [os.path.join(image_dir, n) for n in img_names]
    N_imgs = len(img_paths)

    img_list = []
    if load_img:
        for p in tqdm(img_paths):
            img = imageio.imread(p)[:, :, :3]  # (H, W, 3) np.uint8
            img_list.append(img)
        img_list = np.stack(img_list)  # (N, H, W, 3)
        img_list = torch.from_numpy(img_list).float() / 255  # (N, H, W, 3) torch.float32
        H, W = img_list.shape[1], img_list.shape[2]
    else:
        tmp_img = imageio.imread(img_paths[0])  # load one image to get H, W
        H, W = tmp_img.shape[0], tmp_img.shape[1]

    result = {
        'imgs': img_list,  # (N, H, W, 3) torch.float32
        'img_names': img_names,  # (N, )
        'N_imgs': N_imgs,
        'H': H,
        'W': W,
    }
    return result

def read_meta(in_dir, use_ndc):
    """
    Read the poses_bounds.npy file produced by LLFF imgs2poses.py.
    This function is modified from https://github.com/kwea123/nerf_pl.
    """
    poses_bounds = np.load(os.path.join(in_dir, '../poses_bounds.npy'))  # (N_images, 17)

    c2ws = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    bounds = poses_bounds[:, -2:]  # (N_images, 2)
    H, W, focal = c2ws[0, :, -1]

    # correct c2ws: original c2ws has rotation in form "down right back", change to "right up back".
    # See https://github.com/bmild/nerf/issues/34
    c2ws = np.concatenate([c2ws[..., 1:2], -c2ws[..., :1], c2ws[..., 2:4]], -1)

    # (N_images, 3, 4), (4, 4)
    c2ws, pose_avg = center_poses(c2ws)  # pose_avg @ c2ws -> centred c2ws

    if use_ndc:
        # correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        bounds /= scale_factor
        c2ws[..., 3] /= scale_factor
    
    c2ws = convert3x4_4x4(c2ws)  # (N, 4, 4)

    results = {
        'c2ws': c2ws,       # (N, 4, 4) np
        'bounds': bounds,   # (N_images, 2) np
        'H': int(H),        # scalar
        'W': int(W),        # scalar
        'focal': focal,     # scalar
        'pose_avg': pose_avg,  # (4, 4) np
    }
    return results



class Dataloader_feature_n_colmap:
    """
    Most useful fields:
        self.c2ws:          (N_imgs, 4, 4)      torch.float32
        self.imgs           (N_imgs, H, W, 4)   torch.float32
        self.ray_dir_cam    (H, W, 3)           torch.float32
        self.H              scalar
        self.W              scalar
        self.N_imgs         scalar
    """
    def __init__(self, base_dir, scene_name, res_ratio, num_img_to_load, start=0, end=-1, skip=1,
                 load_sorted=True, load_img=True,use_ndc=True, device='cpu'):
        """
        :param base_dir:
        :param scene_name:
        :param res_ratio:       int [1, 2, 4] etc to resize images to a lower resolution.
        :param start/end/skip:  control frame loading in temporal domain.
        :param load_sorted:     True/False.
        :param load_img:        True/False. If set to false: only count number of images, get H and W,
                                but do not load imgs. Useful when vis poses or debug etc.
        """
        self.base_dir = base_dir
        self.scene_name = scene_name
        self.res_ratio = res_ratio
        self.num_img_to_load = num_img_to_load
        self.start = start
        self.end = end
        self.skip = skip
        self.use_ndc = use_ndc
        self.load_sorted = load_sorted
        self.load_img = load_img

        self.imgs_dir = os.path.join(self.base_dir, self.scene_name)
        
        # all meta info
        meta = read_meta(self.imgs_dir, self.use_ndc)
        self.c2ws = torch.Tensor(meta['c2ws'])  # (N, 4, 4) all camera pose
        self.focal = float(meta['focal'])
        if self.end==-1:
            self.c2ws = self.c2ws[self.start::self.skip]
        else:
            self.c2ws = self.c2ws[self.start:self.end:self.skip]
        # self.total_N_imgs = self.c2ws.shape[0]
        image_data = load_imgs(self.imgs_dir, self.num_img_to_load, self.start, self.end, self.skip,
                                self.load_sorted, self.load_img)
        self.imgs = image_data['imgs']  # (N, H, W, 3) torch.float32
        self.img_names = image_data['img_names']  # (N, )
        self.N_imgs = image_data['N_imgs']
        self.ori_H = image_data['H']
        self.ori_W = image_data['W']
        self.encoder = Vgg19().to(device)

        # always use ndc
        self.near = 0.0
        self.far = 1.0

        if self.res_ratio > 1:
            self.H = self.ori_H // self.res_ratio
            self.W = self.ori_W // self.res_ratio
        else:
            self.H = self.ori_H
            self.W = self.ori_W
        self.focal /= self.res_ratio

        if self.load_img:
            self.imgs = resize_imgs(self.imgs, self.H, self.W).to(device)  # (N, H, W, 3) torch.float32
            self.features = []
            for img in tqdm(self.imgs):
                self.features.append(self.encoder(img.permute(2,0,1)[None,...]))
            # self.features = torch.cat(self.features,0)
            # print(self.features.shape)

if __name__ == '__main__':
    base_dir = '/your/data/path'
    scene_name = 'LLFF/fern/images'
    resize_ratio = 8
    num_img_to_load = -1
    start = 0
    end = -1
    skip = 1
    load_sorted = True
    load_img = True
    use_ndc=True

    scene = Dataloader_feature_n_colmap(base_dir=base_dir,
                                scene_name=scene_name,
                                res_ratio=resize_ratio,
                                num_img_to_load=num_img_to_load,
                                start=start,
                                end=end,
                                skip=skip,
                                load_sorted=load_sorted,
                                load_img=load_img,
                                use_ndc=use_ndc)
