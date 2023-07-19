import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import imageio

from utils.comp_ray_dir import comp_ray_dir_cam
from utils.pose_utils import center_poses
from utils.lie_group_helper import convert3x4_4x4


def resize_imgs(imgs, new_h, new_w):
    """
    :param imgs:    (N, H, W, 3)            torch.float32 RGB
    :param new_h:   int/torch int
    :param new_w:   int/torch int
    :return:        (N, new_H, new_W, 3)    torch.float32 RGB
    """
    imgs = imgs.permute(0, 3, 1, 2)  # (N, 3, H, W)
    imgs = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear')  # (N, 3, new_H, new_W)
    imgs = imgs.permute(0, 2, 3, 1)  # (N, new_H, new_W, 3)

    return imgs  # (N, new_H, new_W, 3) torch.float32 RGB


def load_imgs(image_dir, img_ids, new_h, new_w):
    img_names = np.array(sorted(os.listdir(image_dir)))  # all image names
    img_names = img_names[img_ids]  # image name for this split

    img_paths = [os.path.join(image_dir, n) for n in img_names]

    img_list = []
    for p in tqdm(img_paths):
        img = imageio.imread(p)[:, :, :3]  # (H, W, 3) np.uint8
        img_list.append(img)
    img_list = np.stack(img_list)  # (N, H, W, 3)
    img_list = torch.from_numpy(img_list).float() / 255  # (N, H, W, 3) torch.float32
    img_list = resize_imgs(img_list, new_h, new_w)
    return img_list, img_names


def read_meta(in_dir, use_ndc):
    """
    Read the poses_bounds.npy file produced by LLFF imgs2poses.py.
    This function is modified from https://github.com/kwea123/nerf_pl.
    """
    poses_bounds = np.load(os.path.join(in_dir, 'poses_bounds.npy'))  # (N_images, 17)

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


class DataLoaderWithCOLMAP:
    """
    Most useful fields:
        self.c2ws:          (N_imgs, 4, 4)      torch.float32
        self.imgs           (N_imgs, H, W, 4)   torch.float32
        self.ray_dir_cam    (H, W, 3)           torch.float32
        self.H              scalar
        self.W              scalar
        self.N_imgs         scalar
    """
    def __init__(self, base_dir, scene_name, data_type, res_ratio, num_img_to_load, skip, use_ndc, load_img=True):
        """
        :param base_dir:
        :param scene_name:
        :param data_type:   'train' or 'val'.
        :param res_ratio:   int [1, 2, 4] etc to resize images to a lower resolution.
        :param num_img_to_load/skip: control frame loading in temporal domain.
        :param use_ndc      True/False, just centre the poses and scale them.
        :param load_img:    True/False. If set to false: only count number of images, get H and W,
                            but do not load imgs. Useful when vis poses or debug etc.
        """
        self.base_dir = base_dir
        self.scene_name = scene_name
        self.data_type = data_type
        self.res_ratio = res_ratio
        self.num_img_to_load = num_img_to_load
        self.skip = skip
        self.use_ndc = use_ndc
        self.load_img = load_img

        self.scene_dir = os.path.join(self.base_dir, self.scene_name)
        self.img_dir = os.path.join(self.scene_dir, 'images')

        # all meta info
        meta = read_meta(self.scene_dir, self.use_ndc)
        self.c2ws = meta['c2ws']  # (N, 4, 4) all camera pose
        self.H = meta['H']
        self.W = meta['W']
        self.focal = float(meta['focal'])

        if self.res_ratio > 1:
            self.H = self.H // self.res_ratio
            self.W = self.W // self.res_ratio
            self.focal /= self.res_ratio

        self.near = 0.0
        self.far = 1.0
        self.imgs, self.img_names = load_imgs(self.img_dir, np.arange(num_img_to_load), self.H, self.W)  # (N, H, W, 3) torch.float32
        self.c2ws = self.c2ws[:num_img_to_load]
        self.N_imgs = self.c2ws.shape[0]

        # generate cam ray dir.
        self.ray_dir_cam = comp_ray_dir_cam(self.H, self.W, self.focal)  # (H, W, 3) torch.float32

        # convert np to torch.
        self.c2ws = torch.from_numpy(self.c2ws).float()  # (N, 4, 4) torch.float32
        self.ray_dir_cam = self.ray_dir_cam.float()  # (H, W, 3) torch.float32


if __name__ == '__main__':
    scene_name = 'LLFF/fern'
    use_ndc = True
    scene = DataLoaderWithCOLMAP(base_dir='/your/data/path',
                                 scene_name=scene_name,
                                 data_type='train',
                                 res_ratio=8,
                                 num_img_to_load=-1,
                                 skip=1,
                                 use_ndc=use_ndc)
