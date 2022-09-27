import sys
import os
import argparse

sys.path.append(os.path.join(sys.path[0], '../..'))

from utils.vis_cam_traj import draw_camera_frustum_geometry

import torch
import numpy as np

from dataloader.any_folder import DataLoaderAnyFolder
from utils.training_utils import set_randomness, load_ckpt_to_net
from models.intrinsics import LearnFocal
from models.poses import LearnPose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='./data_dir/nerfmm_release_data')
    parser.add_argument('--scene_name', type=str, default='any_folder_demo/desk')

    parser.add_argument('--learn_focal', default=False, type=bool)
    parser.add_argument('--focal_order', default=2, type=int)
    parser.add_argument('--fx_only', default=False, type=eval, choices=[True, False])

    parser.add_argument('--learn_R', default=False, type=bool)
    parser.add_argument('--learn_t', default=False, type=bool)

    parser.add_argument('--resize_ratio', type=int, default=8, help='lower the image resolution with this ratio')

    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train')
    parser.add_argument('--train_load_sorted', type=bool, default=False)
    parser.add_argument('--train_start', type=int, default=0, help='inclusive')
    parser.add_argument('--train_end', type=int, default=-1, help='exclusive, -1 for all')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')

    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--ckpt_dir', type=str, default='')
    return parser.parse_args()


def main(args):
    my_devices = torch.device('cpu')

    scene_train = DataLoaderAnyFolder(base_dir=args.base_dir,
                                      scene_name=args.scene_name,
                                      res_ratio=args.resize_ratio,
                                      num_img_to_load=args.train_img_num,
                                      start=args.train_start,
                                      end=args.train_end,
                                      skip=args.train_skip,
                                      load_sorted=args.train_load_sorted,
                                      load_img=False)

    print('H: {0:4d}, W: {1:4d}.'.format(scene_train.H, scene_train.W))

    '''Model Loading'''
    focal_net = LearnFocal(scene_train.H, scene_train.W, args.learn_focal, args.fx_only, order=args.focal_order)
    focal_net = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_focal.pth'), focal_net, map_location=my_devices)

    pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, None)
    pose_param_net = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_pose.pth'), pose_param_net, map_location=my_devices)

    '''Get focal'''
    fxfy = focal_net(0)
    if not args.fx_only:
        fxfy = torch.mean(fxfy)
    else:
        fxfy = fxfy[0].item()
    print('learned focal: {0:.2f}'.format(fxfy))

    '''Get all poses in (N, 4, 4)'''
    c2ws_est = torch.stack([pose_param_net(i) for i in range(scene_train.N_imgs)])[:,:3]  # (N, 3, 4)
    last_col = torch.Tensor([scene_train.H,scene_train.W,fxfy])[None,:,None].repeat(len(c2ws_est),1,1)
    poses = torch.cat([c2ws_est,last_col],-1).flatten(1,2) # (N,3,5) -> (N,15)
    last_col = torch.Tensor([0.,1.])[None,:].repeat(len(c2ws_est),1)
    poses_bounds = torch.cat([poses,last_col],-1) # (N,17)
    print(poses_bounds)
    np.save(os.path.join(args.base_dir,args.scene_name,'../poses_bounds.npy'),poses_bounds)
    

if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    with torch.no_grad():
        main(args)
