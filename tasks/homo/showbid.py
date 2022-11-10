import sys
import os
import argparse
from pathlib import Path

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import imageio
from torch.nn import functional as F

sys.path.append(os.path.join(sys.path[0], '../..'))

from dataloader.with_feature_colmap import Dataloader_feature_n_colmap
from utils.training_utils import set_randomness, load_ckpt_to_net
from utils.pose_utils import create_spiral_poses
from utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from utils.lie_group_helper import convert3x4_4x4
from models.nerf_models import fullNeRF,OfficialNerf
from tasks.homo.train_maskDL_long import model_render_image,render_back
from models.intrinsics import LearnFocal
from models.poses import LearnPose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--multi_gpu',  default=False, action='store_true')
    parser.add_argument('--base_dir', type=str, default='./data_dir/nerfmm_release_data',
                        help='folder contains various scenes')
    parser.add_argument('--scene_name', type=str, default='LLFF/fern')
    parser.add_argument('--use_ndc', type=bool, default=True)

    parser.add_argument('--learn_focal', default=False, type=bool)
    parser.add_argument('--fx_only', default=False, type=eval, choices=[True, False])
    parser.add_argument('--focal_order', default=2, type=int)

    parser.add_argument('--learn_R', default=False, type=bool)
    parser.add_argument('--learn_t', default=False, type=bool)

    parser.add_argument('--init_focal_colmap', default=False, type=bool)
    parser.add_argument('--N_importance', type=int, default=0, help='number samples along a ray')

    parser.add_argument('--resize_ratio', type=int, default=4, help='lower the image resolution with this ratio')
    parser.add_argument('--num_rows_eval_img', type=int, default=10, help='split a high res image to rows in eval')
    parser.add_argument('--hidden_dims', type=int, default=128, help='network hidden unit dimensions')
    parser.add_argument('--num_sample', type=int, default=128, help='number samples along a ray')

    parser.add_argument('--pos_enc_levels', type=int, default=10, help='number of freqs for positional encoding')
    parser.add_argument('--pos_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--use_dir_enc', type=bool, default=True, help='use pos enc for view dir?')
    parser.add_argument('--dir_enc_levels', type=int, default=4, help='number of freqs for positional encoding')
    parser.add_argument('--dir_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--spiral_mag_percent', type=float, default=50, help='for np.percentile')
    parser.add_argument('--spiral_axis_scale', type=float, default=[1.0, 1.0, 1.0], nargs=3,
                        help='applied on top of percentile, useful in zoom in motion')
    parser.add_argument('--N_img_per_circle', type=int, default=60)
    parser.add_argument('--N_circle_traj', type=int, default=2)

    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train, -1 for all')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')

    parser.add_argument('--ckpt_dir', type=str, default='')
    return parser.parse_args()


def test_one_epoch(H, W, focal_net, c2ws, near, far, model,model_back,occlusion_net, my_devices, args):
    model.eval()
    focal_net.eval()

    fxfy = focal_net(0)
    ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
    t_vals = torch.linspace(near, far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    # t_vals = 1/(1/(near+1e-15) * (1 - t_steps) + 1/far * t_steps)
    N_img = c2ws.shape[0]

    rendered_img_list = []
    rendered_depth_list = []
    mask_list = []
    back_img_list = []
    back_depth_list = []
    bid_list = []

    for i in tqdm(range(N_img)):
        c2w = c2ws[i].to(my_devices)  # (4, 4)

        # split an image to rows when the input image resolution is high
        rays_dir_cam_split_rows = ray_dir_cam.split(args.num_rows_eval_img, dim=0)
        rendered_img = []
        rendered_depth = []
        mask = []
        back_img = []
        bid = []
        back_depth = []

        for rays_dir_rows in rays_dir_cam_split_rows:
            render_result = model_render_image(c2w, rays_dir_rows, t_vals, near, far, H, W, fxfy,
                                               model, False, 0.0,args,rgb_act_fn=torch.sigmoid,istrain=False,progress=1)
            rgb_rendered_rows = render_result['rgb']  # (num_rows_eval_img, W, 3)
            depth_map = render_result['depth_map']  # (num_rows_eval_img, W)
            dens = render_result['weight'].clone().detach()
            bid.append((render_result['depth_reverse']-render_result['depth_map']).clone().detach())
            dens = torch.cat([dens, render_result['depth_map'].unsqueeze(-1).clone().detach(), render_result['depth_reverse'].unsqueeze(-1).clone().detach()], -1)
            # dens = render_result['rgb_density'][...,-1].clone().detach()
            # dens = F.normalize(dens,dim=-1)
            # dir_ = get_ray_dir(c2w, rays_dir_rows, t_vals, scene_train.H, scene_train.W, fxfy)
            # mask = occlusion_net(encode_position(dir_,2,True)).reshape(depth_map.shape[0],depth_map.shape[1],1)
            mask_row = occlusion_net(dens).squeeze(-1)
            back_results = render_back(c2w, rays_dir_rows,
            #  torch.linspace(near, far, args.num_sample, device=my_devices),
            1/(1/(near+1e-15) * (1 - t_vals) + 1/far * t_vals),
              near, far, H, W, fxfy,
                                               model_back, False, 0.0, args, rgb_act_fn=torch.sigmoid,progress=1)
            rendered_img.append(rgb_rendered_rows)
            rendered_depth.append(depth_map)
            mask.append(mask_row)
            back_img.append(back_results['rgb_fine'] if 'rgb_fine' in back_results.keys() else back_results['rgb'])
            back_depth.append(back_results['depth_fine'] if 'depth_fine' in back_results.keys() else back_results['depth_map'])
            
        # combine rows to an image
        rendered_img = torch.cat(rendered_img, dim=0)  # (H, W, 3)
        rendered_depth = torch.cat(rendered_depth, dim=0)  # (H, W)
        mask = torch.cat(mask,0) # (H,W)
        back_img = torch.cat(back_img, 0) # (H, W, 3)
        back_depth = torch.cat(back_depth, 0) # (H, W)
        bid = torch.cat(bid, 0)

        # for vis
        rendered_img_list.append(rendered_img)
        rendered_depth_list.append(rendered_depth)
        mask_list.append(mask)
        back_img_list.append(back_img)
        back_depth_list.append(back_depth)
        bid_list.append(bid)

    rendered_img_list = torch.stack(rendered_img_list)  # (N, H, W, 3)
    rendered_depth_list = torch.stack(rendered_depth_list)  # (N, H, W)
    mask_list = torch.stack(mask_list)
    back_img_list = torch.stack(back_img_list)
    back_depth_list = torch.stack(back_depth_list)
    bid_list = torch.stack(bid_list)

    result = {
        'imgs': rendered_img_list,
        'depths': rendered_depth_list,
        'bg': back_img_list,
        'bg_depth': back_depth_list,
        'mask': mask_list,
        'bid': bid_list,
    }
    return result


def main(args):
    my_devices = torch.device('cuda:' + str(args.gpu_id))

    '''Create Folders'''
    temp_scene_name = args.scene_name.replace('/','_')
    test_dir = Path(os.path.join('results/',temp_scene_name))
    img_out_dir = Path(os.path.join(test_dir, 'img_out'))
    depth_out_dir = Path(os.path.join(test_dir, 'depth_out'))
    bg_img_out_dir = Path(os.path.join(test_dir, 'bg_img_out'))
    bg_depth_out_dir = Path(os.path.join(test_dir, 'bg_depth_out'))
    mask_dir = Path(os.path.join(test_dir, 'mask_out'))
    video_out_dir = Path(os.path.join(test_dir, 'video_out'))
    
    test_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir.mkdir(parents=True, exist_ok=True)
    bg_img_out_dir.mkdir(parents=True,exist_ok=True)
    bg_depth_out_dir.mkdir(parents=True,exist_ok=True)
    mask_dir.mkdir(parents=True,exist_ok=True)
    video_out_dir.mkdir(parents=True, exist_ok=True)

    '''Scene Meta'''
    scene_train = Dataloader_feature_n_colmap(base_dir=args.base_dir,
                                       scene_name=args.scene_name,
                                    #    data_type='train',
                                       res_ratio=args.resize_ratio,
                                       num_img_to_load=args.train_img_num,
                                       skip=args.train_skip,
                                       use_ndc=args.use_ndc,
                                       load_img=False)

    H, W = scene_train.H, scene_train.W
    colmap_focal = scene_train.focal
    near, far = scene_train.near, scene_train.far

    print('Intrinsic: H: {0:4d}, W: {1:4d}, COLMAP focal {2:.2f}.'.format(H, W, colmap_focal))
    print('near: {0:.1f}, far: {1:.1f}.'.format(near, far))

    '''Model Loading'''
    pos_enc_in_dims = (2 * args.pos_enc_levels + int(args.pos_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    if args.use_dir_enc:
        dir_enc_in_dims = (2 * args.dir_enc_levels + int(args.dir_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    else:
        dir_enc_in_dims = 0

    model = fullNeRF(pos_enc_in_dims, dir_enc_in_dims, args.hidden_dims,6,[3])
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device=my_devices)
    else:
        model = model.to(device=my_devices)
    load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_nerf.pth'), model, map_location=my_devices)

    model_back = fullNeRF(pos_enc_in_dims, dir_enc_in_dims, args.hidden_dims)
    if args.multi_gpu:
        model_back = torch.nn.DataParallel(model_back).to(device=my_devices)
    else:
        model_back = model_back.to(device=my_devices)
    load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_nerfback.pth'), model_back, map_location=my_devices)

    if args.init_focal_colmap:
        focal_net = LearnFocal(H, W, args.learn_focal, args.fx_only, order=args.focal_order, init_focal=colmap_focal)
    else:
        focal_net = LearnFocal(H, W, args.learn_focal, args.fx_only, order=args.focal_order)
    if args.multi_gpu:
        focal_net = torch.nn.DataParallel(focal_net).to(device=my_devices)
    else:
        focal_net = focal_net.to(device=my_devices)
    # do not load learned focal if we use colmap focal
    if not args.init_focal_colmap:
        load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_focal.pth'), focal_net, map_location=my_devices)
    fxfy = focal_net(0)
    print('COLMAP focal: {0:.2f}, learned fx: {1:.2f}, fy: {2:.2f}'.format(colmap_focal, fxfy[0].item(), fxfy[1].item()))

    if args.init_focal_colmap:
        pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, scene_train.c2ws)
    else:
        pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, None)
    if args.multi_gpu:
        pose_param_net = torch.nn.DataParallel(pose_param_net).to(device=my_devices)
    else:
        pose_param_net = pose_param_net.to(device=my_devices)
    load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_pose.pth'), pose_param_net, map_location=my_devices,strict=False)

    occlusion_net = nn.Sequential(
        nn.Linear(args.num_sample+2,64),nn.LeakyReLU(0.1),
        nn.Linear(64,32),nn.LeakyReLU(0.1),
        nn.Linear(32,1),nn.Sigmoid(),
    )
    if args.multi_gpu:
        occlusion_net = torch.nn.DataParallel(occlusion_net).to(device=my_devices)
    else:
        occlusion_net = occlusion_net.to(device=my_devices)
    load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_mask.pth'), occlusion_net, map_location=my_devices)

    learned_poses = torch.stack([pose_param_net(i) for i in range(scene_train.N_imgs)])
    print("All Models Loaded.")
    '''Generate camera traj'''
    # This spiral camera traj code is modified from https://github.com/kwea123/nerf_pl.
    # hardcoded, this is numerically close to the formula
    # given in the original repo. Mathematically if near=1
    # and far=infinity, then this number will converge to 4
    N_novel_imgs = args.N_img_per_circle * args.N_circle_traj
    # focus_depth = 3.5
    # radii = np.percentile(np.abs(learned_poses.cpu().numpy()[:, :3, 3]), args.spiral_mag_percent, axis=0)  # (3,)
    # radii *= np.array(args.spiral_axis_scale)
    # c2ws = create_spiral_poses(radii, focus_depth, n_circle=args.N_circle_traj, n_poses=N_novel_imgs)
    N_frames = 60
    radii = np.linspace(0, np.pi*2, N_frames)
    # dx = np.linspace(0, 0.3, N_frames)
    # dy = np.linspace(0, 0, N_frames)
    dx = np.cos(radii)*0.2
    dy = np.linspace(0, 0, N_frames)#np.sin(radii)*0.5
    # dz = np.linspace(0, 0, N_frames)
    # define poses
    # dataset.poses_test = dataset.poses
    c2ws = np.tile(np.array([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]],dtype=np.float32), (N_frames//2, 1, 1))
    for i in range(N_frames//2):
        c2ws[i, 0, 3] += dx[i]
        c2ws[i, 1, 3] += dy[i]
    c2ws = torch.from_numpy(c2ws).float()  # (N, 3, 4)
    c2ws = convert3x4_4x4(c2ws)  # (N, 4, 4)

    '''Render'''
    result = test_one_epoch(H, W, focal_net, c2ws, near, far, model,model_back,occlusion_net, my_devices, args)
    imgs = result['imgs']
    depths = result['depths']

    '''Write to folder'''
    imgs = (imgs.cpu().numpy() * 255).astype(np.uint8)
    depths = (depths.cpu().numpy() * 255).astype(np.uint8)  # far is 1.0 in NDC # 200
    bg_imgs = (result['bg'].cpu().numpy() * 255).astype(np.uint8)
    bg_depth = (result['bg_depth'].cpu().numpy() * 255).astype(np.uint8)
    mask = ((result['mask']<0.5).cpu().numpy() * 255).astype(np.uint8)
    bid = (result['bid'].cpu().numpy()*255).astype(np.uint8)

    for i in range(c2ws.shape[0]):
        imageio.imwrite(os.path.join(img_out_dir, str(i).zfill(4) + '.jpg'), imgs[i])
        imageio.imwrite(os.path.join(depth_out_dir, str(i).zfill(4) + '.jpg'), depths[i])
        imageio.imwrite(os.path.join(bg_img_out_dir, str(i).zfill(4) + '.jpg'), bg_imgs[i])
        imageio.imwrite(os.path.join(bg_depth_out_dir, str(i).zfill(4) + '.jpg'), bg_depth[i])
        imageio.imwrite(os.path.join(mask_dir, str(i).zfill(4) + '.jpg'), mask[i])
        imageio.imwrite(os.path.join(depth_out_dir, 'rev-'+str(i).zfill(4)+'.jpg'),bid[i] )

    # imageio.mimwrite(os.path.join(video_out_dir, 'img.mp4'), imgs, fps=30, quality=9)
    # imageio.mimwrite(os.path.join(video_out_dir, 'depth.mp4'), depths, fps=30, quality=9)
    # imageio.mimwrite(os.path.join(video_out_dir, 'bg_img.mp4'), bg_imgs, fps=30, quality=9)
    # imageio.mimwrite(os.path.join(video_out_dir, 'bg_depth.mp4'), bg_depth, fps=30, quality=9)
    # imageio.mimwrite(os.path.join(video_out_dir, 'mask.mp4'), mask, fps=30, quality=9)

    # imageio.mimwrite(os.path.join(video_out_dir, 'img.gif'), imgs, fps=30)
    # imageio.mimwrite(os.path.join(video_out_dir, 'depth.gif'), depths, fps=30)
    # imageio.mimwrite(os.path.join(video_out_dir, 'bg_img.gif'), bg_imgs, fps=30)
    # imageio.mimwrite(os.path.join(video_out_dir, 'bg_depth.gif'), bg_depth, fps=30)
    # imageio.mimwrite(os.path.join(video_out_dir, 'mask.gif'), mask, fps=30)

    return


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    with torch.no_grad():
        main(args)
