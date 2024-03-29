import sys
import os
import argparse
from pathlib import Path
import datetime
import shutil
import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from einops import rearrange, reduce, repeat

sys.path.append(os.path.join(sys.path[0], '../..'))

from dataloader.with_feature import DataLoaderAnyFolder
from utils.training_utils import set_randomness, mse2psnr, save_checkpoint, load_ckpt_to_net
from utils.pos_enc import encode_position
from utils.volume_op import volume_sampling_ndc, volume_rendering
from utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from models.nerf_models import OfficialNerf
from models.intrinsics import LearnFocal
from models.poses import LearnPose

from kornia.utils import create_meshgrid

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
    # m.bias.data.fill_(0.01)

def nonzero_var(inputs, dim=0):
    num = torch.sum(inputs!=0,dim)
    # a0 = torch.nan_to_num(inputs,0)
    output = torch.sum((inputs+1e-10)**2,dim)/(num+1e-5) - (torch.sum(inputs+1e-10,dim)/(num+1e-5))**2
    output[num==0] = 0
    return output

def repeat_interleave(input, repeats, dim=0):
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])

def conv(in_planes, out_planes, kernel_size, instancenorm=False):
    if instancenorm:
        m = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.InstanceNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        m = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    return m

@torch.no_grad()
def homo_warp_with_depth(src_feat, proj_mat, depth_values, src_grid=None, ref_g=None, pad=0):
    """
    src_feat: (B, C, H, W)
    proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (B,hw, D)
    out: (B, C, D, H, W)
    """
    if len(src_feat.shape)==5:
        # (B,n,C,H,W)
        depth_values = repeat_interleave(depth_values,src_feat.shape[1],0)
        if (ref_g is not None):
            ref_g = ref_g.repeat(src_feat.shape[1],1,1,1)
        src_feat = src_feat.flatten(0,1)
        proj_mat = proj_mat.flatten(0,1)

    assert(src_grid==None)
    B, C, H, W = src_feat.shape
    device = src_feat.device

    if pad>0:
        H_pad, W_pad = H + pad*2, W + pad*2
    else:
        H_pad, W_pad = H, W

    D = depth_values.shape[-1]
    R = proj_mat[:, :, :3]  # (B, 3, 3)
    T = proj_mat[:, :, 3:]  # (B, 3, 1)

    # create grid from the ref frame
    if ref_g is None:
        ref_grid = create_meshgrid(H_pad, W_pad, normalized_coordinates=False, device=device)  # (1, H, W, 2)
        if pad>0:
            ref_grid -= pad
    else:
        ref_grid = ref_g
        H_pad,W_pad = ref_g.shape[-2:]
        # if (depth_values.shape!=(B,D,H_pad,W_pad)):
        #     depth_values = F.grid_sample(depth_values, ref_g, mode='bilinear',align_corners=True)
    if B==4 and H_pad * W_pad != depth_values.shape[1]:
        import ipdb;ipdb.set_trace()
    # ref_grid = ref_grid.permute(0, 3, 1, 2)  # (B, 2, H, W)
    ref_grid = ref_grid.reshape(B, 2, W_pad * H_pad)  # (B, 2, H*W)
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (B, 3, H*W)
    ref_grid_d = ref_grid.repeat(1, 1, D)  # (B, 3, D*H*W)
    src_grid_d = R @ ref_grid_d + T / depth_values.view(B, 1, D * W_pad * H_pad)
    del ref_grid_d, ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory

    src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:]  # divide by depth (B, 2, D*H*W)
    del src_grid_d
    src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
    src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
    src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
    src_grid = src_grid.view(B, D, W_pad * H_pad, 2)
    warped_src_feat = F.grid_sample(src_feat, src_grid,
                                    mode='nearest', padding_mode='zeros',
                                    align_corners=True)  # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, -1, D, H_pad, W_pad)
    # src_grid = src_grid.view(B, 1, D, H_pad, W_pad, 2)
    return warped_src_feat, src_grid


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10000, type=int)
    parser.add_argument('--eval_interval', default=100, type=int, help='run eval every this epoch number')

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--multi_gpu',  default=False, type=eval, choices=[True, False])
    parser.add_argument('--base_dir', type=str, default='./data_dir/nerfmm_release_data')
    parser.add_argument('--scene_name', type=str, default='any_folder_demo/desk')

    parser.add_argument('--nerf_lr', default=0.001, type=float)

    parser.add_argument('--learn_focal', default=True, type=bool)
    parser.add_argument('--focal_order', default=2, type=int)
    parser.add_argument('--fx_only', default=False, type=eval, choices=[True, False])
    parser.add_argument('--focal_lr', default=0.001, type=float)
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--resume', default=False, action="store_true",
                        help='Resume training')
    parser.add_argument('--ckpt_dir', default=None)

    parser.add_argument('--learn_R', default=True, type=eval, choices=[True, False])
    parser.add_argument('--learn_t', default=True, type=eval, choices=[True, False])
    parser.add_argument('--pose_lr', default=0.001, type=float)

    parser.add_argument('--resize_ratio', type=int, default=4, help='lower the image resolution with this ratio')
    parser.add_argument('--num_rows_eval_img', type=int, default=10, help='split a high res image to rows in eval')
    parser.add_argument('--hidden_dims', type=int, default=128, help='network hidden unit dimensions')
    parser.add_argument('--train_rand_rows', type=int, default=32, help='rand sample these rows to train')
    parser.add_argument('--train_rand_cols', type=int, default=32, help='rand sample these cols to train')
    parser.add_argument('--num_sample', type=int, default=128, help='number samples along a ray')
    parser.add_argument('--N_importance', type=int, default=0, help='number samples along a ray')

    parser.add_argument('--pos_enc_levels', type=int, default=10, help='number of freqs for positional encoding')
    parser.add_argument('--pos_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--use_dir_enc', type=bool, default=True, help='use pos enc for view dir?')
    parser.add_argument('--dir_enc_levels', type=int, default=4, help='number of freqs for positional encoding')
    parser.add_argument('--dir_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train, -1 for all')
    parser.add_argument('--train_load_sorted', type=bool, default=True)
    parser.add_argument('--train_start', type=int, default=0, help='inclusive')
    parser.add_argument('--train_end', type=int, default=-1, help='exclusive, -1 for all')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')

    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')

    parser.add_argument('--alias', type=str, default='', help="experiments alias")
    return parser.parse_args()


def gen_detail_name(args):
    outstr = 'lr_' + str(args.nerf_lr) + \
             '_gpu' + str(args.gpu_id) + \
             '_seed_' + str(args.rand_seed) + \
             '_resize_' + str(args.resize_ratio) + \
             '_Nsam_' + str(args.num_sample) + \
             '_Ntr_img_'+ str(args.train_img_num) + \
             '_freq_' + str(args.pos_enc_levels) + \
             '_' + str(args.alias) + \
             '_' + str(datetime.datetime.now().strftime('%y%m%d_%H%M'))
    return outstr


def model_render_image(c2w, rays_cam, t_vals, near, far, H, W, fxfy, model, perturb_t, sigma_noise_std,
                       args, rgb_act_fn,features=None,proj_mats=None,pixel_i=None,cost_volume_cnn=None,cost_volume=None,istrain=True):
    """Render an image or pixels.
    :param c2w:         (4, 4)                  pose to transform ray direction from cam to world.
    :param rays_cam:    (someH, someW, 3)       ray directions in camera coordinate, can be random selected
                                                rows and cols, or some full rows, or an entire image.
    :param t_vals:      (N_samples)             sample depth along a ray.
    :param fxfy:        a float or a (2, ) torch tensor for focal.
    :param perturb_t:   True/False              whether add noise to t.
    :param sigma_noise_std: a float             std dev when adding noise to raw density (sigma).
    :rgb_act_fn:        sigmoid()               apply an activation fn to the raw rgb output to get actual rgb.
    :return:            (someH, someW, 3)       volume rendered images for the input rays.
    """
    # (H, W, N_sample, 3), (H, W, 3), (H, W, N_sam)
    sample_pos, ray_ori_world, ray_dir_world, t_vals_noisy = volume_sampling_ndc(c2w, rays_cam, t_vals, near, far,
                                                                     H, W, fxfy, perturb_t)
    if cost_volume is None:
        warpped_features,_ = homo_warp_with_depth(torch.stack(features),proj_mats,
                t_vals_noisy.flatten(0,1)[None,...].repeat(len(proj_mats),1,1),
                ref_g = pixel_i.repeat(len(proj_mats),1,1,1)
                ) # (B C D H W)
        temp_shape = list(warpped_features.shape)
        temp_shape[1],temp_shape[2]=temp_shape[2],32
        warpped_features = cost_volume_cnn(warpped_features.transpose(1,2).flatten(0,1)).view(temp_shape).transpose(1,2) # B,C,D,H,W
        cost_volume = torch.var(warpped_features, 0).permute(2,3,1,0) # B,C',D,H,W
    # encode position: (H, W, N_sample, (2L+1)*C = 63)
    pos_enc = encode_position(sample_pos, levels=args.pos_enc_levels, inc_input=args.pos_enc_inc_in)

    # encode direction: (H, W, N_sample, (2L+1)*C = 27)
    if args.use_dir_enc:
        ray_dir_world = F.normalize(ray_dir_world, p=2, dim=2)  # (H, W, 3)
        dir_enc = encode_position(ray_dir_world, levels=args.dir_enc_levels, inc_input=args.dir_enc_inc_in)  # (H, W, 27)
        dir_enc_ = dir_enc.unsqueeze(2).expand(-1, -1, args.num_sample, -1)  # (H, W, N_sample, 27)
    else:
        dir_enc_ = None

    # inference rgb and density using position and direction encoding.
    rgb_density = model(pos_enc, dir_enc_)  # (H, W, N_sample, 4)
    
    # render_result = volume_rendering(rgb_density, t_vals_noisy, sigma_noise_std, rgb_act_fn)
    if istrain:
        render_result = volume_rendering(rgb_density, t_vals_noisy, sigma_noise_std, rgb_act_fn)
    else:
        density = rgb_density[...,-1].clone()
        # deriv = torch.cat([density[:,:,1:]-density[:,:,:-1],torch.zeros_like(density[:,:,[0]])],-1)
        mean_density = 1e-8
        t_maxpos = torch.cummax((density > mean_density)  , -1)[0] # 00011000001111 -> 000111...1
        mask = t_maxpos * (density<mean_density)# * (deriv>0)
        t_maskpos = torch.cummax(mask,-1)[0]
        # density[~t_maskpos] = 0.
        temp_mask = torch.sum(t_maskpos[...,:int(t_maskpos.shape[-1]*0.9)],-1)>0
        density[(temp_mask[...,None].repeat(1,1,t_maskpos.shape[-1])) * (~t_maskpos)] = 0 # rgb_density[...,-1][temp_mask].
        # density (density>mean_density)*t_maskpos 
        # density[temp_mask][...,:int(density.shape[-1]*0.5)] = 0
        # density[~t_maskpos] = 0.
        render_result = volume_rendering(rgb_density, t_vals_noisy, sigma_noise_std, rgb_act_fn,density)

    rgb_rendered = render_result['rgb']  # (H, W, 3)
    depth_map = render_result['depth_map']  # (H, W)

    result = {
            'rgb': rgb_rendered,  # (H, W, 3)
            'sample_pos': sample_pos,  # (H, W, N_sample, 3)
            'depth_map': depth_map,  # (H, W)
            'rgb_density': rgb_density,  # (H, W, N_sample, 4) 
            'weight': render_result['weight'].detach() * torch.mean(cost_volume,-1),
        }

    if not istrain:
        result['bg_rgb'] = render_result['bg_rgb']
        result['bg_depth'] = render_result['bg_depth_map']

    if args.N_importance > 0:
        t_vals_mid = 0.5*(t_vals_noisy[...,:-1]+t_vals_noisy[...,1:])
        t_vals_ = sample_pdf(t_vals_mid.flatten(0,1), render_result['weight'][...,1:-1].detach().flatten(0,1),args.N_importance, det=not perturb_t).view(t_vals_mid.shape[0],t_vals_mid.shape[1],-1)
        t_vals = torch.sort(torch.cat([t_vals_noisy, t_vals_],-1),-1)[0]
        sample_pos = ray_ori_world.unsqueeze(2) + ray_dir_world.unsqueeze(2) * t_vals.unsqueeze(3)
        pos_enc = encode_position(sample_pos, levels=args.pos_enc_levels, inc_input=args.pos_enc_inc_in)
        if args.use_dir_enc:
            dir_enc_ = dir_enc.unsqueeze(2).expand(-1, -1, (args.num_sample+args.N_importance), -1)  # (H, W, N_sample, 27)
        else:
            dir_enc_ = None
        rgb_density_fine = model(pos_enc, dir_enc_)
        render_result_fine = volume_rendering(rgb_density_fine, t_vals, sigma_noise_std, rgb_act_fn)
        rgb_rendered_fine = render_result_fine['rgb']  # (H, W, 3)
        depth_map_fine = render_result_fine['depth_map']  # (H, W)
        result['rgb_fine'] = rgb_rendered_fine
        result['depth_fine'] = depth_map_fine

    return result


def eval_one_epoch(eval_c2ws, scene_train, model, focal_net, pose_param_net, cost_volume_cnn,
                   my_devices, args, epoch_i, depth_path, rgb_act_fn):
    model.eval()
    focal_net.eval()
    pose_param_net.eval()

    fxfy = focal_net(0)
    ray_dir_cam = comp_ray_dir_cam_fxfy(scene_train.H, scene_train.W, fxfy[0], fxfy[1])
    t_steps = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    if args.use_disp:
        t_vals = 1/(1/(scene_train.near+1e-15) * (1 - t_steps) + 1/scene_train.far * t_steps)
    else:
        t_vals = scene_train.near * (1 - t_steps) + scene_train.far * t_steps
    N_img, H, W = eval_c2ws.shape[0], scene_train.H, scene_train.W

    rendered_img_list = []
    rendered_depth_list = []
    bg_img_list = []
    bg_depth_list = []

    for i in tqdm(range(N_img)):
        c2w = eval_c2ws[i].to(my_devices)  # (4, 4)
        K = torch.eye(4).to(fxfy.device)
        K[0, 0] = fxfy[0]
        K[1, 1] = fxfy[1]
        K[0, 2] = scene_train.W/2
        K[1, 2] = scene_train.H/2
        # split an image to rows when the input image resolution is high
        rays_dir_cam_split_rows = ray_dir_cam.split(args.num_rows_eval_img, dim=0)
        rendered_img = []
        rendered_depth = []
        bg_img = []
        bg_depth = []
## 
        # c2ws_all = torch.stack([pose_param_net(j) for j in range(N_img)],0)
        # others = torch.argsort(torch.sum((c2ws_all-c2w)[:,:3,3]**2,axis=-1))
        others = np.arange(len(scene_train.features))
        np.random.shuffle(others)
        # cost volume construction here - 0903
        to_matrix = torch.inverse(K @ c2w)
        features,proj_mats = [], []
        for j in others[:5]:
            c2w_from = pose_param_net(j) # (4,4)
            proj_mats.append( (K @ c2w_from) @ to_matrix)
            features.append(scene_train.features[j])
        
##
        for ii,rays_dir_rows in enumerate(rays_dir_cam_split_rows):
            cr_rows_to_eval = min((ii+1)*args.num_rows_eval_img,scene_train.H) - ii*args.num_rows_eval_img
            h_,w_=torch.meshgrid(torch.arange(cr_rows_to_eval)+ii*args.num_rows_eval_img, torch.arange(scene_train.W))
            h_ = 2 * h_/scene_train.H - 1
            w_ = 2 * w_/scene_train.W - 1
            pixel_i = torch.stack([h_,w_])
            warpped_features,_ = homo_warp_with_depth(torch.stack(features),torch.stack(proj_mats).to(my_devices),
                t_vals.view(1,1,-1).repeat(len(proj_mats),cr_rows_to_eval*scene_train.W,1).to(my_devices),
                ref_g = pixel_i.repeat(len(proj_mats),1,1,1).to(my_devices)
                ) # (B C D H W)
            temp_shape = list(warpped_features.shape)
            temp_shape[1],temp_shape[2]=temp_shape[2],32
            warpped_features = cost_volume_cnn(warpped_features.transpose(1,2).flatten(0,1)).view(temp_shape).transpose(1,2) # B,C,D,H,W
            # import ipdb;ipdb.set_trace()
            cost_volume_split_rows = torch.var(warpped_features,0).permute(2,3,1,0) # B,C',D,H,W
            render_result = model_render_image(c2w, rays_dir_rows, t_vals, scene_train.near, scene_train.far,
                                               scene_train.H, scene_train.W, fxfy,
                                               model, False, 0.0, args, rgb_act_fn,cost_volume=cost_volume_split_rows,istrain=False,
                                               )
            if 'rgb_fine' in render_result.keys():
                rgb_rendered_rows = render_result['rgb_fine']  # (num_rows_eval_img, W, 3)
                depth_map = render_result['depth_fine']  # (num_rows_eval_img, W)
            else:
                rgb_rendered_rows = render_result['rgb']  # (num_rows_eval_img, W, 3)
                depth_map = render_result['depth_map']  # (num_rows_eval_img, W)
                bg_rgb_rendered_rows = render_result['bg_rgb']
                bg_depth_map = render_result['bg_depth']

            rendered_img.append(rgb_rendered_rows)
            rendered_depth.append(depth_map)
            bg_depth.append(bg_depth_map)
            bg_img.append(bg_rgb_rendered_rows)

        # combine rows to an image
        rendered_img = torch.cat(rendered_img, dim=0)
        rendered_depth = torch.cat(rendered_depth, dim=0).unsqueeze(0)  # (1, H, W)
        
        bg_img = torch.cat(bg_img, 0)
        bg_depth = torch.cat(bg_depth,0).unsqueeze(0)

        # for vis
        rendered_img_list.append(rendered_img.cpu().numpy())
        rendered_depth_list.append(rendered_depth.cpu().numpy())
        bg_img_list.append(bg_img.cpu().numpy())
        bg_depth_list.append(bg_depth.cpu().numpy())
    # random display an eval image to tfboard
    return {'img':np.array(rendered_img_list),
            'depth':np.array(rendered_depth_list)}


def train_one_epoch(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose, optimizer_volume, model, focal_net, pose_param_net, cost_volume_cnn,
                    my_devices, args, rgb_act_fn, epoch_i):
    model.train()
    focal_net.train()
    pose_param_net.train()

    # t_vals = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    t_steps = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    if args.use_disp:
        t_vals = 1/(1/(scene_train.near+1e-15) * (1 - t_steps) + 1/scene_train.far * t_steps)
    else:
        t_vals = scene_train.near * (1 - t_steps) + scene_train.far * t_steps
    N_img, H, W = scene_train.N_imgs, scene_train.H, scene_train.W
    L2_loss_epoch = []
    cost_volume_loss_epoch = []

    # shuffle the training imgs
    ids = np.arange(N_img)
    np.random.shuffle(ids)
    
    for i in ids:
        fxfy = focal_net(0)
        ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
        img = scene_train.imgs[i].to(my_devices)  # (H, W, 3)
        c2w = pose_param_net(i)  # (4, 4)
        K = torch.eye(4).to(fxfy.device)
        K[0, 0] = fxfy[0]
        K[1, 1] = fxfy[1]
        K[0, 2] = scene_train.W/2
        K[1, 2] = scene_train.H/2

        # sample pixel on an image and their rays for training.
        r_id = torch.randperm(H, device=my_devices)[:args.train_rand_rows]  # (N_select_rows)
        c_id = torch.randperm(W, device=my_devices)[:args.train_rand_cols]  # (N_select_cols)
        ray_selected_cam = ray_dir_cam[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)
        img_selected = img[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)
##
        others = np.concatenate( [ids[:i],ids[i+1:]], 0 )
        np.random.shuffle(others)
        # cost volume construction here - 0903
        to_matrix = torch.inverse(K @ c2w)
        features,proj_mats = [], []
        for j in others[:4]:
            c2w_from = pose_param_net(j) # (4,4)
            proj_mats.append( (K @ c2w_from) @ to_matrix)
            features.append(scene_train.features[j])
        h_,w_=torch.meshgrid(r_id, c_id)
        h_ = 2 * h_/scene_train.H - 1
        w_ = 2 * w_/scene_train.W - 1
        pixel_i = torch.stack([h_,w_])
##
        # render an image using selected rays, pose, sample intervals, and the network
        render_result = model_render_image(c2w, ray_selected_cam, t_vals, scene_train.near, scene_train.far,
                                           scene_train.H, scene_train.W, fxfy,
                                           model, True, 0.0, args, rgb_act_fn,
                                        features,torch.stack(proj_mats),pixel_i,cost_volume_cnn,istrain=True)  # (N_select_rows, N_select_cols, 3)
        rgb_rendered = render_result['rgb']  # (N_select_rows, N_select_cols, 3)

        # cost_volume_new = torch.mean(render_result['weight'])#render_result['weight'].permute(2,0,1) * torch.mean(cost_volume,0)
        # (H, W, D)         (D, C', H, W)  
        cost_volume_loss = torch.mean(render_result['weight'])

        # if 'rgb_fine' in render_result.keys():
        #     rgb_fine = render_result['rgb_fine']
        #     L2_loss = 0.5*F.mse_loss(rgb_rendered, img_selected)+F.mse_loss(rgb_fine, img_selected)  # loss for one image
        # else:
        L2_loss = F.mse_loss(rgb_rendered, img_selected)

        tot_loss = L2_loss+cost_volume_loss*50
        tot_loss.backward()
        optimizer_nerf.step()
        optimizer_focal.step()
        optimizer_pose.step()
        optimizer_volume.step()
        optimizer_nerf.zero_grad()
        optimizer_focal.zero_grad()
        optimizer_pose.zero_grad()
        optimizer_volume.zero_grad()
        # cost_volume_loss.backward()
        
        

        L2_loss_epoch.append(L2_loss.item())
        cost_volume_loss_epoch.append(cost_volume_loss.item())

    L2_loss_epoch_mean = np.mean(L2_loss_epoch)  # loss for all images.
    cost_volume_loss_epoch_mean = np.mean(cost_volume_loss_epoch)
    mean_losses = {
        'L2': L2_loss_epoch_mean,
        'cost_volume': cost_volume_loss_epoch_mean,
    }
    return mean_losses


def main(args):
    my_devices = torch.device('cuda:' + str(args.gpu_id))

    '''Create Folders'''
    exp_root_dir = Path(os.path.join('./logs/homo_back', args.scene_name))
    exp_root_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = Path(os.path.join(exp_root_dir, gen_detail_name(args)))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy('./models/nerf_models.py', experiment_dir)
    shutil.copy('./models/intrinsics.py', experiment_dir)
    shutil.copy('./models/poses.py', experiment_dir)
    shutil.copy('./tasks/homo/train_homo_back.py', experiment_dir)

    '''LOG'''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(experiment_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(args)

    '''Summary Writer'''
    writer = SummaryWriter(log_dir=str(experiment_dir))

    '''Data Loading'''
    scene_train = DataLoaderAnyFolder(base_dir=args.base_dir,
                                      scene_name=args.scene_name,
                                      res_ratio=args.resize_ratio,
                                      num_img_to_load=args.train_img_num,
                                      start=args.train_start,
                                      end=args.train_end,
                                      skip=args.train_skip,
                                      load_sorted=args.train_load_sorted)

    print('Train with {0:6d} images.'.format(scene_train.imgs.shape[0]))

    # We have no eval pose in this any_folder task. Eval with a 4x4 identity pose.
    # eval_c2ws = torch.eye(4).unsqueeze(0).float()  # (1, 4, 4)

    '''Model Loading'''
    pos_enc_in_dims = (2 * args.pos_enc_levels + int(args.pos_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    if args.use_dir_enc:
        dir_enc_in_dims = (2 * args.dir_enc_levels + int(args.dir_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    else:
        dir_enc_in_dims = 0

    model = OfficialNerf(pos_enc_in_dims, dir_enc_in_dims, args.hidden_dims)
    model.apply(init_weights)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device=my_devices)
    else:
        model = model.to(device=my_devices)

    # learn focal parameter
    focal_net = LearnFocal(scene_train.H, scene_train.W, args.learn_focal, args.fx_only, order=args.focal_order)
    focal_net.apply(init_weights)
    if args.multi_gpu:
        focal_net = torch.nn.DataParallel(focal_net).to(device=my_devices)
    else:
        focal_net = focal_net.to(device=my_devices)

    # learn pose for each image
    pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, None)

    pose_param_net.apply(init_weights)
    if args.multi_gpu:
        pose_param_net = torch.nn.DataParallel(pose_param_net).to(device=my_devices)
    else:
        pose_param_net = pose_param_net.to(device=my_devices)

    cost_volume_cnn = nn.Conv2d(640, 32, kernel_size=3,stride=1, padding=1, bias=False)
    #torch.nn.Sequential(
        # conv(640,256,3),
        # conv(256,128,3),
        # conv(128,64,3),
        # nn.Conv2d(640, 32, kernel_size=3,stride=1, padding=1, bias=False),
        # conv(640,32,3)
    # )
    if args.resume and os.path.exists(os.path.join(exp_root_dir, 'cost_volume_cnn.pth')):
        print("Using saved cost volume checkpoint..")
        load_ckpt_to_net(os.path.join(exp_root_dir, 'cost_volume_cnn.pth'),cost_volume_cnn,my_devices)
    else:
        cost_volume_cnn.apply(init_weights)
    if args.multi_gpu:
        cost_volume_cnn = torch.nn.DataParallel(cost_volume_cnn).to(device=my_devices)
    else:
        cost_volume_cnn = cost_volume_cnn.to(device=my_devices)

    if args.resume and args.ckpt_dir is not None and os.path.exists(os.path.join(args.ckpt_dir,'latest_nerf.pth')):
        print("Using saved model checkpoint..")
        load_ckpt_to_net(os.path.join(args.ckpt_dir,'latest_nerf.pth'),model,my_devices)
        load_ckpt_to_net(os.path.join(args.ckpt_dir,'latest_focal.pth'),focal_net,my_devices)
        load_ckpt_to_net(os.path.join(args.ckpt_dir,'latest_pose.pth'),pose_param_net,my_devices)
    elif args.resume:
        import ipdb;ipdb.set_trace()
    '''Set Optimiser'''
    optimizer_nerf = torch.optim.Adam(model.parameters(), lr=args.nerf_lr)
    optimizer_focal = torch.optim.Adam(focal_net.parameters(), lr=args.focal_lr)
    # optimizer_pose = torch.optim.Adam(list(pose_param_net.parameters())+list(cost_volume_cnn.parameters()), lr=args.pose_lr)
    optimizer_pose = torch.optim.Adam(pose_param_net.parameters(), lr=args.pose_lr)
    optimizer_volume = torch.optim.Adam(cost_volume_cnn.parameters(), lr=2e-3)

    scene_train.features = [feature.to(my_devices) for feature in scene_train.features]
    eval_c2ws = torch.stack([pose_param_net(i) for i in range(scene_train.N_imgs)])
    '''Training'''
    depth_path = os.path.join(args.base_dir,args.scene_name,'../')
    os.makedirs(depth_path,exist_ok=True)
    with torch.no_grad():
        res = eval_one_epoch(eval_c2ws, scene_train, model, focal_net, pose_param_net,cost_volume_cnn, my_devices, args, 0, depth_path, torch.sigmoid)
        np.save(depth_path+'depth.npy',res['depth'])
        print(res['depth'].shape)
    return


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    main(args)
