import sys
import os
import argparse
from pathlib import Path
import datetime
import shutil
import logging

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from einops import rearrange, reduce, repeat

sys.path.append(os.path.join(sys.path[0], '../..'))

from dataloader.any_folder import DataLoaderAnyFolder
from utils.training_utils import set_randomness, mse2psnr, save_checkpoint
from utils.pos_enc import encode_position
from utils.volume_op import volume_sampling_ndc, volume_rendering
from utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from models.nerf_models import OfficialNerf
from models.intrinsics import LearnFocal
from models.poses import LearnPose

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
    parser.add_argument('--nerf_milestones', default=list(range(0, 10000, 10)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--nerf_lr_gamma', type=float, default=0.9954, help="learning rate milestones gamma")

    parser.add_argument('--learn_focal', default=True, type=bool)
    parser.add_argument('--focal_order', default=2, type=int)
    parser.add_argument('--fx_only', default=False, type=eval, choices=[True, False])
    parser.add_argument('--focal_lr', default=0.001, type=float)
    parser.add_argument('--focal_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--focal_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')

    parser.add_argument('--learn_R', default=True, type=eval, choices=[True, False])
    parser.add_argument('--learn_t', default=True, type=eval, choices=[True, False])
    parser.add_argument('--pose_lr', default=0.001, type=float)
    parser.add_argument('--pose_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--pose_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")

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
                       args, rgb_act_fn):
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

    render_result = volume_rendering(rgb_density, t_vals_noisy, sigma_noise_std, rgb_act_fn)
    rgb_rendered = render_result['rgb']  # (H, W, 3)
    depth_map = render_result['depth_map']  # (H, W)
    result = {
            'rgb': rgb_rendered,  # (H, W, 3)
            'sample_pos': sample_pos,  # (H, W, N_sample, 3)
            'depth_map': depth_map,  # (H, W)
            'rgb_density': rgb_density,  # (H, W, N_sample, 4)
        }
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


def eval_one_epoch(eval_c2ws, scene_train, model, focal_net, pose_param_net,
                   my_devices, args, epoch_i, writer, rgb_act_fn):
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

    for i in range(N_img):
        c2w = eval_c2ws[i].to(my_devices)  # (4, 4)

        # split an image to rows when the input image resolution is high
        rays_dir_cam_split_rows = ray_dir_cam.split(args.num_rows_eval_img, dim=0)
        rendered_img = []
        rendered_depth = []
        for rays_dir_rows in rays_dir_cam_split_rows:
            render_result = model_render_image(c2w, rays_dir_rows, t_vals, scene_train.near, scene_train.far,
                                               scene_train.H, scene_train.W, fxfy,
                                               model, False, 0.0, args, rgb_act_fn)
            if 'rgb_fine' in render_result.keys():
                rgb_rendered_rows = render_result['rgb_fine']  # (num_rows_eval_img, W, 3)
                depth_map = render_result['depth_fine']  # (num_rows_eval_img, W)
            else:
                rgb_rendered_rows = render_result['rgb']  # (num_rows_eval_img, W, 3)
                depth_map = render_result['depth_map']  # (num_rows_eval_img, W)

            rendered_img.append(rgb_rendered_rows)
            rendered_depth.append(depth_map)

        # combine rows to an image
        rendered_img = torch.cat(rendered_img, dim=0)
        rendered_depth = torch.cat(rendered_depth, dim=0).unsqueeze(0)  # (1, H, W)

        # for vis
        rendered_img_list.append(rendered_img.cpu().numpy())
        rendered_depth_list.append(rendered_depth.cpu().numpy())

    # random display an eval image to tfboard
    rand_num = np.random.randint(low=0, high=N_img)
    disp_img = np.transpose(rendered_img_list[rand_num], (2, 0, 1))  # (3, H, W)
    disp_depth = rendered_depth_list[rand_num]  # (1, H, W)
    writer.add_image('eval_img', disp_img, global_step=epoch_i)
    writer.add_image('eval_depth', disp_depth, global_step=epoch_i)
    return


def train_one_epoch(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose, model, focal_net, pose_param_net,
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

    # shuffle the training imgs
    ids = np.arange(N_img)
    np.random.shuffle(ids)

    for i in ids:
        fxfy = focal_net(0)
        ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
        img = scene_train.imgs[i].to(my_devices)  # (H, W, 3)
        c2w = pose_param_net(i)  # (4, 4)

        # sample pixel on an image and their rays for training.
        r_id = torch.randperm(H, device=my_devices)[:args.train_rand_rows]  # (N_select_rows)
        c_id = torch.randperm(W, device=my_devices)[:args.train_rand_cols]  # (N_select_cols)
        ray_selected_cam = ray_dir_cam[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)
        img_selected = img[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)

        # render an image using selected rays, pose, sample intervals, and the network
        render_result = model_render_image(c2w, ray_selected_cam, t_vals, scene_train.near, scene_train.far,
                                           scene_train.H, scene_train.W, fxfy,
                                           model, True, 0.0, args, rgb_act_fn)  # (N_select_rows, N_select_cols, 3)
        rgb_rendered = render_result['rgb']  # (N_select_rows, N_select_cols, 3)
        # if 'rgb_fine' in render_result.keys():
        #     rgb_fine = render_result['rgb_fine']
        #     L2_loss = 0.5*F.mse_loss(rgb_rendered, img_selected)+F.mse_loss(rgb_fine, img_selected)  # loss for one image
        # else:
        L2_loss = F.mse_loss(rgb_rendered, img_selected)

        L2_loss.backward()
        optimizer_nerf.step()
        optimizer_focal.step()
        optimizer_pose.step()
        optimizer_nerf.zero_grad()
        optimizer_focal.zero_grad()
        optimizer_pose.zero_grad()

        L2_loss_epoch.append(L2_loss.item())

    L2_loss_epoch_mean = np.mean(L2_loss_epoch)  # loss for all images.
    mean_losses = {
        'L2': L2_loss_epoch_mean,
    }
    return mean_losses


def main(args):
    my_devices = torch.device('cuda:' + str(args.gpu_id))

    '''Create Folders'''
    exp_root_dir = Path(os.path.join('./logs/any_folder', args.scene_name))
    exp_root_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = Path(os.path.join(exp_root_dir, gen_detail_name(args)))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy('./models/nerf_models.py', experiment_dir)
    shutil.copy('./models/intrinsics.py', experiment_dir)
    shutil.copy('./models/poses.py', experiment_dir)
    shutil.copy('./tasks/any_folder/train.py', experiment_dir)

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
    eval_c2ws = torch.eye(4).unsqueeze(0).float()  # (1, 4, 4)

    '''Model Loading'''
    pos_enc_in_dims = (2 * args.pos_enc_levels + int(args.pos_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    if args.use_dir_enc:
        dir_enc_in_dims = (2 * args.dir_enc_levels + int(args.dir_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    else:
        dir_enc_in_dims = 0

    model = OfficialNerf(pos_enc_in_dims, dir_enc_in_dims, args.hidden_dims)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device=my_devices)
    else:
        model = model.to(device=my_devices)

    # learn focal parameter
    focal_net = LearnFocal(scene_train.H, scene_train.W, args.learn_focal, args.fx_only, order=args.focal_order)
    if args.multi_gpu:
        focal_net = torch.nn.DataParallel(focal_net).to(device=my_devices)
    else:
        focal_net = focal_net.to(device=my_devices)

    # learn pose for each image
    pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, None)
    if args.multi_gpu:
        pose_param_net = torch.nn.DataParallel(pose_param_net).to(device=my_devices)
    else:
        pose_param_net = pose_param_net.to(device=my_devices)

    '''Set Optimiser'''
    optimizer_nerf = torch.optim.Adam(model.parameters(), lr=args.nerf_lr)
    optimizer_focal = torch.optim.Adam(focal_net.parameters(), lr=args.focal_lr)
    optimizer_pose = torch.optim.Adam(pose_param_net.parameters(), lr=args.pose_lr)

    # scheduler_nerf = torch.optim.lr_scheduler.MultiStepLR(optimizer_nerf, milestones=args.nerf_milestones,
    #                                                       gamma=args.nerf_lr_gamma)
    # scheduler_focal = torch.optim.lr_scheduler.MultiStepLR(optimizer_focal, milestones=args.focal_milestones,
    #                                                        gamma=args.focal_lr_gamma)
    # scheduler_pose = torch.optim.lr_scheduler.MultiStepLR(optimizer_pose, milestones=args.pose_milestones,
    #                                                       gamma=args.pose_lr_gamma)
    scheduler_nerf = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_nerf, args.epoch)
    scheduler_focal = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_focal, args.epoch)
    scheduler_pose = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pose, args.epoch)

    '''Training'''
    for epoch_i in tqdm(range(args.epoch), desc='epochs'):
        rgb_act_fn = torch.sigmoid
        train_epoch_losses = train_one_epoch(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose,
                                             model, focal_net, pose_param_net, my_devices, args, rgb_act_fn, epoch_i)
        train_L2_loss = train_epoch_losses['L2']
        scheduler_nerf.step()
        scheduler_focal.step()
        scheduler_pose.step()

        train_psnr = mse2psnr(train_L2_loss)
        writer.add_scalar('train/mse', train_L2_loss, epoch_i)
        writer.add_scalar('train/psnr', train_psnr, epoch_i)
        writer.add_scalar('train/lr', scheduler_nerf.get_lr()[0], epoch_i)

        if 'weight' in train_epoch_losses.keys():
            writer.add_scalar('train/weight', train_epoch_losses['weight'], epoch_i)

        logger.info('{0:6d} ep: Train: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, train_L2_loss, train_psnr))
        tqdm.write('{0:6d} ep: Train: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, train_L2_loss, train_psnr))

        if epoch_i % args.eval_interval == 0 and epoch_i > 0:
            with torch.no_grad():
                eval_one_epoch(eval_c2ws, scene_train, model, focal_net, pose_param_net, my_devices, args, epoch_i, writer, rgb_act_fn)

                fxfy = focal_net(0)
                tqdm.write('Est fx: {0:.2f}, fy {1:.2f}'.format(fxfy[0].item(), fxfy[1].item()))
                logger.info('Est fx: {0:.2f}, fy {1:.2f}'.format(fxfy[0].item(), fxfy[1].item()))

                # save the latest model
                save_checkpoint(epoch_i, model, optimizer_nerf, experiment_dir, ckpt_name='latest_nerf')
                save_checkpoint(epoch_i, focal_net, optimizer_focal, experiment_dir, ckpt_name='latest_focal')
                save_checkpoint(epoch_i, pose_param_net, optimizer_pose, experiment_dir, ckpt_name='latest_pose')
    return


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    main(args)
