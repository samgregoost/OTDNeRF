import argparse
import copy
import os
import random
import time
from builtins import print
from torchrbf import RBFInterpolator

import imageio
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from tqdm import tqdm, trange

from lib import tineuvox, utils
from lib.load_data import load_data
import random
import matplotlib.pyplot as plt
import pandas as pd
import csv
from geomloss import SamplesLoss
def config_parser():
    '''Define command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=0,
                        help='Random seed')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')
    parser.add_argument("--eval_psnr", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=2000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--fre_test", type=int, default=30000,
                        help='frequency of test')
    parser.add_argument("--step_to_half", type=int, default=19000,
                        help='The iteration when fp32 becomes fp16')

    parser.add_argument("--wreg", default=False, type=lambda x: x == '1')
    parser.add_argument("--interp", type=lambda x: x == '1', default=False)
    parser.add_argument("--diff", type=lambda x: x == '1', default=False)
    parser.add_argument("--rndm_bck", type=lambda x: x == '1', default=False)

    parser.add_argument("--depthloss", type=lambda x: x == '1', default=False)
    parser.add_argument('--kernel', type=str, default='linear')
    parser.add_argument('--smoothing', type=float, default=0.0)
    parser.add_argument('--gpunum', type=int, default=0)
    parser.add_argument('--expname', type=str, default='teddy')
    parser.add_argument('--ot', type=str, default='sliced-ws')

    parser.add_argument("--interp_dataset", type=lambda x: x == '1', default=False)

    parser.add_argument("--pose_aug", type=lambda x: x == '1', default=False)

    parser.add_argument('--data_interval', type=int, default=4)


    return parser

@torch.no_grad()
def render_viewpoints_hyper(model, data_class, ndc, render_kwargs, test=True, 
                                all=False, savedir=None, eval_psnr=False):
    
    rgbs = []
    rgbs_gt =[]
    rgbs_tensor =[]
    rgbs_gt_tensor =[]
    depths = []
    psnrs = []
    ms_ssims =[]

    if test:
        if all:
            idx = data_class.i_test
        else:
            idx = data_class.i_test[::16]
    else:
        if all:
            idx = data_class.i_train
        else:
            idx = data_class.i_train[::16]
    for i in tqdm(idx):
        rays_o, rays_d, viewdirs,rgb_gt = data_class.load_idx(i, not_dic=True)
        keys = ['rgb_marched', 'depth']
        time_one = data_class.all_time[i]*torch.ones_like(rays_o[:,0:1])
        cam_one = data_class.all_cam[i]*torch.ones_like(rays_o[:,0:1])
        bacth_size = 1000
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd,ts, cams,**render_kwargs).items() if k in keys}
            for ro, rd, vd ,ts,cams in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0),
                                             viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0),cam_one.split(bacth_size, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(data_class.h,data_class.w,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb_gt = rgb_gt.reshape(data_class.h,data_class.w,-1).cpu().numpy()
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        rgbs.append(rgb)
        depths.append(depth)
        rgbs_gt.append(rgb_gt)
        if eval_psnr:
            p = -10. * np.log10(np.mean(np.square(rgb - rgb_gt)))
            psnrs.append(p)
            rgbs_tensor.append(torch.from_numpy(np.clip(rgb,0,1)).reshape(-1,data_class.h,data_class.w))
            rgbs_gt_tensor.append(torch.from_numpy(np.clip(rgb_gt,0,1)).reshape(-1,data_class.h,data_class.w))
        if i==0:
            print('Testing', rgb.shape)
    if eval_psnr:
        rgbs_tensor = torch.stack(rgbs_tensor,0)
        rgbs_gt_tensor = torch.stack(rgbs_gt_tensor,0)
        ms_ssims = ms_ssim(rgbs_gt_tensor, rgbs_tensor, data_range=1, size_average=True )
    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        print('Testing ms_ssims', ms_ssims, '(avg)')

    if savedir is not None:
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
    rgbs = np.array(rgbs)
    depths = np.array(depths)
    return rgbs,depths

@torch.no_grad()
def render_viewpoints_iphone(model, data_class, ndc, render_kwargs, test=True,
                            all=False, savedir=None, eval_psnr=False):

    rgbs = []
    rgbs_gt = []
    rgbs_tensor = []
    rgbs_gt_tensor = []
    depths = []
    psnrs = []
    ms_ssims = []
    lpips_alex = []
    covisibles = []

    if test:
        if cfg.train_config.interp_dataset:
            idx = data_class.i_train
        else:
            if all:
                idx = data_class.i_test
            else:
                idx = data_class.i_test[::16]
    else:
        if all:
            idx = data_class.i_train
        else:
            idx = data_class.i_train[::16]
    for i in tqdm(idx):
        if cfg.train_config.interp_dataset:
            rays_o, rays_d, viewdirs, rgb_gt, covisible, depth_gt = data_class.load_idx(0, not_dic=True)

        else:
            rays_o, rays_d, viewdirs, rgb_gt, covisible, depth_gt = data_class.load_idx(i, not_dic=True)
        keys = ['rgb_marched', 'depth']
        time_one = data_class.all_time[i] * torch.ones_like(rays_o[:, 0:1])
        cam_one = data_class.all_cam[i] * torch.ones_like(rays_o[:, 0:1])
        bacth_size = 1000
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, ts, cams, **render_kwargs).items() if k in keys}
            for ro, rd, vd, ts, cams in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0),
                                            viewdirs.split(bacth_size, 0), time_one.split(bacth_size, 0),
                                            cam_one.split(bacth_size, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(data_class.h, data_class.w, -1)
            for k in render_result_chunks[0].keys()
        }
        rgb_gt = rgb_gt.reshape(data_class.h, data_class.w, -1).cpu().numpy()
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        rgbs.append(rgb)
        depths.append(depth)
        rgbs_gt.append(rgb_gt)
        if eval_psnr:
            if cfg.train_config.interp_dataset:
                covisible = 1
            else:
                covisible = covisible.reshape(data_class.h, data_class.w, -1)
            covisibles.append(covisible)
            p = -10. * np.log10(np.mean(np.square((rgb - rgb_gt)*covisible)))
            psnrs.append(p)
            rgbs_tensor.append(torch.from_numpy(np.clip(rgb*covisible, 0, 1)).reshape(-1, data_class.h, data_class.w))
            rgbs_gt_tensor.append(torch.from_numpy(np.clip(rgb_gt*covisible, 0, 1)).reshape(-1, data_class.h, data_class.w))
            if cfg.train_config.interp_dataset:
                rgb_gt = rgbs_gt[0]
            lpips_alex.append(
                utils.rgb_lpips(np.float32(np.clip(rgb*covisible, 0, 1)).reshape(data_class.h, data_class.w,-1),
                                np.float32(np.clip(rgb_gt*covisible, 0, 1)).reshape(data_class.h, data_class.w,-1), net_name='alex', device=rgbs_gt_tensor[0].device))
        if i == 0:
            print('Testing', rgb.shape)
    if eval_psnr:
        rgbs_tensor = torch.stack(rgbs_tensor, 0)
        rgbs_gt_tensor = torch.stack(rgbs_gt_tensor, 0)
        ms_ssims = ms_ssim(rgbs_gt_tensor, rgbs_tensor, data_range=1, size_average=True)


    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        print('Testing ms_ssims', ms_ssims, '(avg)')
        print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')



    if savedir is not None:
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i]*covisibles[i])
            filename = os.path.join(savedir, '{:03d}_covisible.png'.format(i))
            imageio.imwrite(filename, rgb8)
    rgbs = np.array(rgbs)
    depths = np.array(depths)
    return rgbs, depths, np.mean(psnrs), ms_ssims, np.mean(lpips_alex)

@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, test_times=None, render_factor=0, eval_psnr=False,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False,):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''

    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor
    rgbs = []
    depths = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = tineuvox.get_rays_of_a_view(
                H, W, K, c2w, ndc)
        keys = ['rgb_marched', 'depth']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        time_one = test_times[i]*torch.ones_like(rays_o[:,0:1])
        bacth_size=1000
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd,ts, **render_kwargs).items() if k in keys}
            for ro, rd, vd ,ts in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0), viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor == 0:
            if eval_psnr:
                p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name = 'alex', device = c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name = 'vgg', device = c2w.device))

    if len(psnrs):
        if eval_psnr: print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if savedir is not None:
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    return rgbs, depths


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)
    if cfg.data.dataset_type == 'hyper_dataset':
        kept_keys = {
            'data_class',
            'near', 'far',
            'i_train', 'i_val', 'i_test',}
        for k in list(data_dict.keys()):
            if k not in kept_keys:
                data_dict.pop(k)
        return data_dict

    elif cfg.data.dataset_type == 'iphone_dataset':
        kept_keys = {
            'data_class',
            'near', 'far',
            'i_train', 'i_val', 'i_test',}
        for k in list(data_dict.keys()):
            if k not in kept_keys:
                data_dict.pop(k)
        return data_dict

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images','times','render_times'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device = 'cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict



def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = tineuvox.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,ndc=cfg.data.ndc)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max


def compute_bbox_by_cam_frustrm_hyper(args, cfg,data_class):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for i in data_class.i_train:
        rays_o, _, viewdirs,_ = data_class.load_idx(i,not_dic=True)
        pts_nf = torch.stack([rays_o+viewdirs*data_class.near, rays_o+viewdirs*data_class.far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

def compute_bbox_by_cam_frustrm_iphone(args, cfg,data_class):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for i in data_class.i_train:
        rays_o, _, viewdirs,_, _, _ = data_class.load_idx(i,not_dic=True)
        pts_nf = torch.stack([rays_o+viewdirs*data_class.near, rays_o+viewdirs*data_class.far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift

    if cfg.data.dataset_type !='hyper_dataset' and cfg.data.dataset_type !='iphone_dataset':
        HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images ,times,render_times= [
            data_dict[k] for k in [
                'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 
                'render_poses', 'images',
                'times','render_times'
            ]
        ]
        times = torch.Tensor(times)
        times_i_train = times[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
    else:
        data_class = data_dict['data_class']
        near = data_class.near
        far = data_class.far
        i_train = data_class.i_train
        i_test = data_class.i_test

    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
    # init model and optimizer
    start = 0
    # init model
    model_kwargs = copy.deepcopy(cfg_model)

    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale) :
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
    model = tineuvox.TiNeuVox(
        xyz_min=xyz_min, xyz_max=xyz_max,
        num_voxels=num_voxels,
        **model_kwargs)
    model = model.to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

    # init rendering setup
    render_kwargs = {
        'near': near,
        'far': far,
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
    }
    # init batch rays sampler
    def gather_training_rays_hyper():
        now_device = 'cpu'  if cfg.data.load2gpu_on_the_fly else device
        N = len(data_class.i_train)*data_class.h*data_class.w
        rgb_tr = torch.zeros([N,3], device=now_device)
        rays_o_tr = torch.zeros_like(rgb_tr)
        rays_d_tr = torch.zeros_like(rgb_tr)
        viewdirs_tr = torch.zeros_like(rgb_tr)
        times_tr = torch.ones([N,1], device=now_device)
        cam_tr = torch.ones([N,1], device=now_device)
        imsz = []
        top = 0
        for i in data_class.i_train:
            rays_o, rays_d, viewdirs,rgb = data_class.load_idx(i,not_dic=True)
            n = rgb.shape[0]
            if data_class.add_cam:
                cam_tr[top:top+n] = cam_tr[top:top+n]*data_class.all_cam[i]
            times_tr[top:top+n] = times_tr[top:top+n]*data_class.all_time[i]
            rgb_tr[top:top+n].copy_(rgb)
            rays_o_tr[top:top+n].copy_(rays_o.to(now_device))
            rays_d_tr[top:top+n].copy_(rays_d.to(now_device))
            viewdirs_tr[top:top+n].copy_(viewdirs.to(now_device))
            imsz.append(n)
            top += n
        assert top == N
        index_generator = tineuvox.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, times_tr,cam_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    def gather_training_rays_iphone():
        now_device = 'cpu'  if cfg.data.load2gpu_on_the_fly else device
        N = len(data_class.i_train)*data_class.h*data_class.w
        rgb_tr = torch.zeros([N,3], device=now_device)
        rays_o_tr = torch.zeros_like(rgb_tr)
        rays_d_tr = torch.zeros_like(rgb_tr)
        viewdirs_tr = torch.zeros_like(rgb_tr)
        depth_tr = torch.ones([N,1], device=now_device)
        times_tr = torch.ones([N,1], device=now_device)
        cam_tr = torch.ones([N,1], device=now_device)
        imsz = []
       # depth_tr = []
        top = 0
        for i in data_class.i_train:
            rays_o, rays_d, viewdirs,rgb,_,depth = data_class.load_idx(i,not_dic=True)
            n = rgb.shape[0]
            if data_class.add_cam:
                cam_tr[top:top+n] = cam_tr[top:top+n]*data_class.all_cam[i]
            times_tr[top:top+n] = times_tr[top:top+n]*data_class.all_time[i]
            rgb_tr[top:top+n].copy_(rgb)
            rays_o_tr[top:top+n].copy_(rays_o.to(now_device))
            rays_d_tr[top:top+n].copy_(rays_d.to(now_device))
            # check if the list has at least not none value
            if depth is not None:
                depth_tr[top:top + n].copy_(depth.to(now_device))
            viewdirs_tr[top:top+n].copy_(viewdirs.to(now_device))
            imsz.append(n)
            top += n
        assert top == N
        index_generator = tineuvox.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, times_tr,cam_tr,rays_o_tr, rays_d_tr, viewdirs_tr, depth_tr,  imsz, batch_index_sampler

    def rbf_interpolation(coords, pixels):
        interpolator = RBFInterpolator(coords, pixels, smoothing=cfg_train.smoothing,
                                       kernel=cfg_train.kernel, device = coords.device, epsilon = cfg_train.epsilon)

        x = torch.linspace(0, 1, int(data_class.w/2))
        y = torch.linspace(0, 1, int(data_class.h/2))
        grid_points = torch.meshgrid(x, y, indexing='ij')
        grid_points = torch.stack(grid_points, dim=-1).reshape(-1, 2)

        # Query RBF on grid points
        interp_vals = interpolator(grid_points)

        return interp_vals

    def gather_wreg_training_rays():
        #i = random.sample(data_class.i_train + data_class.i_test, 1)
        if cfg_train.interp_dataset:
            i = random.sample(data_class.i_train.tolist(), 1)
        else:
            i = random.sample(data_class.i_train, 1)
        rays_o, rays_d, viewdirs, rgb, coords  = data_class.load_idx_wreg(i[0], not_dic=True)

        random_idx = random.sample(list(np.arange(rays_o.shape[0])), cfg_train.sample_size)

        rays_o = rays_o[random_idx]
        rays_d = rays_d[random_idx]
        viewdirs = viewdirs[random_idx]
        coords = coords[random_idx]
       # rgb = rgb[random_idx]



        random_idx = random.sample(list(np.arange(rays_o.shape[0])), cfg_train.sample_size)

        rays_o2 = rays_o[random_idx]
        rays_d2 = rays_d[random_idx]
        viewdirs2 = viewdirs[random_idx]
        coords2 = coords[random_idx]


        time1 = torch.ones([rays_o.shape[0], 1]) * random.uniform(0, 1-cfg_train.interval)
        time2 = time1 + cfg_train.interval * random.uniform(0, 1)

        return rays_o, rays_d, viewdirs, rgb, time1, time2, coords, rays_o2, rays_d2, viewdirs2, coords2, i[0], random_idx

    def gather_poseaug_training_rays(pose, noise, random_idx):
        data_dict = data_class.load_aug_pose(pose, noise)
        rays_o = data_dict['rays_ori']
        rays_d = data_dict['rays_dir']
        viewdirs = data_dict['viewdirs']
        coords = data_dict['xy_coords']

    #    random_idx = random.sample(list(np.arange(rays_o.shape[0])), cfg_train.sample_size)
        rays_o = rays_o[random_idx]
        rays_d = rays_d[random_idx]
        viewdirs = viewdirs[random_idx]
        coords = coords[random_idx]

        return rays_o, rays_d, viewdirs, coords

    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            print('cfg_train.ray_sampler =in_maskcache')
            rgb_tr, times_flaten,rays_o_tr, rays_d_tr, viewdirs_tr, imsz = tineuvox.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,times=times_i_train,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, 
                    model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            print('cfg_train.ray_sampler =flatten')
            rgb_tr, times_flaten,rays_o_tr, rays_d_tr, viewdirs_tr, imsz = tineuvox.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,times=times_i_train,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc,)
        else:
            print('cfg_train.ray_sampler =random')
            rgb_tr, times_flaten,rays_o_tr, rays_d_tr, viewdirs_tr, imsz = tineuvox.get_training_rays(
                rgb_tr=rgb_tr_ori,times=times_i_train,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc,)
        index_generator = tineuvox.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr,times_flaten, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler
    if cfg.data.dataset_type !='hyper_dataset' and cfg.data.dataset_type !='iphone_dataset':
        rgb_tr,times_flaten, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()
    elif cfg.data.dataset_type =='hyper_dataset':
        rgb_tr,times_flaten,cam_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays_hyper()
    elif cfg.data.dataset_type =='iphone_dataset':
        rgb_tr,times_flaten,cam_tr, rays_o_tr, rays_d_tr, viewdirs_tr, depth_tr, imsz, batch_index_sampler = gather_training_rays_iphone()

    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1

    if cfg_train.ot == 'energy':
        ot_dist = SamplesLoss("energy", p=2, blur=0.5, scaling=0.99)
    if cfg_train.ot == 'gaussian':
        ot_dist = SamplesLoss("gaussian")
    if cfg_train.ot == 'laplace':
        ot_dist = SamplesLoss("laplacian")
    if cfg_train.ot == 'sinkhorn':
        ot_dist = SamplesLoss("sinkhorn")

    for global_step in trange(1+start, 1+cfg_train.N_iters):
        if global_step == args.step_to_half:
                model.feature.data=model.feature.data.half()
        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(model, tineuvox.TiNeuVox):
                model.scale_volume_grid(cur_voxels)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache'] or (cfg.data.dataset_type =='hyper_dataset' or cfg.data.dataset_type =='iphone_dataset'):
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
            depth_target = depth_tr[sel_i]
            times_sel = times_flaten[sel_i]
            if cfg.data.dataset_type == 'hyper_dataset' or cfg.data.dataset_type == 'iphone_dataset':
                if data_class.add_cam == True:
                    cam_sel = cam_tr[sel_i]
                    cam_sel = cam_sel.to(device)
                    render_kwargs.update({'cam_sel':cam_sel})
                if data_class.use_bg_points == True:
                    sel_idx = torch.randint(data_class.bg_points.shape[0], [cfg_train.N_rand//3])
                    bg_points_sel = data_class.bg_points[sel_idx]
                    bg_points_sel = bg_points_sel.to(device)
                    render_kwargs.update({'bg_points_sel':bg_points_sel})
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
            times_sel = times_flaten[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)
            times_sel = times_sel.to(device)
            depth_target = depth_target.to(device)

        # volume rendering
        render_result = model(rays_o, rays_d, viewdirs, times_sel, global_step=global_step, **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none = True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach())


        if cfg_train.depthloss:
            depth_target = depth_target.view(-1)
            mask = (depth_target!=0)
            loss +=  0.1 * cfg_train.weight_main * F.mse_loss(render_result['depth']*mask, depth_target * mask)


        if cfg.data.dataset_type =='hyper_dataset' or cfg.data.dataset_type =='iphone_dataset':
            if data_class.use_bg_points == True:
                loss = loss+F.mse_loss(render_result['bg_points_delta'],bg_points_sel)
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss
        if cfg_train.wreg:
          #  rays_o, rays_d, viewdirs, rgb, time1, time2, coords = gather_wreg_training_rays()
            rays_o, rays_d, viewdirs, rgb, time1, time2, coords, rays_o2, rays_d2, viewdirs2, coords2, pose, random_idx = gather_wreg_training_rays()
            render_result_1 = model(rays_o, rays_d, viewdirs, time1, global_step=global_step, **render_kwargs)
            rgb_1 = render_result_1['rgb_marched']

            if cfg_train.diff:
                render_result_2 = model(rays_o2, rays_d2, viewdirs2, time2, global_step=global_step, rndm_bck = cfg.train_config.rndm_bck, **render_kwargs)
            else:
                if cfg_train.pose_aug:
                    if global_step % 2 == 0:
                        rays_o_pose2, rays_d_pose2, viewdirs_pose2, coords_pose2 = gather_poseaug_training_rays(pose,
                                                                                                                0.00 , random_idx)
                        render_result_2_pose = model(rays_o_pose2, rays_d_pose2, viewdirs_pose2, time1,
                                                     global_step=global_step, **render_kwargs)
                        rgb_2 = render_result_2_pose['rgb_marched']
                    else:
                        render_result_2 = model(rays_o, rays_d, viewdirs, time2, global_step=global_step, rndm_bck = cfg.train_config.rndm_bck, **render_kwargs)
                        rgb_2 = render_result_2['rgb_marched']
                else:
                    render_result_2 = model(rays_o, rays_d, viewdirs, time2, global_step=global_step,
                                            rndm_bck=cfg.train_config.rndm_bck, **render_kwargs)
                    rgb_2 = render_result_2['rgb_marched']




            # if cfg_train.pose_aug == True and :
            #     rays_o_pose2, rays_d_pose2, viewdirs_pose2, coords_pose2 = gather_poseaug_training_rays(pose, 0.2)
            #     render_result_2_pose = model(rays_o_pose2, rays_d_pose2, viewdirs_pose2, time1, global_step=global_step, **render_kwargs)
            #     rgb_2 = render_result_2_pose['rgb_marched']
            #   #  rgb_pose2 = rbf_interpolation(coords, rgb_pose2)

            if cfg_train.interp == True and global_step < cfg_train.wreg_off:
                if cfg_train.diff:
                    rgb_1 = rbf_interpolation(coords, rgb_1)
                    rgb_2 = rbf_interpolation(coords2, rgb_2)
                else:
                    rgb_1 = rbf_interpolation(coords, rgb_1)
                    if global_step % 2 == 0 and cfg_train.pose_aug:
                        rgb_2 = rbf_interpolation(coords_pose2, rgb_2)
                    else:
                        rgb_2 = rbf_interpolation(coords, rgb_2)

                if global_step % 10 == 0 or global_step % 11 == 0:
                    plt.imshow(rgb_1.reshape(int(data_class.w/2),
                                             int(data_class.h/2), 3).detach().cpu().numpy())
                 #   plt.scatter(coords[:, 0].detach().cpu().numpy() * data_class.h/2,
                  #              coords[:, 1].detach().cpu().numpy() * data_class.w/2, c='red', s=0.2)
                    plt.savefig(f'interp_results/interpolation_rgb1-{global_step}_{cfg_train.kernel}.png')
                    plt.close()

                    plt.imshow(rgb_2.reshape(int(data_class.w / 2),
                                             int(data_class.h / 2), 3).detach().cpu().numpy())
                    #   plt.scatter(coords[:, 0].detach().cpu().numpy() * data_class.h/2,
                    #              coords[:, 1].detach().cpu().numpy() * data_class.w/2, c='red', s=0.2)
                    plt.savefig(f'interp_results/interpolation_rgb2-{global_step}_{cfg_train.kernel}.png')



            if cfg_train.weightnet == True:
                rgb_1 = rgb_1 * model.weightnet(torch.cat([rgb_1, coords], dim = -1))
                rgb_2 = rgb_2 * model.weightnet(torch.cat([rgb_2, coords], dim = -1))

            if cfg_train.gaussian_smoothing == True:
                rgb_1 = rgb_1 + torch.randn(*rgb_1.shape) * 0.01
                rgb_2 = rgb_2 + torch.randn(*rgb_2.shape) * 0.01

            if cfg_train.ot == 'sliced-ws':

                wvec = torch.rand(256, 1, rgb_1.shape[-1]).to(device)
                wvec = wvec / torch.linalg.vector_norm(wvec, dim=2, keepdim=True)

              #  wvec2 = torch.rand(256, 1, rgb_1.shape[-1]).to(device)
             #   wvec2 = wvec2 / torch.linalg.vector_norm(wvec2, dim=2, keepdim=True)

                p = (rgb_1[None, :, :] * wvec).sum(2)
                # if cfg_train.pose_aug == True and global_step%2 == 0:
                #     q = (rgb_pose2[None, :, :] * wvec).sum(2)
                # else:
                q = (rgb_2[None, :, :] * wvec).sum(2)

                p, _ = torch.sort(p, dim=1)
                q, _ = torch.sort(q, dim=1)

                wreg_loss = torch.mean(torch.abs(p - q) / (1 + (p - q) ** 2))

            else:
                wreg_loss = ot_dist(rgb_1, rgb_2)
            loss += wreg_loss * cfg_train.weight_wreg

        loss.backward()

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_feature>0:
                model.feature_total_variation_add_grad(
                    cfg_train.weight_tv_feature/len(rays_o), global_step<cfg_train.tv_feature_before)
        optimizer.step()
        psnr_lst.append(psnr.item())
        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        if global_step%args.i_print == 0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction : iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []

        if False: #global_step%(args.fre_test) == 0:
            render_viewpoints_kwargs = {
                'model': model,
                'ndc': cfg.data.ndc,
                'render_kwargs': {
                    'near': near,
                    'far': far,
                    'bg': 1 if cfg.data.white_bkgd else 0,
                    'stepsize': cfg_model.stepsize,

                    },
                }
            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'{global_step}-test')
            if os.path.exists(testsavedir) == False:
                os.makedirs(testsavedir)
            if cfg.data.dataset_type != 'hyper_dataset' and cfg.data.dataset_type != 'iphone_dataset':
                rgbs,disps = render_viewpoints(
                    render_poses=data_dict['poses'][data_dict['i_test']],
                    HW=data_dict['HW'][data_dict['i_test']],
                    Ks=data_dict['Ks'][data_dict['i_test']],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                    savedir=testsavedir,
                    test_times=data_dict['times'][data_dict['i_test']],
                    eval_psnr=args.eval_psnr, eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    **render_viewpoints_kwargs)
            else:
                rgbs,disps = render_viewpoints_hyper(
                    data_class=data_class,
                    savedir=testsavedir, all=False, test=True, eval_psnr=args.eval_psnr,
                    **render_viewpoints_kwargs)

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
        }, last_ckpt_path)
        print('scene_rep_reconstruction : saved checkpoints at', last_ckpt_path)

def train(args, cfg, data_dict=None):

    # init
    print('train: start')
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # coarse geometry searching
    if cfg.data.dataset_type == 'hyper_dataset':
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm_hyper(args = args, cfg = cfg,data_class = data_dict['data_class'])
    elif cfg.data.dataset_type == 'iphone_dataset':
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm_iphone(args=args, cfg=cfg, data_class=data_dict['data_class'])
    else:
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args = args, cfg = cfg, **data_dict)
    coarse_ckpt_path = None



    # fine detail reconstruction
    eps_time = time.time()
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.model_and_render, cfg_train=cfg.train_config,
            xyz_min=xyz_min, xyz_max=xyz_max,
            data_dict=data_dict)
    eps_loop = time.time() - eps_time
    eps_time_str = f'{eps_loop//3600:02.0f}:{eps_loop//60%60:02.0f}:{eps_loop%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')

from mmengine.config import Config
if __name__=='__main__':


    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    cfg.data.data_interval = args.data_interval
    # init enviroment

    cfg.train_config.diff = args.diff
    cfg.train_config.smoothing = args.smoothing
    cfg.train_config.interp = args.interp
    cfg.train_config.kernel = args.kernel
    cfg.train_config.wreg = args.wreg
    cfg.train_config.gpunum = args.gpunum
    cfg.expname = args.expname
    cfg.train_config.rndm_bck = args.rndm_bck
    cfg.train_config.depthloss = args.depthloss
    cfg.train_config.interp_dataset = args.interp_dataset
    cfg.train_config.pose_aug = args.pose_aug
    cfg.train_config.ot = args.ot

    torch.cuda.set_device(cfg.train_config.gpunum)



    print(args)

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    seed_everything()

    data_dict = None
    # load images / poses / camera settings / data split
    data_dict = load_everything(args = args, cfg = cfg)

    # train
    if not args.render_only :
        train(args, cfg, data_dict = data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        model_class = tineuvox.TiNeuVox
        model = utils.load_model(model_class, ckpt_path).to(device)
        near=data_dict['near']
        far=data_dict['far']
        stepsize = cfg.model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': near,
                'far': far,
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'render_depth': True,
            },
        }

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}') #
        os.makedirs(testsavedir, exist_ok = True)
        if cfg.data.dataset_type  != 'hyper_dataset' and cfg.data.dataset_type  != 'iphone_dataset':
            rgbs, disps = render_viewpoints(
                    render_poses=data_dict['poses'][data_dict['i_train']],
                    HW=data_dict['HW'][data_dict['i_train']],
                    Ks=data_dict['Ks'][data_dict['i_train']],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                    savedir=testsavedir,
                    test_times=data_dict['times'][data_dict['i_train']],
                    eval_psnr=args.eval_psnr, eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    **render_viewpoints_kwargs)
        elif cfg.data.dataset_type == 'hyper_dataset' or cfg.data.dataset_type == 'iphone_dataset':
            rgbs,disps = render_viewpoints_hyper(
                    data_calss=data_dict['data_class'],
                    savedir=testsavedir, all=True, test=False,
                    eval_psnr=args.eval_psnr,
                    **render_viewpoints_kwargs)
        else:
            raise NotImplementedError

        imageio.mimwrite(os.path.join(testsavedir, 'train_video.rgb.mp4'), utils.to8b(rgbs), fps = 30, quality = 8)
        imageio.mimwrite(os.path.join(testsavedir, 'train_video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps = 30, quality = 8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        if cfg.data.dataset_type  != 'hyper_dataset' and cfg.data.dataset_type  != 'iphone_dataset':
            rgbs, disps = render_viewpoints(
                    render_poses=data_dict['poses'][data_dict['i_test']],
                    HW=data_dict['HW'][data_dict['i_test']],
                    Ks=data_dict['Ks'][data_dict['i_test']],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                    savedir=testsavedir,
                    test_times=data_dict['times'][data_dict['i_test']],
                    eval_psnr=args.eval_psnr,eval_ssim = args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    **render_viewpoints_kwargs)
        elif cfg.data.dataset_type == 'hyper_dataset':
            rgbs,disps = render_viewpoints_hyper(
                    data_class=data_dict['data_class'],
                    savedir=testsavedir,all=True,test=True,
                    eval_psnr=args.eval_psnr,
                    **render_viewpoints_kwargs)
        elif cfg.data.dataset_type == 'iphone_dataset':
            rgbs,disps, psnr, ms_ssim, lpips_alex = render_viewpoints_iphone(
                    data_class=data_dict['data_class'],
                    savedir=testsavedir,all=True,test=True,
                    eval_psnr=args.eval_psnr,
                    **render_viewpoints_kwargs)

        else:
            raise NotImplementedError

        imageio.mimwrite(os.path.join(testsavedir, 'test_video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'test_video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)
        if cfg.data.dataset_type == 'iphone_dataset':
            if not os.path.isfile('results_sinkhorn.csv'):
                # df = pd.read_csv('results.csv')
                # method = f'diff {args.diff} + smoothing {args.smoothing} interp {args.interp} kernel {args.kernel} + wreg {args.wreg}'
                # df_temp = pd.DataFrame([{'dataset': cfg.expname, 'method' : method, 'mPSNR': psnr, 'mSSIM': ms_ssim, 'LPIPs': lpips_alex}])
                # df = df._append(df_temp)
                with open('results_sinkhorn.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    field = ["dataset", "method", "mPSNR", "mSSIM", "LPIPs"]
                    method = f'diff {args.diff}  smoothing {args.smoothing} interp {args.interp} kernel {args.kernel}  wreg {args.wreg}'
                    writer.writerow(field)
                    writer.writerow([cfg.expname,  method, psnr, ms_ssim,  lpips_alex])
                    file.close()

            else:
                with open('results_sinkhorn.csv', 'a') as file:
                    writer = csv.writer(file)
                    method = f'diff {args.diff}  smoothing {args.smoothing} interp {args.interp} kernel {args.kernel}  wreg {args.wreg}'
                    data = [cfg.expname,  method, psnr, ms_ssim,  lpips_alex]
                    writer.writerow(data)
                    file.close()

            #     df = pd.DataFrame()
            # df.to_csv('results.csv')

    # render video
    if args.render_video:
        if cfg.data.dataset_type  != 'hyper_dataset' and cfg.data.dataset_type  != 'iphone_dataset':
            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}_time')
            os.makedirs(testsavedir, exist_ok=True)
            rgbs, disps = render_viewpoints(
                    render_poses=data_dict['render_poses'],
                    HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                    Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                    render_factor=args.render_video_factor,
                    savedir=testsavedir,
                    test_times=data_dict['render_times'],
                    **render_viewpoints_kwargs)
            imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality =8)
        else:
            raise NotImplementedError

    print('Done')

