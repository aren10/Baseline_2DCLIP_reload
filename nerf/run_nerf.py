from operator import gt
import os, sys
from pickle import TRUE
from matplotlib import image
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
from load_nesf import load_Nesf_data
from load_nesf_clip import load_Nesf_CLIP_data
import sys
from torch.nn.functional import normalize

import clip
#from sklearn.decomposition import PCA



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    # print("chunck: ", chunk)
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    # viewdirs is None
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) #input shape torch.Size([4096, 64, 3]). Inputs have already considered view direction
    embedded = embed_fn(inputs_flat) #torch.Size([262144, 63])
    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    outputs_flat = batchify(fn, netchunk)(embedded)
    # print("outputs_flat:", outputs_flat.shape)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    # print("outputs," ,outputs.shape)
    return outputs


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                use_saliency = False,
                use_CLIP = False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3] torch.Size([4096, 64, 3])
#     raw = run_network(pts)
    #print("start raw")
    raw = network_query_fn(pts, viewdirs, network_fn) # torch.Size([4096, 64, 769])
    #print("____raw")
    #print(raw)
    #print(raw.size())
    if use_saliency:
        saliency_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, saliency = True, clip = False)
        rgb_map = saliency_map
    elif use_CLIP:
        clip_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, saliency = False, clip = True)
        rgb_map = clip_map
    else:
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, saliency = False, clip = False)
    #print("---------clip_map")
    #print(clip_map)
    #print(clip_map.size())
    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)
        # print("raw output be like:", raw.shape)
        if use_saliency:
            saliency_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, saliency = True)
            rgb_map = saliency_map
        else:
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def batchify_rays(rays_flat, chunk=1024*32, use_saliency = False, use_CLIP = False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        #print("i: ", i)
        ret = render_rays(rays_flat[i:i+chunk],use_saliency = use_saliency, use_CLIP = use_CLIP, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,use_saliency = False, use_CLIP = False,
                  **kwargs):
    """
    clip_est, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                        verbose=i < 10, retraw=True,
                        **render_kwargs_train, use_saliency= False, use_CLIP = True)
    Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else: #here
        # use provided ray batch
        rays_o, rays_d = rays
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [4096, 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1) #torch.Size([4096, 8])
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs, use_saliency = use_saliency, use_CLIP = use_CLIP)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]



def render_CLIP_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, use_clip = False):
    H, W, focal = hwf
    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
    clips_ests = []
    disps = []
    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        #print(i, time.time() - t)
        t = time.time()
        clips_est, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs, use_CLIP=use_clip)
        clips_ests.append(clips_est.cpu().numpy())
        disps.append(disp.cpu().numpy())
        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """
        if savedir is not None:
            np.save(savedir, '{:03d}_clips_est'.format(i), clips_est.cpu())
        if gt_imgs is not None:
            imgs = to8b(gt_imgs[-1])
            filename = os.path.join(savedir, '{:03d}_gt.png'.format(i))
            imageio.imwrite(filename, imgs)
    clips_ests = np.stack(clips_ests, 0)
    disps = np.stack(disps, 0)
    return clips_ests, disps


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            np.save(savedir, '{:03d}.png'.format(i), rgb8)
        if gt_imgs is not None:
            imgs = to8b(gt_imgs[-1])
            filename = os.path.join(savedir, '{:03d}_gt.png'.format(i))
            imageio.imwrite(filename, imgs)



    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args, flag, test_file):
    """Instantiate NeRF's MLP model.
    """
    #------------------positional encoding stuff-------------------------
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed) #10, 0, input_ch = 63
    #print(embed_fn(torch.tensor([1,2,3])))
    #print(input_ch)
    # print("creating nerf")
    # print("input_ch ", input_ch)
    # print("use viewdirs: ",args.use_viewdirs)
    # print("N importance: ", args.N_importance)
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs: #args.use_viewdirs = False, It means input is 3D
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    #______________________________________
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    """
    print(args.N_importance) = 0, meaning no fine sampling
    print(args.netdepth) = 8
    print(args.netwidth) = 256
    print(input_ch) = 63
    print(output_ch) = 4 why output is 4 without fine
    print(input_ch_views) = 0
    print(args.use_viewdirs) = False
    print(args.with_saliency) = False
    print(args.with_clip) = True
    """
    #coarse network
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, with_saliency = args.with_saliency, with_CLIP=args.with_clip, clip_dim=768).to(device)
    print("nerf created:")
    print("D, w, input_ch, ouput_ch:", args.netdepth, args.netwidth, input_ch, output_ch) #8 256 63 4
    print("skips, input_ch_views, use_viewdirs", skips, input_ch_views, args.use_viewdirs) #[4] 0 False
    print("--------------------------------------")
    grad_vars = list(model.parameters())
    #fine network
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, with_saliency = args.with_saliency, with_CLIP=args.with_clip, clip_dim=768).to(device)
        grad_vars += list(model_fine.parameters())
    """
    print(args.N_importance) = 0
    print(model_fine) = None
    """
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn, #network_fn can still be changed, but embeddirs_fn is fixed
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)
    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    start = 0
    basedir = args.basedir #./logs/
    expname = args.expname #mac0
    ##########################
    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        if(flag == "train"):
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]
        elif(flag == "test" or flag == "video"):
            ckpts = [os.path.join(basedir, expname, test_file)]
    print('Found ckpts', ckpts) #[]
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    #exit(0)
    ##########################
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb, #1.0
        'N_importance' : args.N_importance, #0
        'network_fine' : model_fine, #None
        'N_samples' : args.N_samples, #64
        'network_fn' : model, #NeRF 63->256, 256->128, 256->4, 256->256, 256->1, 128->768
        'use_viewdirs' : args.use_viewdirs, #False
        'white_bkgd' : args.white_bkgd, #False
        'raw_noise_std' : args.raw_noise_std, #0
    }
    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, saliency = False, clip = False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    #raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    #raw2alpha = lambda raw, dists, act_fn=torch.sigmoid: 1.-torch.exp(-act_fn(raw)*dists)
    raw2alpha = lambda raw, dists, act_fn=torch.sigmoid: (1.-torch.exp(-act_fn(raw)*dists))
    #raw2alpha = lambda raw, dists, act_fn=torch.tanh: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    if saliency:
        saliency_s = torch.sigmoid(raw[...,0]) 
        print("saliency shape:", saliency_s.shape)
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[...,1].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[...,1].shape)) * raw_noise_std
                noise = torch.Tensor(noise)
        
        alphaS = raw2alpha(raw[...,1] + noise, dists)
        weightsS = alphaS * torch.cumprod(torch.cat([torch.ones((alphaS.shape[0], 1)), 1.-alphaS + 1e-10], -1), -1)[:, :-1]
        print("weight shape:", weightsS.shape)
        saliency_s = torch.reshape(saliency_s,(saliency_s.shape[0],saliency_s.shape[1],1))
        saliency_map = torch.sum(weightsS[...,None] * saliency_s, -2)  # [N_rays, 1]

        depth_map = torch.sum(weightsS * z_vals, -1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weightsS, -1))
        acc_map = torch.sum(weightsS, -1)
    
    elif clip:

        clip_s = torch.tanh(raw[...,:-1])
        #print("------raw")
        #print(raw)
        #print("____clip_s")
        #print(clip_s)
        #print(clip_s.size())
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[...,-1].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[...,-1].shape)) * raw_noise_std
                noise = torch.Tensor(noise)
        alphaCLIP = raw2alpha(raw[...,-1] + noise, dists)  # [N_rays, N_samples] torch.Size([4096, 64])
        #print("____raw[...,-1]")
        #print(raw[...,-1])
        #print(raw[...,-1].size())
        #print("____alphaCLIP")
        #print(alphaCLIP)
        #print(alphaCLIP.size())
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weightsCLIP = alphaCLIP * torch.cumprod(torch.cat([torch.ones((alphaCLIP.shape[0], 1)), 1.-alphaCLIP + 1e-10], -1), -1)[:, :-1]
        #print(weightsCLIP)
        clip_map = torch.sum(weightsCLIP[...,None] * clip_s, -2)  # [N_rays, 768] torch.Size([4096, 768]) 
        depth_map = torch.sum(weightsCLIP * z_vals, -1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weightsCLIP, -1))
        acc_map = torch.sum(weightsCLIP, -1)

    else:
        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        print("rgb shape:", rgb.shape)
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[...,3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)
        print("weight shape:", weights.shape)

    if white_bkgd:
        if saliency:
            saliency_map = saliency_map + (1.-acc_map[...,None])
        elif clip:
            clip_map = clip_map + (1.-acc_map[...,None])
        else:
            rgb_map = rgb_map + (1.-acc_map[...,None])


    # if saliency: #TODO: add saliency later
        # return rgb_map, disp_map, acc_map, weights, depth_map, saliency_map
    # else:
    #     return rgb_map, disp_map, acc_map, weights, depth_map
    if saliency:
        return saliency_map, disp_map, acc_map, weightsS, depth_map
    elif clip:
        return clip_map, disp_map, acc_map, weightsCLIP, depth_map
    else:
        return rgb_map, disp_map, acc_map, weights, depth_map



def config_parser(env, flag):
    import configargparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, default="mac0",
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    if(env == 'mac'):
        parser.add_argument("--datadir", type=str, default='../data/toybox-13/0/', 
                            help='input data directory')
        parser.add_argument("--clip_datadir", type=str, default='../data/Nesf0_2D/', 
                            help='input data directory')
        parser.add_argument("--root_path", type=str, default='../data/', 
                            help='input data directory')
        parser.add_argument("--data_path", type=str, default= '../data/' + 'toybox-13/0/', 
                            help='input data directory') 
    elif(env == 'linux'):
        parser.add_argument("--datadir", type=str, default='/users/aren10/data/0/', 
                            help='input data directory')
        parser.add_argument("--clip_datadir", type=str, default='/users/aren10/data/Nesf0_2D/', 
                            help='input data directory')
        parser.add_argument("--root_path", type=str, default='/users/aren10/data/', 
                            help='input data directory')
        parser.add_argument("--data_path", type=str, default= '/users/aren10/data/0/', 
                            help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*5, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    if(flag == "train"):
        print("_____________________training")
        parser.add_argument("--render_only", action='store_true', default = False,
                            help='do not optimize, reload weights and render out render_poses path')
        parser.add_argument("--render_test", action='store_true', default = False,
                            help='render the test set instead of render_poses path')
        parser.add_argument("--render_factor", type=int, default=0, 
                            help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    elif(flag == "test"):
        print("_____________________testing")
        parser.add_argument("--render_only", action='store_true', default = True,
                            help='do not optimize, reload weights and render out render_poses path')
        parser.add_argument("--render_test", action='store_true', default = True,
                            help='render the test set instead of render_poses path')
        parser.add_argument("--render_factor", type=int, default=0, 
                            help='downsampling factor to speed up rendering, set 4 or 8 for fast preview') 
    elif(flag == "video"): 
        print("_____________________video")
        parser.add_argument("--render_only", action='store_true', default = False,
                            help='do not optimize, reload weights and render out render_poses path')
        parser.add_argument("--render_test", action='store_true', default = False,
                            help='render the test set instead of render_poses path')
        parser.add_argument("--render_factor", type=int, default=0, 
                            help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
            
    if(flag == "video"):
        parser.add_argument("--render_query_video", action='store_true', default = True)
        parser.add_argument("--render_compressed_feature_video", action='store_true', default = True)
    else:
        parser.add_argument("--render_query_video", action='store_true', default = False)
        parser.add_argument("--render_compressed_feature_video", action='store_true', default = False)

    
    parser.add_argument("--text", type=str, default="chair")

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='nesf_clip', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    parser.add_argument("--with_saliency", type=bool, default=False, 
                        help='train with or without saliency')

    parser.add_argument("--with_clip", type=bool, default=True, 
                        help='train with or without clip')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights")
    parser.add_argument("--i_testset", type=int, default=1000000000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=2000000000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--env")
    parser.add_argument("--flag")
    parser.add_argument("--test_file")
    return parser












































def render_query_video(text_embedding_address, render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, use_clip = False):
    H, W, focal = hwf
    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
    queries = []
    disps = []
    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        #print(i, time.time() - t)
        t = time.time()
        clips_est, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs, use_CLIP=use_clip)
        clips_est = torch.Tensor(clips_est).to(device)
        clips_est = normalize(clips_est, p = 2, dim = -1)
        nerf_img_clip = clips_est
        image_features_normalized = nerf_img_clip
        image_features_normalized = image_features_normalized.to(torch.float) #text_features_normalized = (text_features - torch.min(text_features)) / (torch.max(text_features) - torch.min(text_features))
        gt_text_clip = torch.tensor(np.load(text_embedding_address))
        text_features_normalized = gt_text_clip
        text_features_normalized = text_features_normalized.to(torch.float)
        r,c,f = image_features_normalized.size()
        input = torch.empty(r, c, 1)
        query_map = torch.zeros_like(input)
        for i in range(r):
            for j in range(c):
                query_map[i,j,0] = (torch.dot(image_features_normalized[i,j,:], text_features_normalized) / (np.linalg.norm(image_features_normalized[i,j,:].cpu().detach().numpy()) * np.linalg.norm(text_features_normalized.cpu().detach().numpy())))
        query_map = query_map.cpu().float().numpy()
        query_map = np.squeeze(query_map)
        query_map_remapped = (query_map - np.min(query_map)) / (np.max(query_map) - np.min(query_map))
        r,c = np.shape(query_map_remapped)
        query_map_3d = np.zeros((r,c,3))
        query_map_3d[:,:,0] = query_map_remapped
        query_map_3d[:,:,1] = query_map_remapped
        query_map_3d[:,:,2] = query_map_remapped
        queries.append(query_map_3d)
        disps.append(disp.cpu().numpy())
        if savedir is not None:
            np.save(savedir, '{:03d}_clips_est'.format(i), clips_est.cpu())
        if gt_imgs is not None:
            imgs = to8b(gt_imgs[-1])
            filename = os.path.join(savedir, '{:03d}_gt.png'.format(i))
            imageio.imwrite(filename, imgs)
    queries = np.stack(queries, 0)
    disps = np.stack(disps, 0)
    return queries, disps

"""
def render_compressed_feature_video(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, use_clip = False):
    H, W, focal = hwf
    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
    compressed_features = []
    disps = []
    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        #print(i, time.time() - t)
        t = time.time()
        clips_est, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs, use_CLIP=use_clip)
        clips_est = torch.Tensor(clips_est).to(device)
        clips_est = normalize(clips_est, p = 2, dim = -1)
        #sklearn.decomposition.PCA()

        compressed_features.append(compressed_f.cpu().numpy())


        disps.append(disp.cpu().numpy())
        if savedir is not None:
            np.save(savedir, '{:03d}_clips_est'.format(i), clips_est.cpu())
        if gt_imgs is not None:
            imgs = to8b(gt_imgs[-1])
            filename = os.path.join(savedir, '{:03d}_gt.png'.format(i))
            imageio.imwrite(filename, imgs)
    compressed_features = np.stack(compressed_features, 0)
    disps = np.stack(disps, 0)
    return compressed_features, disps
"""






















def train(env, flag, test_file, i_weights):

    parser = config_parser(env, flag)
    args = parser.parse_args()
    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        if  args.with_saliency:
            images, poses, render_poses, hwf, i_split, saliencies = load_blender_data(args.datadir, args.half_res, args.testskip, args.with_saliency)
        else:
            images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip, args.with_saliency)
        # print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir, args.with_saliency)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    elif args.dataset_type == 'nesf':
        if  args.with_saliency:
            images, saliencies, poses, render_poses, hwf, i_split, near, far, K = load_Nesf_data(args.datadir, use_saliency = True)
        else:
            images, poses, render_poses, hwf, i_split, near, far, K = load_Nesf_data(args.datadir)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        images = images[...,:3]
        # near = 0.
        # far = 50.
    elif args.dataset_type == 'nesf_clip':
        print("_____________importing")
        dataloader_train, dataloader_test, render_poses, hwf, near, far, K = load_Nesf_CLIP_data(args.datadir, args.clip_datadir, args.env, True)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())


    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args, flag, test_file)
    global_step = start
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)



















    if args.render_query_video:
        with torch.no_grad():
            queries,queries_disps = render_query_video(args.root_path + "Nesf0_2D/" + args.text + "_clip_feature.npy", render_poses, hwf, K, args.chunk, render_kwargs_test, use_clip = True)
            rgbs, rgbs_disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
        imageio.mimwrite(args.root_path + "Nesf0_2D/render_query_video.mp4", to8b(queries), fps=30, quality=8)
        imageio.mimwrite(args.root_path + "Nesf0_2D/render_query_video_disp.mp4", to8b(queries_disps / np.max(queries_disps)), fps=30, quality=8)
        imageio.mimwrite(args.root_path + "Nesf0_2D/render_path.mp4", to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(args.root_path + "Nesf0_2D/render_path_disp.mp4", to8b(rgbs_disps / np.max(rgbs_disps)), fps=30, quality=8)
        return
    """
    if args.render_compressed_feature_video:
        with torch.no_grad():
            compressed_features, disps = render_compressed_feature_video(render_poses, hwf, K, args.chunk, render_kwargs_test)
        imageio.mimwrite(args.root_path + "Nesf0_2D/render_compressed_feature_video.mp4", to8b(compressed_features), fps=30, quality=8)
        imageio.mimwrite(args.root_path + "Nesf0_2D/render_compressed_feature_video_disp.mp4", to8b(disps / np.max(disps)), fps=30, quality=8)
        return
    """
















    #Test
    if args.render_only:
        print('___________Test')
        with torch.no_grad():
            for i in range(len(dataloader_test)):
                #images
                images = []
                image = dataloader_test[i]["image"]
                images.append(image)
                images = np.array(images).astype(np.float32) # keep all 4 channels (RGBA)
                images = torch.Tensor(images).to(device)
                images = normalize(images, p = 2, dim = -1)
                #poses
                poses = []
                pose = dataloader_test[i]["pose"]
                poses.append(pose)
                poses = np.array(poses).astype(np.float32)
                render_poses = torch.Tensor(poses).to(device)
                testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
                os.makedirs(testsavedir, exist_ok=True)
                if args.with_clip:
                    #clips
                    img_id = dataloader_test[i]["img_ids"]
                    print(img_id)
                    clips = []
                    fname = "rgba_" + img_id[-5:] + '_image_clip_feature.npy'
                    fname = os.path.join(args.clip_datadir, fname)
                    clip = np.load(fname)
                    clips.append(clip)
                    clips = np.array(clips).astype(np.float32)
                    clips = torch.Tensor(clips).to(device)
                    clips = normalize(clips, p = 2, dim = -1)
                    #clips_ests
                    clips_ests, _ = render_CLIP_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=None, savedir=testsavedir, render_factor=args.render_factor, use_clip = args.with_clip)
                    clips_ests = torch.Tensor(clips_ests).to(device)
                    clips_ests = normalize(clips_ests, p = 2, dim = -1) #clips_ests_normalized = (clips_ests - torch.unsqueeze(torch.min(clips_ests,-1)[0],-1)) / (torch.unsqueeze(torch.max(clips_ests,-1)[0],-1) - torch.unsqueeze(torch.min(clips_ests,-1)[0],-1))
                    #loss
                    print("loss in test is: ", clip_loss(clips_ests[0,:,:,:], clips[0,:,:,:]))
                    #nerf_query_map
                    nerf_img_clip = torch.tensor(np.squeeze(clips_ests[0,:,:,:].cpu().detach().numpy()))
                    image_features_normalized = nerf_img_clip
                    image_features_normalized = image_features_normalized.to(torch.float) #text_features_normalized = (text_features - torch.min(text_features)) / (torch.max(text_features) - torch.min(text_features))
                    gt_text_clip = torch.tensor(np.load(args.root_path + "Nesf0_2D/" + args.text + "_clip_feature.npy"))
                    text_features_normalized = gt_text_clip
                    text_features_normalized = text_features_normalized.to(torch.float)
                    r,c,f = image_features_normalized.size()
                    input = torch.empty(r, c, 1)
                    query_map = torch.zeros_like(input)
                    for i in range(r):
                        for j in range(c):
                            query_map[i,j,0] = (torch.dot(image_features_normalized[i,j,:], text_features_normalized) / (np.linalg.norm(image_features_normalized[i,j,:].cpu().detach().numpy()) * np.linalg.norm(text_features_normalized.cpu().detach().numpy())))
                    query_map = query_map.cpu().float().numpy()
                    query_map = np.squeeze(query_map)
                    query_map_remapped = (query_map - np.min(query_map)) / (np.max(query_map) - np.min(query_map))
                    r,c = np.shape(query_map_remapped)
                    query_map_3d = np.zeros((r,c,3))
                    query_map_3d[:,:,0] = query_map_remapped
                    query_map_3d[:,:,1] = query_map_remapped
                    query_map_3d[:,:,2] = query_map_remapped
                    plt.imshow(query_map_3d)
                    plt.imsave(args.root_path + "Nesf0_2D/nerf_query_map.png", query_map_3d)
                    #gt_query_map
                    gt_img_clip = torch.tensor(np.squeeze(clips[0,:,:,:].cpu().detach().numpy()))
                    image_features_normalized = gt_img_clip
                    image_features_normalized = image_features_normalized.to(torch.float) #text_features_normalized = (text_features - torch.min(text_features)) / (torch.max(text_features) - torch.min(text_features))
                    r,c,f = image_features_normalized.size()
                    input = torch.empty(r, c, 1)
                    query_map = torch.zeros_like(input)
                    for i in range(r):
                        for j in range(c):
                            query_map[i,j,0] = (torch.dot(image_features_normalized[i,j,:], text_features_normalized) / (np.linalg.norm(image_features_normalized[i,j,:].cpu().detach().numpy()) * np.linalg.norm(text_features_normalized.cpu().detach().numpy())))
                    query_map = query_map.cpu().float().numpy()
                    query_map = np.squeeze(query_map)
                    query_map_remapped = (query_map - np.min(query_map)) / (np.max(query_map) - np.min(query_map))
                    r,c = np.shape(query_map_remapped)
                    query_map_3d = np.zeros((r,c,3))
                    query_map_3d[:,:,0] = query_map_remapped
                    query_map_3d[:,:,1] = query_map_remapped
                    query_map_3d[:,:,2] = query_map_remapped
                    plt.imshow(query_map_3d)
                    plt.imsave(args.root_path + "Nesf0_2D/gt_query_map.png", query_map_3d)
                else:
                    rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
                    print('Done rendering', testsavedir)
                    imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

        return











    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand #4096
    use_batching = args.no_batching #False
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N_train)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0
    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    #______________________________________
    N_iters = 100000 + 1

    losses = []
    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()
        print("iter_______: ", i)
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]
            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        else:
            img_i = np.random.choice(len(dataloader_train))
            #image_target
            image = dataloader_train[img_i]["image"]
            image = np.array(image).astype(np.float32)
            target = image
            target = torch.Tensor(target).to(device)
            target = normalize(target, p = 2, dim = -1)
            #pose
            pose = dataloader_train[img_i]["pose"]
            pose = np.array(pose).astype(np.float32)
            pose = pose[:3,:4]
            pose = torch.Tensor(pose).to(device)
            if args.with_saliency:
                saliency = saliencies[img_i]
                saliency = torch.Tensor(saliency).to(device)
            #clip
            if args.with_clip:
                img_id = dataloader_train[img_i]["img_ids"]
                print("img_id: ", img_id)
                fname = "rgba_" + img_id[-5:] + '_image_clip_feature.npy'
                fname = os.path.join(args.clip_datadir, fname)
                clip = np.load(fname)
                clip = np.array(clip).astype(np.float32)
                clip = torch.Tensor(clip).to(device)
                clip = normalize(clip, p = 2, dim = -1)
            """
            test = torch.unsqueeze(torch.tensor([1.0,1.0]),dim = 0)
            normalized_test = normalize(test)
            print(test)
            print(normalized_test)
            print(torch.norm(normalized_test))
            """
            """
            one_clip = torch.unsqueeze(clip[0,0,:],dim = 0)
            normalized_one_clip = normalize(one_clip)
            print(one_clip[0,:10])
            print(normalized_one_clip[0,:10])
            print(normalized_one_clip.size())
            print(torch.norm(normalized_one_clip))
            exit(0)
            """
            """
            print(clip[0,0,:10])
            print(torch.norm(clip[0,0,:]))
            clip_normalized = normalize(clip, p = 2, dim = -1)
            print(clip_normalized[0,0,:10])
            print(torch.norm(clip_normalized[0,0,:]))
            """
            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                #target_s
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3), torch.Size([4096, 3])
                if args.with_saliency:
                    select_coords[:, 0] = select_coords[:, 0]*(saliency.shape[0]/target.shape[0])
                    select_coords[:, 1] = select_coords[:, 1]*(saliency.shape[1]/target.shape[1])
                    saliency_s = saliency[select_coords[:, 0], select_coords[:, 1]]
                    saliency_s = saliency_s[:,0]
                #clip_s             
                if args.with_clip:
                    clip_s = clip[select_coords[:, 0], select_coords[:, 1]]

        #clip_est = Model()
        if args.with_saliency:
            saliency, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train, use_saliency= TRUE, use_CLIP = False)
        elif args.with_clip:
            clip_est, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                    verbose=i < 10, retraw=True,
                                    **render_kwargs_train, use_saliency= False, use_CLIP = True)
            clip_est = normalize(clip_est, p = 2, dim = -1)
            print("clip_s: ", clip_s[0,:3])
            print("clip_est: ", clip_est[0,:3])
            #clips_ests_normalized = (clip_est - torch.unsqueeze(torch.min(clip_est,1)[0],-1)) / (torch.unsqueeze(torch.max(clip_est,1)[0],-1) - torch.unsqueeze(torch.min(clip_est,1)[0],-1))
        else:
            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        #loss: dot product
        optimizer.zero_grad()
        if args.with_saliency:
            img_loss = img2mse(saliency, saliency_s)
        elif args.with_clip:
            img_loss = clip_loss(clip_est, clip_s) #torch.Size([4096, 768])
        else:
            img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        print("training loss: ", loss)
        losses.append(loss.cpu().detach().numpy())
        psnr = mse2psnr(img_loss)
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
        loss.backward()
        optimizer.step()
        #Update
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        dt = time.time()-time0
        #Logging
        if i%i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                #'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1
    plt.plot(losses)
    plt.savefig("losses.png")
    plt.show()

import argparse
if __name__=='__main__':
    parser_old = argparse.ArgumentParser()
    parser_old.add_argument('--env', required=True, type=str, choices=['mac', 'linux']) # 000550.tar
    parser_old.add_argument('--flag', required=True, choices=['train', 'test', "video"])
    parser_old.add_argument('--test_file', type=str, default="None") # 000550.tar
    parser_old.add_argument("--i_weights", type=int, default=0)
    args_old = parser_old.parse_args()
    if(args_old.env == 'mac'):
        torch.set_default_tensor_type('torch.FloatTensor')
    elif(args_old.env == 'linux'):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train(args_old.env, args_old.flag, args_old.test_file, int(args_old.i_weights))