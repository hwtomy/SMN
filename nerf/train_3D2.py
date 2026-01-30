import os
import sys
import time
import json
import argparse
import random

import numpy as np
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from pe import *
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from modelinr import ModMLP

# tf.compat.v1.enable_eager_execution()


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""

    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)
    if viewdirs is not None:
        input_dirs = tf.broadcast_to(viewdirs[:, None], inputs.shape)
        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = tf.concat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = tf.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
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
                verbose=False):
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

    def raw2outputs(raw, z_vals, rays_d):
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
        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=tf.nn.relu): return 1.0 - \
            tf.exp(-act_fn(raw) * dists)

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        dists = tf.concat(
            [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
            axis=-1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        # Extract RGB of each sample position along each ray.
        rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        # Add noise to model's predictions for density. Can be used to 
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * \
            tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True)

        # Computed weighted color of each sample along each ray.
        rgb_map = tf.reduce_sum(
            weights[..., None] * rgb, axis=-2)  # [N_rays, 3]

        # Estimated depth map is expected distance.
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

        # Disparity map is inverse depth.
        disp_map = 1./tf.maximum(1e-10, depth_map /
                                 tf.reduce_sum(weights, axis=-1))

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = tf.reduce_sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    ###############################
    # batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = tf.linspace(0., 1., N_samples)
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = tf.broadcast_to(z_vals, [N_rays, N_samples])

    # Perturb sampling time along each ray.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # Points in space to evaluate model at.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # Evaluate model at each point.
    raw = network_query_fn(pts, viewdirs, network_fn)  # [N_rays, N_samples, 4]
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # Obtain additional integration times to evaluate based on the weights
        # assigned to colors in the coarse model.
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = tf.stop_gradient(z_samples)

        # Obtain all points to evaluate color, density at.
        z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        # Make predictions with network_fine.
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = tf.math.reduce_std(z_samples, -1)  # [N_rays]

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal,
           chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays

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
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(
            H, W, focal, tf.cast(1., tf.float32), rays_o, rays_d)

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    near, far = near * \
        tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, near, far], axis=-1)
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = tf.concat([rays, viewdirs], axis=-1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(
            H, W, focal, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            print(p)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args, device):
    # Embedders
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    # Model
    # model = NeRF(D=args.netdepth, W=args.netwidth,
    #              input_ch=input_ch, input_ch_views=input_ch_views,
    #              output_ch=4, skips=[4], use_viewdirs=args.use_viewdirs).to(device)
    out_ch = 4
    model = ModMLP(input_ch, out_ch, args.netwidth, n_hidden_layers=args.netdepth).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, input_ch_views=input_ch_views,
                          output_ch=4, skips=[4], use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    # Network query function
    def network_query_fn(inputs, viewdirs, network_fn):
        return run_network(inputs, viewdirs, network_fn,
                           embed_fn, embeddirs_fn,
                           netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'network_fn': model,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
    }
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = 0.
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, model, model_fine, grad_vars

# -----------------------------------------------------------------------------
# Argparse
# -----------------------------------------------------------------------------
def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument("--expname", type=str, required=False)
    parser.add_argument("--basedir", type=str, default='./logs/')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern')
    # Training options
    parser.add_argument("--netdepth", type=int, default=8)
    parser.add_argument("--netwidth", type=int, default=256)
    parser.add_argument("--netdepth_fine", type=int, default=8)
    parser.add_argument("--netwidth_fine", type=int, default=256)
    parser.add_argument("--N_rand", type=int, default=32*32*4)
    parser.add_argument("--lrate", type=float, default=5e-4)
    parser.add_argument("--lrate_decay", type=int, default=250)
    parser.add_argument("--chunk", type=int, default=1024*32)
    parser.add_argument("--netchunk", type=int, default=1024*64)
    parser.add_argument("--N_samples", type=int, default=64)
    parser.add_argument("--N_importance", type=int, default=0)
    parser.add_argument("--perturb", type=float, default=1.)
    parser.add_argument("--use_viewdirs", action='store_true')
    parser.add_argument("--i_embed", type=int, default=0)
    parser.add_argument("--multires", type=int, default=10)
    parser.add_argument("--multires_views", type=int, default=4)
    parser.add_argument("--raw_noise_std", type=float, default=0.)
    parser.add_argument("--white_bkgd", action='store_true')
    parser.add_argument("--half_res", action='store_true')
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--no_ndc", action='store_true')
    parser.add_argument("--spherify", action='store_true')
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--i_print", type=int, default=100)
    parser.add_argument("--i_img", type=int, default=500)
    parser.add_argument("--i_weights", type=int, default=10000)
    parser.add_argument("--i_testset", type=int, default=200000)
    parser.add_argument("--i_video", type=int, default=200000)
    return parser

# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------
def train():
    parser = config_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(0); torch.manual_seed(0)

    # Load data (LLFF/blender/deepvoxels)
    if 'llff' in args.datadir:
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir, args.factor, recenter=True,
            bd_factor=.75, spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        i_val = i_test
        i_train = [i for i in range(images.shape[0]) if i not in i_test]
        near, far = (0., 1.) if not args.no_ndc else (bds.min()*0.9, bds.max()*1.)
    else:
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, testskip=0)
        i_train, i_val, i_test = i_split
        near, far = 2., 6.
        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:]) if args.white_bkgd else images[...,:3]

    H, W, focal = hwf; H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir
    basedir = args.basedir; expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    writer = SummaryWriter(os.path.join(basedir, expname, 'summaries'))

    # Create NeRF and render kwargs
    render_kwargs_train, render_kwargs_test, model, model_fine, grad_vars = create_nerf(args, device)
    optimizer = torch.optim.Adam(grad_vars, lr=args.lrate)
    global_step = 0

    # Move data to device
    render_poses = torch.Tensor(render_poses).to(device)
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)

    # Prepare ray batch for training
    rays_rgb = None
    if True:
        # random ray batching
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses.cpu().numpy()], 0)
        rays = np.concatenate([rays, images.cpu().numpy()[:,None]], 1)
        rays = np.transpose(rays, [0,2,3,1,4])
        rays = rays[i_train]
        rays = np.reshape(rays, [-1,3,3])
        np.random.shuffle(rays)
        rays_rgb = torch.Tensor(rays).to(device)
        i_batch = 0

    N_iters = 200000
    for i in range(N_iters):
        # sample random rays
        batch = rays_rgb[i_batch:i_batch+args.N_rand]
        batch = batch.permute(1,0,2)
        batch_rays, target_s = batch[:2], batch[2]
        i_batch += args.N_rand
        if i_batch >= rays_rgb.shape[0]:
            perm = torch.randperm(rays_rgb.shape[0]); rays_rgb = rays_rgb[perm]; i_batch = 0

        # core optimization loop
        rgb, disp, acc, extras = render(
            H, W, focal, chunk=args.chunk,
            rays=(batch_rays[0], batch_rays[1]),
            **render_kwargs_train)
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss + (img2mse(extras['rgb0'], target_s) if 'rgb0' in extras else 0.)
        psnr = mse2psnr(img_loss)
        loss.backward(); optimizer.step()

        # learning rate decay
        new_lrate = args.lrate * (0.1 ** (global_step / (args.lrate_decay*1000)))
        for pg in optimizer.param_groups:
            pg['lr'] = new_lrate

        # logging
        if i % args.i_print == 0:
            print(f"[TRAIN] Iter: {i} Loss: {loss.item():.4f} PSNR: {psnr.item():.4f}")
            writer.add_scalar('loss', loss.item(), i)
            writer.add_scalar('psnr', psnr.item(), i)

        # save checkpoints
        if i % args.i_weights == 0:
            torch.save(model.state_dict(), os.path.join(basedir, expname, f"model_{i:06d}.pth"))

        # render test/video
        if i % args.i_video == 0 and i > 0:
            with torch.no_grad():
                rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            rgb8 = to8b(rgbs)
            imageio.mimwrite(os.path.join(basedir, expname, f"video_{i:06d}.mp4"), rgb8, fps=30)

        global_step += 1

if __name__ == '__main__':
    train()
