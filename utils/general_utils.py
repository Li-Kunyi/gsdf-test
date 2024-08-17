#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random, math

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def get_linear_noise_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = lr_init * (1 - t) + lr_final * t
        return delay_rate * log_lerp

    return helper


def get_minimum_axis(scales, rotations):
    sorted_idx = torch.argsort(scales, descending=False, dim=-1)
    R = build_rotation(rotations)
    R_sorted = torch.gather(R, dim=1, index=sorted_idx[:, :, None].repeat(1, 1, 3)).squeeze()
    x_axis = R_sorted[:, 0, :]  # normalized by defaut

    return x_axis

def get_sorted_axis(scales, rotations):
    scale_sorted, sorted_idx = torch.sort(scales, descending=False, dim=-1)
    R = build_rotation(rotations)
    R_sorted = torch.gather(R, dim=1, index=sorted_idx[:, :, None].repeat(1, 1, 3)).squeeze()
    return R_sorted, scale_sorted


def flip_align_view(normal, viewdir):
    # normal: (N, 3), viewdir: (N, 3)
    dotprod = torch.sum(
        normal * -viewdir, dim=-1, keepdims=True)  # (N, 1)
    non_flip = dotprod >= 0  # (N, 1)
    normal_flipped = normal * torch.where(non_flip, 1, -1)  # (N, 3)
    return normal_flipped, non_flip


def depth2normal(depth: torch.Tensor, focal: float = None):
    if depth.dim() == 2:
        depth = depth[None, None]
    elif depth.dim() == 3:
        depth = depth.squeeze()[None, None]
    if focal is None:
        focal = depth.shape[-1] / 2 / np.tan(torch.pi / 6)
    depth = torch.cat([depth[:, :, :1], depth, depth[:, :, -1:]], dim=2)
    depth = torch.cat([depth[..., :1], depth, depth[..., -1:]], dim=3)
    kernel = torch.tensor([[[0, 0, 0],
                            [-.5, 0, .5],
                            [0, 0, 0]],
                           [[0, -.5, 0],
                            [0, 0, 0],
                            [0, .5, 0]]], device=depth.device, dtype=depth.dtype)[:, None]
    normal = torch.nn.functional.conv2d(depth, kernel, padding='valid')[0].permute(1, 2, 0)
    normal = normal / (depth[0, 0, 1:-1, 1:-1, None] + 1e-10) * focal
    normal = torch.cat([normal, torch.ones_like(normal[..., :1])], dim=-1)
    normal = normal / normal.norm(dim=-1, keepdim=True)
    return normal.permute(2, 0, 1)

def coordinates(voxel_dim, device: torch.device, flatten=True):
    if type(voxel_dim) is int:
        nx = ny = nz = voxel_dim
    else:
        nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    if not flatten:
        return torch.stack([x, y, z], dim=-1)

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3).float().cuda()
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points

def get_edge_aware_distortion_map(gt_image, distortion_map):
    grad_img_left = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, :-2]), 0)
    grad_img_right = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, 2:]), 0)
    grad_img_top = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, :-2, 1:-1]), 0)
    grad_img_bottom = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 2:, 1:-1]), 0)
    max_grad = torch.max(torch.stack([grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1), dim=-1)[0]
    # pad
    max_grad = torch.exp(-max_grad)
    max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
    return distortion_map * max_grad


def get_rays_from_uv(i, j, R, T, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.

    """
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R).to(device)
        T = torch.from_numpy(T).to(device)

    dirs = torch.stack(
        [(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], dim=-1).to(device)
    dirs = dirs.reshape(-1, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * R, -1)
    rays_o = T.expand(rays_d.shape)
    return rays_o, rays_d

def select_uv(i, j, n, color, device='cuda:0'):
    """
    Select n uv from dense uv.

    """
    channel = color.shape[-1]
    depth = color[..., -1]
    non_zero_idx = (depth.reshape(-1) > 0).nonzero()
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n,), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]  # (n)
    j = j[indices]  # (n)
    color = color.reshape(-1, channel)
    color = color[indices]  # (n,3)
    return i, j, color

def get_sample_uv(H0, H1, W0, W1, n, color, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    color = color[H0:H1, W0:W1]
    i, j = torch.meshgrid(torch.linspace(
        W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device))
    i = i.t()  # transpose
    j = j.t()
    i, j, color = select_uv(i, j, n, color, device=device)
    return i, j, color

def get_all_rays(H, W, fx, fy, cx, cy, R, T, device):
    """
    Get rays for a whole image.

    """
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()

    dirs = torch.stack(
        [(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], dim=-1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * R, -1)  # [H, W, 3]
    rays_o = T.expand(rays_d.shape)
    return rays_o, rays_d

def get_samples(H0, H1, W0, W1, n, fx, fy, cx, cy, R, T, color, device):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.
    """
    i, j, sample_color = get_sample_uv(
        H0, H1, W0, W1, n, color, device=device)
    rays_o, rays_d = get_rays_from_uv(i, j, R, T, fx, fy, cx, cy, device)
    return rays_o, rays_d, sample_color

def near_far_from_cube(self, rays_o, rays_d, bound):
    tmin = (-bound - rays_o) / (rays_d + 1e-15) # [N, 3]
    tmax = (bound - rays_o) / (rays_d + 1e-15)
    near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)
    far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)
    # if far < near, means no intersection, set both near and far to inf (1e9 here)
    mask = far < near
    near[mask] = 1e9
    far[mask] = 1e9
    # restrict near to a minimal value
    near = torch.clamp(near, min=self.near)
    far = torch.clamp(far, max=self.far)
    return near, far

def sample_along_rays(gt_depth, n_samples, n_surface, device):
    # [N, C]
    gt_depth = gt_depth.reshape(-1, 1)  # [n_pixels, 1]
    
    gt_none_zero_mask = gt_depth > 0
    gt_none_zero = gt_depth[gt_none_zero_mask]
    gt_none_zero = gt_none_zero.unsqueeze(-1)

    # for thoe with validate depth, sample near the surface
    gt_depth_surface = gt_none_zero.repeat(1, n_surface)  # [n_pixels, n_samples//2]
    t_vals_surface = torch.rand(n_surface).to(device)
    z_vals_surface_depth_none_zero = (gt_depth_surface * 0.95) * (1.-t_vals_surface) + (gt_depth_surface * 1.05) * (t_vals_surface)
    z_vals_near_surface = torch.zeros(gt_depth.shape[0], n_surface).to(device)
    gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
    z_vals_near_surface[gt_none_zero_mask, :] = z_vals_surface_depth_none_zero

    # for those with zero depth, random sample along the space
    near = torch.min(gt_depth) - 0.1
    far = torch.max(gt_depth) + 0.1
    t_vals_surface = torch.rand(n_surface).to(device)
    z_vals_surface_depth_zero = near * (1.-t_vals_surface) + far * (t_vals_surface)
    z_vals_surface_depth_zero.unsqueeze(0).repeat((~gt_none_zero_mask).sum(), 1)
    z_vals_near_surface[~gt_none_zero_mask, :] = z_vals_surface_depth_zero

    # none surface
    if n_samples > 0:
        gt_depth_samples = gt_depth.repeat(1, n_samples)  # [n_pixels, n_samples]
        near = gt_depth_samples * 0.01
        far = gt_depth_samples * 1.1
        t_vals = torch.linspace(0., 1., steps=n_samples, device=device)
        z_vals = near * (1.-t_vals) + far * (t_vals)

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_near_surface], -1), -1)  # [n_pixels, n_samples]
    else:
        z_vals, _ = torch.sort(z_vals_near_surface, -1)

    return z_vals_near_surface.float(), z_vals.float()

def getVoxels(x_max, x_min, y_max, y_min, z_max, z_min, voxel_size=None, resolution=None):

    if not isinstance(x_max, float):
        x_max = float(x_max)
        x_min = float(x_min)
        y_max = float(y_max)
        y_min = float(y_min)
        z_max = float(z_max)
        z_min = float(z_min)
    
    if voxel_size is not None:
        Nx = round((x_max - x_min) / voxel_size + 0.0005)
        Ny = round((y_max - y_min) / voxel_size + 0.0005)
        Nz = round((z_max - z_min) / voxel_size + 0.0005)

        tx = torch.linspace(x_min, x_max, Nx + 1)
        ty = torch.linspace(y_min, y_max, Ny + 1)
        tz = torch.linspace(z_min, z_max, Nz + 1)
    else:
        tx = torch.linspace(x_min, x_max, resolution)
        ty = torch.linspace(y_min, y_max,resolution)
        tz = torch.linspace(z_min, z_max, resolution)


    return tx, ty, tz