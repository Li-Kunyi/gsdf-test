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
import math
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.sdf_gaussian_model_v3 import GaussianModel
# from scene.gaussian_model import GaussianModel
import torch.nn.functional as F
from utils.sh_utils import eval_sh
from utils.general_utils import depth_to_normal, get_samples, sample_along_rays, get_all_rays

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size: float, scaling_modifier = 1.0, override_color = None, subpixel_offset=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    view2gaussian_precomp = None
    # pipe.compute_view2gaussian_python = True
    if pipe.compute_view2gaussian_python:
        view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # # we local direction
            # cam_pos_local = view2gaussian_precomp[:, 3, :3]
            # cam_pos_local_scaled = cam_pos_local / scales
            # dir_pp = -cam_pos_local_scaled
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=view2gaussian_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def sdf_render_v2(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size: float, scaling_modifier = 1.0, override_color = None, subpixel_offset=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    with torch.no_grad():
        frustum_mask, _, _ = project_to_image(viewpoint_camera, pc.get_xyz)

    means3D = pc.get_xyz
    means3D = means3D[frustum_mask]

    means2D = screenspace_points
    means2D = means2D[frustum_mask]

    gaussian_sdf = pc.query_sdf(means3D)
    opacity = torch.clip(pc.opacity_activation(gaussian_sdf), max=1.0)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
        cov3D_precomp = cov3D_precomp[frustum_mask]
    else:
        scales = pc.get_scaling
        scales = scales[frustum_mask]
        rotations = pc.get_rotation
        rotations = rotations[frustum_mask]

    view2gaussian_precomp = None
    # pipe.compute_view2gaussian_python = True
    if pipe.compute_view2gaussian_python:
        view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # # we local direction
            # cam_pos_local = view2gaussian_precomp[:, 3, :3]
            # cam_pos_local_scaled = cam_pos_local / scales
            # dir_pp = -cam_pos_local_scaled
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            shs = shs[frustum_mask]
    else:
        colors_precomp = override_color
        colors_precomp = colors_precomp[frustum_mask]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=view2gaussian_precomp)

    gaussian_gradient = None
    # gaussian_gradient = gradient(means3D, pc.query_sdf)
    # gaussian_gradient = torch.norm(gaussian_gradient, dim=-1)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "gaussian_sdf": gaussian_sdf,
            "opacity": opacity,
            "frustum_mask": frustum_mask,
            "gaussian_gradient": gaussian_gradient,
            }


def sdf_render_v3(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size: float, scaling_modifier = 1.0, override_color = None, subpixel_offset=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    with torch.no_grad():
        frustum_mask, _, _ = project_to_image(viewpoint_camera, pc.get_xyz)
    mask_xyz = means3D[frustum_mask]
    dir_pp = (mask_xyz - viewpoint_camera.camera_center.repeat(mask_xyz.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

    out = pc.query_sdf(mask_xyz, dir=dir_pp_normalized, return_opacity=True, return_rot_scale=True, return_color=False)
    gsdf, gopacity, gscale, grot, gcolor = out['sdf'], out['opacity'], out['scale'], out['rot'], out['color']
    opacity = torch.zeros_like(pc.get_xyz[:, :1], dtype=pc.get_xyz.dtype, device="cuda")
    opacity[frustum_mask] = gopacity

    scales = torch.zeros((opacity.shape[0], 3), dtype=pc.get_xyz.dtype, device="cuda")
    scales[frustum_mask] = gscale
    rotations = torch.zeros((opacity.shape[0], 4), dtype=pc.get_xyz.dtype, device="cuda")
    rotations[frustum_mask] = grot

    cov3D_precomp = None

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # scales = None
    # rotations = None
    # cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    #     cov3D_precomp = cov3D_precomp
    # else:
    #     scales = pc.get_scaling
    #     rotations = pc.get_rotation

    view2gaussian_precomp = None
    # # pipe.compute_view2gaussian_python = True
    # if pipe.compute_view2gaussian_python:
    #     view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)
    
    shs = None
    if gcolor is not None:
        colors_precomp = torch.zeros((opacity.shape[0], 3), dtype=pc.get_xyz.dtype, device="cuda")
        colors_precomp[frustum_mask] = gcolor
    else:
        colors_precomp = None
        gaussian_color = None
        if gaussian_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                # # we local direction
                # cam_pos_local = view2gaussian_precomp[:, 3, :3]
                # cam_pos_local_scaled = cam_pos_local / scales
                # dir_pp = -cam_pos_local_scaled
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = gaussian_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=view2gaussian_precomp)

    gaussian_gradient = None

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "gaussian_sdf": gsdf,
            "opacity": gopacity,
            "gaussian_gradient": gaussian_gradient,
            "frustum_mask": frustum_mask,
            }

def get_sdf_loss_with_gaussian_depth(gaussians, c2w, fx, fy, depth, alpha, normal,
                                     n_pixel=2048, n_sample=96, n_sample_surface=21,
                                     truncation=0.1, full_image=False, ray_sampling=True, 
                                     device='cuda'):  
    with torch.no_grad():
        H, W = depth.shape
        R = c2w[:3, :3]
        T = c2w[:3, 3]

        if full_image:
            rays_o, rays_d = get_all_rays(H, W, fx, fy, (W)/2, (H)/2, R, T, device)
            rays_o = rays_o.reshape(-1, 3)  # [N, 3]
            rays_d = rays_d.reshape(-1, 3)
            sp = torch.cat((depth[..., None], alpha[:, :, None]), dim=-1).reshape(-1, 5)
        else:
        
            rays_o, rays_d, sp = get_samples(0, H-0, 0, W-0, n_pixel, fx, fy, (W-1)/2, (H-1)/2, R, T,
                                            torch.cat((depth[:, :, None], alpha[:, :, None], normal), dim=-1),
                                            device)  # [n_pixels, C]
        
        depth_sp = sp[..., 0:1]
        alpha_sp = sp[..., 1:2]
        normal_sp = sp[..., 2:]

        if ray_sampling:
            z_vals_surface, z_vals = sample_along_rays(depth_sp, n_sample, n_sample_surface, device)  # [n_pixels, n_samples]
        else:
            z_vals = depth_sp

        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # [n_rays, n_samples, 3]

    out = gaussians.query_sdf(pts.reshape(-1, 3))
    sdf = out['sdf']
    sdf = sdf.reshape(pts.shape[0], pts.shape[1])

    # before truncation
    front_mask = torch.where(z_vals < (depth_sp - truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # after truncation
    back_mask = torch.where(z_vals > (depth_sp + truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # valid mask
    depth_mask = torch.where(depth_sp > 0.0, torch.ones_like(depth_sp), torch.zeros_like(depth_sp))

    # Valid sdf region
    sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask
    front_mask = front_mask * depth_mask

    volume_fs_loss = (((sdf * front_mask - torch.ones_like(sdf) * front_mask) ** 2)).mean()
    volume_sdf_loss = ((((z_vals + sdf) * sdf_mask - depth_sp * sdf_mask) ** 2)).mean()

    return volume_fs_loss, volume_sdf_loss


def gradient(x, fn, voxel_size=0.001):    
    x = torch.reshape(x, [-1, x.shape[-1]]).float()
    eps = voxel_size / np.sqrt(3)
    k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  
    k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  
    k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  
    k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  
    
    sdf1 = fn(x + k1 * eps)
    sdf2 = fn(x + k2 * eps)
    sdf3 = fn(x + k3 * eps)
    sdf4 = fn(x + k4 * eps)
    gradients = (k1 * sdf1 + k2 * sdf2 + k3 * sdf3 + k4 * sdf4) / (4.0 * eps)
    
    return gradients

def sample_ellipse_planes(normals, U, V, u_scale, v_scale, centers, num_samples=9):
    """
    Sample points on multiple ellipse planes.

    Args:
        normals (torch.Tensor): Normal vectors of the planes.
        U, V (torch.Tensor): Axes of ellipse planes.
        centers (torch.Tensor): Center points of the ellipses in world coordinates.
        u_scale (torch.Tensor): Lengths of the major axes.
        v_scale (torch.Tensor): Lengths of the minor axes.
        num_samples (int): Number of points to sample on each plane.

    Returns:
        torch.Tensor: Sampled points on the ellipse planes in world coordinates.
    """
    # Generate random angles for sampling points
    u_sample = torch.rand(normals.shape[0], num_samples).to(normals.device)
    v_sample = torch.rand(normals.shape[0], num_samples).to(normals.device)

    # Compute sampled points in local coordinates for all planes
    u = 1.2 * u_scale[:, None] * u_sample
    v = 1.2 * v_scale[:, None] * v_sample

    # Transform local coordinates to world coordinates for all planes
    local_U = u[:, :, None] * U[:, None, :]
    local_V = v[:, :, None] * V[:, None, :]

    world_points = local_U + local_V + centers[:, None, :]

    return world_points # [n_gaussian, n_sample, 3]

def project_to_image(viewpoint_camera, pts, device="cuda"):
    # for each gaussian, project to image coordinate to match depth
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    K = torch.eye(3).to(device)
    K[0, 0] = W / (2 * math.tan(viewpoint_camera.FoVx / 2))
    K[1, 1] = H / (2 * math.tan(viewpoint_camera.FoVy / 2))
    K[0, 2] = (W - 1) / 2
    K[1, 2] = (H - 1) / 2
    rel_w2c = viewpoint_camera.world_view_transform.T
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]
    u, v, d = transformed_pts[:, 0], transformed_pts[:, 1], transformed_pts[:, 2]
    image_coordinates = (K @ (torch.stack([u / d, v / d, d / d], dim=1)).T).T[:, :2]
    valid_indices = (image_coordinates[:, 0] > -10) & (image_coordinates[:, 0] < W+10) & \
                    (image_coordinates[:, 1] > -10) & (image_coordinates[:, 1] < H+10) & \
                    (d > 0)
    valid_coordinates = torch.round(image_coordinates[valid_indices, :]).long()
    proj_depth = d[valid_indices]
    return valid_indices, valid_coordinates, proj_depth

# def project_to_image(viewpoint_camera, pts, normal, dir_pp_normalized, device="cuda"):
#     # for each gaussian, project to image coordinate to match depth
#     H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
#     K = torch.eye(3).to(device)
#     K[0, 0] = W / (2 * math.tan(viewpoint_camera.FoVx / 2))
#     K[1, 1] = H / (2 * math.tan(viewpoint_camera.FoVy / 2))
#     K[0, 2] = (W - 1) / 2
#     K[1, 2] = (H - 1) / 2
#     rel_w2c = viewpoint_camera.world_view_transform.T
#     pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
#     pts4 = torch.cat((pts, pts_ones), dim=1)
#     transformed_pts = (rel_w2c @ pts4.T).T[:, :3]
#     u, v, d = transformed_pts[:, 0], transformed_pts[:, 1], transformed_pts[:, 2]
#     image_coordinates = (K @ (torch.stack([u / d, v / d, d / d], dim=1)).T).T[:, :2]
#     valid_indices = (image_coordinates[:, 0] > 0) & (image_coordinates[:, 0] < W-1) & \
#                     (image_coordinates[:, 1] > 0) & (image_coordinates[:, 1] < H-1) & \
#                     (d > 0) & ((normal * dir_pp_normalized).sum(dim=1) < 0)
#     valid_coordinates = torch.round(image_coordinates[valid_indices, :]).long()
#     proj_depth = d[valid_indices]
#     return valid_indices, valid_coordinates, proj_depth


def get_values(mlp, pts, reflect_normals, scales, num_process=10000, device='cuda'):    
    batch_size = pts.shape[0]
        
    density = []
    sdf = []
    sdf_gradient = []
    sh = []
    for i in range(batch_size // num_process + 1):
        start = i * num_process
        end = min(batch_size, (i + 1) * num_process)
        if end - start > 0:
            _density, _sdf, _gradient, _sh = mlp.query_sdf_gradient_sh(pts[start:end], reflect_normals[start:end])
            density.append(_density)
            sdf.append(_sdf)
            sdf_gradient.append(_gradient)
            sh.append(_sh)

    density = torch.cat(density, dim=0).to(device)
    free_energy = scales[:, None] * density
    alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
    sdf = torch.cat(sdf, dim=0).to(device)
    sdf_gradient = torch.cat(sdf_gradient, dim=0).to(device)
    sh = torch.cat(sh, dim=0).to(device)

    return alpha, sdf, sdf_gradient, sh


def get_sdf_disk(mlp, pts, num_process=10000, device='cuda'):        
    batch_size, n_sample, c = pts.shape
    sdf = []
    for i in range(batch_size // num_process + 1):
        start = i * num_process
        end = min(batch_size, (i + 1) * num_process)
        if end - start > 0:
            _sdf = mlp.query_sdf(pts[start:end].reshape(-1, c))
            sdf.append(_sdf.reshape(-1, n_sample, 1))
            
    sdf = torch.cat(sdf, dim=0).to(device)
    return sdf


def volume_rendering(mlp, camera, depth, n_sample=10, n_sample_surface=11, num_process=4096, truncation=0.2, full_image=False, device='cuda'):  
    H, W = depth.shape
    fx = W / (2 * math.tan(camera.FoVx  / 2))
    fy = H / (2 * math.tan(camera.FoVy  / 2))
    w2c = camera.world_view_transform.T
    c2w = torch.linalg.inv(w2c)
    R = c2w[:3, :3]
    T = c2w[:3, 3]
    
    if full_image:
        rays_o, rays_d = get_all_rays(H, W, fx, fy, (W-1)/2, (H-1)/2, R, T, device)
        rays_o = rays_o.reshape(-1, 3)  # [N, 3]
        rays_d = rays_d.reshape(-1, 3)
        depth_sp = depth.reshape(-1, 1)
    else:
        rays_o, rays_d, depth_sp = get_samples(H, W, 8192, fx, fy, (W-1)/2, (H-1)/2, R, T,
                                        depth[:, :, None], device)  # [n_pixels, C]

    z_vals = sample_along_rays(depth_sp, n_sample, n_sample_surface, device)  # [n_pixels, n_samples]

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
    
    batch_size, n_sample, c = pts.shape
    sdf = []
    density = []
    for i in range(batch_size // num_process + 1):
        start = i * num_process
        end = min(batch_size, (i + 1) * num_process)
        if end - start > 0:
            _density, _sdf = mlp.query_alpha_sdf(pts[start:end].reshape(-1, c))
            density.append(_density.reshape(-1, n_sample))
            sdf.append(_sdf.reshape(-1, n_sample))
    density = torch.cat(density, dim=0).to(device)        
    sdf = torch.cat(sdf, dim=0).to(device)

    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    free_energy = dists * density
    shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
    alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
    transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
    weights = alpha * transmittance # probability of the ray hits something her
    
    # rgb_map = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3)
    volume_depth = torch.sum(weights * z_vals, -1)  # (N_rays)

    front_mask = torch.where(z_vals < (depth_sp - truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # after truncation
    back_mask = torch.where(z_vals > (depth_sp + truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # valid mask
    depth_mask = torch.where(depth_sp > 0.0, torch.ones_like(depth_sp), torch.zeros_like(depth_sp))
    # Valid sdf region
    sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

    # volume_fs_loss = F.l1_loss(sdf * front_mask, torch.ones_like(sdf) * front_mask)
    volume_fs_loss = torch.tensor(0).to(sdf.device)
    volume_sdf_loss = F.l1_loss((z_vals + sdf) * sdf_mask, depth_sp * sdf_mask)
    # volume_sdf_loss = torch.tensor(0).to(sdf.device) 
    volume_depth_loss = F.l1_loss(volume_depth[:, None], depth_sp)
    # volume_depth_loss = F.l1_loss(volume_depth[:, None] * depth_mask, depth_sp * depth_mask)

    return volume_fs_loss, volume_sdf_loss, volume_depth_loss, volume_depth


def volume_rendering_full(mlp, camera, depth, n_sample=0, n_sample_surface=11, num_process=2048, truncation=0.1, full_image=False, mlp_warm_up=True, device='cuda'):  
    H, W = depth.shape
    fx = W / (2 * math.tan(camera.FoVx  / 2))
    fy = H / (2 * math.tan(camera.FoVy  / 2))
    w2c = camera.world_view_transform.T
    c2w = torch.linalg.inv(w2c)
    R = c2w[:3, :3]
    T = c2w[:3, 3]

    rays_o, rays_d = get_all_rays(H, W, fx, fy, (W-1)/2, (H-1)/2, R, T, device)
    rays_o = rays_o.reshape(-1, 3)  # [N, 3]
    rays_d = rays_d.reshape(-1, 3)
    depth_sp = depth.reshape(-1, 1)

    z_vals = sample_along_rays(depth_sp, n_sample, n_sample_surface, device)  # [n_pixels, n_samples]

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
    
    batch_size, n_sample, c = pts.shape
    sdf = []
    density = []
    for i in range(batch_size // num_process + 1):
        start = i * num_process
        end = min(batch_size, (i + 1) * num_process)
        if end - start > 0:
            _density, _sdf = mlp.query_alpha_sdf(pts[start:end].reshape(-1, c))
            density.append(_density.reshape(-1, n_sample))
            sdf.append(_sdf.reshape(-1, n_sample))
    density = torch.cat(density, dim=0).to(device)        
    sdf = torch.cat(sdf, dim=0).to(device)

    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    free_energy = dists * density
    shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
    alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
    transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
    weights = alpha * transmittance # probability of the ray hits something her
    
    # rgb_map = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3)
    volume_depth = torch.sum(weights * z_vals, -1)  # (N_rays)
    # valid mask
    depth_mask = torch.where(depth_sp > 0.0, torch.ones_like(depth_sp), torch.zeros_like(depth_sp))


    return volume_depth * depth_mask[:, 0]



def sdf_render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size: float,
               mlp = None, full_image=False, mlp_warm_up=False,
               scaling_modifier = 1.0, override_color = None, subpixel_offset=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity#_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling#_with_3D_filter
        rotations = pc.get_rotation

    view2gaussian_precomp = None
    # pipe.compute_view2gaussian_python = True
    if pipe.compute_view2gaussian_python:
        view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)

    # Three step rendering:
    # 1. render gaussian opacity(each gaussian).
    # 2. render sdf value of disk points(sample few gaussians).
    # 3. volume rendering(sample few pixels, using rendered depth from gs). GS model works as a coarse model
    hybrid = False
    if not mlp == None:
        # normal = (pc.get_normal_axis(dir_pp_normalized=dir_pp_normalized, return_delta=True))
        sorted_axis, sorted_scale = pc.get_sorted_axis()
        normal_axis = sorted_axis[:, 0, :]
        normal = normal_axis / normal_axis.norm(dim=1, keepdim=True)  # (N, 3)
        u_axis = sorted_axis[:, 1, :]
        u_axis = u_axis / u_axis.norm(dim=1, keepdim=True)  # (N, 3)
        v_axis = sorted_axis[:, 2, :]
        v_axis = v_axis / v_axis.norm(dim=1, keepdim=True)  # (N, 3)
        min_scales = sorted_scale[:, 0]
        u_scales = sorted_scale[:, 1]
        v_scales = sorted_scale[:, 2]

        ref_dir = 2 * (dir_pp_normalized * normal).sum(dim=-1, keepdim=True) * normal - dir_pp_normalized
        ref_dir= ref_dir / (torch.norm(ref_dir, dim=-1, keepdim=True) + 1e-8)

        disk_points = sample_ellipse_planes(normal, u_axis, v_axis, u_scales, v_scales, means3D, num_samples=9)
        
        if mlp_warm_up:
            mlp_opacity, gaussian_sdf, sdf_gradient, ref_shs_view = get_values(mlp, means3D.detach(), ref_dir.detach(), min_scales.detach())
            disk_sdf = get_sdf_disk(mlp, disk_points.detach())
            disk_sdf = torch.cat((disk_sdf.squeeze(-1), gaussian_sdf), dim=-1)
            hybrid = False
        else:
            mlp_opacity, gaussian_sdf, sdf_gradient, ref_shs_view = get_values(mlp, means3D, ref_dir, min_scales)
            opacity = mlp_opacity
            disk_sdf = get_sdf_disk(mlp, disk_points)
            disk_sdf = torch.cat((disk_sdf.squeeze(-1), gaussian_sdf), dim=-1)

            pc.set_opacity(opacity)
            hybrid = False

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)

    if not mlp == None and hybrid:            
        ref_shs_view = ref_shs_view.reshape(-1, 3, (pc.max_ref_sh_degree+1)**2)
        ref_rgb = eval_sh(pc.active_ref_sh_degree, ref_shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) + torch.clamp_min(ref_rgb + 0.5, 0.0)
    else:
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendering, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=view2gaussian_precomp)
    
    depth_map = rendering[6, :, :]
    volume_depth_map = torch.zeros_like(depth_map).to(depth_map.device)
    gaussian_density_map = torch.zeros_like(depth_map).to(depth_map.device)
    if not mlp == None:
        distortion_map = rendering[8, :, :]
        
        depth_normal_map, _ = depth_to_normal(viewpoint_camera, depth_map[None, ...])
        depth_normal_map = depth_normal_map.permute(2, 0, 1)

        render_normal_map = rendering[3:6, :, :]
        render_normal_map = torch.nn.functional.normalize(render_normal_map, p=2, dim=0)

        if mlp_warm_up:
            volume_fs_loss, volume_sdf_loss, volume_depth_loss, volume_depth = volume_rendering(mlp, viewpoint_camera, depth_map.detach(),
                                                                                                full_image=full_image)
        else:
            volume_fs_loss, volume_sdf_loss, volume_depth_loss, volume_depth = volume_rendering(mlp, viewpoint_camera, depth_map,
                                                                                                full_image=full_image)
        # volume_fs_loss = torch.tensor(0).to(means3D.device)
        # volume_sdf_loss =  torch.tensor(0).to(means3D.device)
        # volume_depth_loss =  torch.tensor(0).to(means3D.device)
        # volume_depth = torch.tensor(0).to(means3D.device)
        
        if full_image:
            # volume_depth = volume_rendering_full(mlp, viewpoint_camera, depth_map, full_image=full_image, mlp_warm_up=mlp_warm_up)
            volume_depth_map = volume_depth.reshape(depth_map.shape)

            valid_indices, valid_coordinates = project_to_image(viewpoint_camera, means3D)
            valid_opacity = mlp_opacity[valid_indices]

            for k in range(valid_coordinates.shape[0]):
                i = valid_coordinates[:, 1]
                j = valid_coordinates[:, 0]
                gaussian_density_map[i, j] += valid_opacity[k]

        return {"render": rendering,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "min_scales": min_scales,
                "depth_map": depth_map,
                "distortion_map": distortion_map,
                "depth_normal_map": depth_normal_map,
                "render_normal_map": render_normal_map,
                "gaussian_sdf": gaussian_sdf, 
                "sdf_gradient": sdf_gradient,
                "gaussian_normal": normal,
                "disk_sdf": disk_sdf,
                "volume_fs_loss": volume_fs_loss,
                "volume_sdf_loss": volume_sdf_loss, 
                "volume_depth_loss": volume_depth_loss,
                "volume_depth_map": volume_depth_map,
                "dir_pp_normalized": dir_pp_normalized,
                "gaussian_opacity": opacity.detach().clone(),
                "mlp_opacity": mlp_opacity,
                "gaussian_density_map": gaussian_density_map}
    else:
        return {"render": rendering,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "volume_depth_map": volume_depth_map,
                "gaussian_density_map": gaussian_density_map}

def integrate(points3D, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size: float, scaling_modifier = 1.0, override_color = None, subpixel_offset=None):
    """
    integrate Gaussians to the points, we also render the image for visual comparison. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    view2gaussian_precomp = None
    # pipe.compute_view2gaussian_python = True
    if pipe.compute_view2gaussian_python:
        view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # # we local direction
            # cam_pos_local = view2gaussian_precomp[:, 3, :3]
            # cam_pos_local_scaled = cam_pos_local / scales
            # dir_pp = -cam_pos_local_scaled
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, alpha_integrated, color_integrated, radii = rasterizer.integrate(
        points3D = points3D,
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=view2gaussian_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "alpha_integrated": alpha_integrated,
            "color_integrated": color_integrated,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}



def integrate_sdf(points3D, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size: float, scaling_modifier = 1.0, override_color = None, subpixel_offset=None):
    """
    integrate Gaussians to the points, we also render the image for visual comparison. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0.0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    gaussian_sdf = pc.query_sdf(means3D)
    opacity = pc.opacity_activation(gaussian_sdf)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    view2gaussian_precomp = None
    # pipe.compute_view2gaussian_python = True
    if pipe.compute_view2gaussian_python:
        view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # # we local direction
            # cam_pos_local = view2gaussian_precomp[:, 3, :3]
            # cam_pos_local_scaled = cam_pos_local / scales
            # dir_pp = -cam_pos_local_scaled
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, alpha_integrated, color_integrated, radii = rasterizer.integrate(
        points3D = points3D,
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=view2gaussian_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "alpha_integrated": alpha_integrated,
            "color_integrated": color_integrated,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}