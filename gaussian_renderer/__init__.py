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
import os
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F

import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal

# import nvdiffrast.torch as dr
from pbr import CubemapLight, pbr_shading_2dgs, gsir_deferred_shading
from utils.general_utils import safe_normalize, reflect, dot
import numpy as np

def linear_to_srgb(linear: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError

# Tone Mapping
def aces_film(rgb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    EPS = 1e-6
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    rgb = (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e)
    if isinstance(rgb, np.ndarray):
        return rgb.clip(min=0.0, max=1.0)
    elif isinstance(rgb, torch.Tensor):
        return rgb.clamp(min=0.0, max=1.0)

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
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

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }


    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    # render_normal.retain_grad()

    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })

    return rets

def pbr_render_fw(viewpoint_camera, pc: GaussianModel, 
               light:CubemapLight, pipe, bg_color : torch.Tensor, 
               brdf_lut: Optional[torch.Tensor] = None, 
               speed=False, 
               scaling_modifier = 1.0, ):
    """
    forward shading, <arm> parameterazation
    """
    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    numG = means3D.shape[0]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # precompute color for each local surfel
    # for now for simplity, just use (cam_center - gs_3d_center).normalized as wo, it should be (intersection_point - gs_3d_center).normalized
    # to do the latter, should change cuda.

    #prepare all input for pbr: material, wi, wo, normals, light
    view_pos = viewpoint_camera.camera_center.repeat(numG, 1) # (numG, 3)
    wo_W = safe_normalize(view_pos - means3D) # (numG, 3) wo directs from gs to camera

    ngw = pc.get_normals
    normalsG_W = ngw  # (numG, 3)
    # try:
    #     normalsG_W.retain_grad()
    # except:
    #     pass

    cos = dot(normalsG_W, wo_W) # (numG, 1)
    mul = torch.where(cos > 0, 1., -1.) # (numG, 1)
    normalsG_W = normalsG_W * mul # (numG, 3)

    wi_W = safe_normalize(reflect(wo_W, normalsG_W)) # (numG, 3)

    albedo=pc.get_albedo
    roughness=pc.get_roughness
    metallic=pc.get_metallic

    results = pbr_shading_2dgs(light = light, 
                              normals=normalsG_W[None, None,:,:], # ( 1, 1, numG, 3)
                              wo=wo_W, # (numG, 3)
                              wi=wi_W[None, None,:,:],# ( 1, 1, numG, 3)
                              albedo=albedo,
                              roughness=roughness,
                              metallic=metallic,
                              brdf_lut = brdf_lut
                              )

    colors_precomp = results["rgb"] # [numG, 3]
    diffuse_color = results["diffuse"] # [numG, 3]
    specular_color = results["specular"] # [numG, 3]
    diffuse_light = results["diffuse_light"] # [numG, 3]
    specular_light = results["specular_light"] # [numG, 3]

    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

    
    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    # render_normal = safe_normalize(render_normal)

    try:
        gt_mask = viewpoint_camera.gt_normal_mask.cuda()
        print("process forward shading with gt mask")
    except:
        gt_mask = None
    if gt_mask is not None:
        mask = gt_mask
    else:
        mask = (render_normal != 0).all(0, keepdim=True)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

   
    rendered_image = torch.where(mask, rendered_image, bg_color[:,None,None])
    render_normal = torch.where(mask, render_normal, torch.zeros_like(render_normal))
    surf_normal = torch.where(mask, surf_normal, torch.zeros_like(surf_normal))
    render_dist = torch.where(mask, render_dist, torch.zeros_like(render_dist))
    render_alpha = torch.where(mask, render_alpha, torch.zeros_like(render_alpha))

    if pipe.tone:
        rendered_image = aces_film(rendered_image)
    else:
        rendered_image = rendered_image.clamp(min=0.0, max=1.0)
    if pipe.gamma:
        rendered_image = linear_to_srgb(rendered_image.squeeze())


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }

    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            # 'ng_w': normalsG_W
    })

    
    if speed:
        return rets

    render_extras = {
        "diffuse_rgb": diffuse_color,
        "specular_rgb": specular_color,
        "diffuse_light": diffuse_light,
        "specular_light": specular_light,
        "albedo": albedo,
        "roughness": roughness.repeat(1, 3),
        "metallic": metallic.repeat(1, 3) 
        # if w_metallic else None,
    }

    out_extras = {}
    with torch.no_grad():
        for k in render_extras.keys():
            if render_extras[k] is None: continue
            image = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = render_extras[k],
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)[0]
            out_extras[k] = torch.where(mask, image, bg_color[:,None,None])
    rets.update(out_extras)
    # if not w_metallic:
    #     rets["metallic"] = torch.ones_like(rets["roughness"]).cuda()


    return rets



def pbr_render_df(viewpoint_camera, 
                  pc: GaussianModel, 
                  light:CubemapLight, 
                  pipe, 
                  bg_color: torch.Tensor, 
                  view_dirs : torch.Tensor, #[H,W,3] 
                  brdf_lut: Optional[torch.Tensor] = None, 
                  speed=False, scaling_modifier = 1.0):
    
    """
    deffered shading, <arm> parameterazation
    """
    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

     # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    albedo=pc.get_albedo
    roughness=pc.get_roughness
    metallic= pc.get_metallic

    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = albedo,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None
    )

    deffered_input ={}
    deffered_input["albedo"] = rendered_image

    render_alpha = allmap[1:2]
    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    try:
        gt_mask = viewpoint_camera.gt_normal_mask.cuda()
        print("process deferred shading with gt mask")
    except:
        gt_mask = None
    if gt_mask is not None:
        mask = gt_mask
    else:
        mask = (render_normal != 0).all(0, keepdim=True)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    # # just for verify the normals
    # normalsG_W = pc.get_normals
    # numG = means3D.shape[0]
    # cos = dot(normalsG_W, view_dirs[400,400,0:3].repeat(numG, 1)) # (numG, 1)
    # mul = torch.where(cos > 0, 1., -1.) # (numG, 1)
    # normalsG_W = normalsG_W * mul # (numG, 3)

    pre_blend = {
        "metallic": metallic.repeat(1, 3),
        "roughness": roughness.repeat(1, 3),
        # "normals_blended": normalsG_W,
    }

    
    # here should keep tracking gradient of all input
    for k in pre_blend.keys():
        if pre_blend[k] is None: continue
        image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = pre_blend[k],
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)[0]
        deffered_input[k] = image
    
    
    
    # close = torch.isclose(deffered_input["normals_blended"], rn, rtol=1e-3, atol=1e-3)
    # num_close = close.all(dim=0).sum().item()
    # print(f"normals close: {num_close}/{close.shape[2]}*{close.shape[1]}")
    # assert torch.allclose(deffered_input["normals_blended"], rn, rtol=1e-2, atol=1e-2), "normals not match"

  
    results, extras = gsir_deferred_shading(light, 
                                    render_normal.permute(1,2,0).contiguous(), 
                                    view_dirs, #already normalized outside
                                    deffered_input["albedo"].permute(1,2,0).contiguous(), 
                                    deffered_input["roughness"][0,:,:][None,:,:].permute(1,2,0).contiguous(), 
                                    brdf_lut,
                                    metallic= deffered_input["metallic"][0,:,:][None,:,:].permute(1,2,0).contiguous()
                                    )
    
    
    rendered_image = results
    
    rendered_image = torch.where(mask, rendered_image, bg_color[:,None,None])
    render_normal = torch.where(mask, render_normal, torch.zeros_like(render_normal))
    surf_normal = torch.where(mask, surf_normal, torch.zeros_like(surf_normal))
    render_dist = torch.where(mask, render_dist, torch.zeros_like(render_dist))
    render_alpha = torch.where(mask, render_alpha, torch.zeros_like(render_alpha))

    if pipe.tone:
        rendered_image = aces_film(rendered_image)
    else:
        rendered_image = rendered_image.clamp(min=0.0, max=1.0)
    if pipe.gamma:
        rendered_image = linear_to_srgb(rendered_image.squeeze())


    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    }

    if speed:
        return rets



    for key in deffered_input.keys():
        if deffered_input[key] is not None:
            print(key)
            deffered_input[key] = torch.where(mask, deffered_input[key], bg_color[:,None,None])
    for key in extras.keys():
        if extras[key] is not None:
            print(key)
            extras[key] = torch.where(mask, extras[key], bg_color[:,None,None])

    rets.update(deffered_input) # albedo, roughness, metallic
    rets.update(extras) #diffuse_light, specular_light, diffuse_rgb, specular_rgb
    return rets


def pbr_render_st(viewpoint_camera, pc: GaussianModel, 
                  light:CubemapLight, pipe, bg_color : torch.Tensor, 
                  view_dirs : torch.Tensor, #[H,W,3]
                  brdf_lut: Optional[torch.Tensor] = None, 
                  speed=False, fw_rate=0.5, scaling_modifier = 1.0):
    """
    mixed stochastic shading with rate fw_rate, <arm> parameterazation
    """

    assert fw_rate > 0 and fw_rate < 1, "fw_rate should be in (0,1)"

    render_pkg_fw = pbr_render_fw(
                viewpoint_camera=viewpoint_camera,
                pc=pc,
                light=light,
                pipe=pipe,
                bg_color=bg_color,
                brdf_lut=brdf_lut,
                speed=speed,
                )
    

    render_pkg_df = pbr_render_df(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        light=light,
        pipe=pipe,
        bg_color=bg_color,
        view_dirs = view_dirs,
        brdf_lut= brdf_lut,
        speed=speed,
        )
    
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    fw_mask = torch.rand(H, W, device="cuda") < fw_rate # [H,W]
    
    render_pkg = render_pkg_fw
    for key in ["render", "albedo", "roughness", "metallic", 
                "diffuse_rgb", "specular_rgb", 
                "diffuse_light", "specular_light"]:
        if render_pkg_fw[key] is not None:
            render_pkg[key] = fw_mask[None,:, :] * render_pkg_fw[key] + (~fw_mask[None,:, :]) * render_pkg_df[key]
    
    return render_pkg
