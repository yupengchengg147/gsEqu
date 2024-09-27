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
import torch.nn.functional as F

from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import pbr_render_fw, pbr_render_df
import torchvision
from pbr import CubemapLight, get_brdf_lut

from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import apply_depth_colormap, turbo_cmap

import numpy as np


def render_set(model_path, name, iteration, views, gaussians, cubemap,  pipeline, background, canonical_rays, mode):

    brdf_lut = get_brdf_lut().cuda()

    # build mip for environment light
    cubemap.build_mips()
    envmap = cubemap.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
    os.makedirs(os.path.join(model_path, name), exist_ok=True)
    envmap_path = os.path.join(model_path, name, "envmap.png")
    torchvision.utils.save_image(envmap, envmap_path)

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    brdf_path = os.path.join(model_path, name, "ours_{}".format(iteration), "brdf")
    pbr_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pbr")
    alpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "alpha")
    normalRender_path = os.path.join(model_path, name, "ours_{}".format(iteration), "n_Render")
    normalDepth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "n_Depth")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    light_path = os.path.join(model_path, name, "ours_{}".format(iteration), "light")
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(brdf_path, exist_ok=True)
    makedirs(pbr_path, exist_ok=True)
    makedirs(alpha_path, exist_ok=True)
    makedirs(normalRender_path, exist_ok=True)
    makedirs(normalDepth_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(light_path, exist_ok=True)

    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()

        if mode == "fw":

            render_pkg = pbr_render_fw(
            viewpoint_camera=view,
            pc=gaussians,
            light=cubemap,
            pipe=pipeline,
            bg_color=background,
            brdf_lut=brdf_lut,
            speed=False,
            inference = True)
        
        elif mode == "df":

            H, W = view.image_height, view.image_width
            c2w = torch.inverse(view.world_view_transform.T)  # [4, 4]
            view_dirs = -(( F.normalize(canonical_rays[:, None, :], p=2, dim=-1)* c2w[None, :3, :3]).sum(dim=-1) #[HW,3]
                        .reshape(H, W, 3)) # direct from screen to cam center
            render_pkg = pbr_render_df(
                    viewpoint_camera=view,
                    pc=gaussians,
                    light=cubemap,
                    pipe=pipeline,
                    bg_color=background,
                    view_dirs = view_dirs,
                    brdf_lut= brdf_lut,
                    speed=False,
                    inference = True
                    )
            
        elif mode == "iterative":
            render_pkg_fw = pbr_render_fw(
                viewpoint_camera=view,
                pc=gaussians,
                light=cubemap,
                pipe=pipeline,
                bg_color=background,
                brdf_lut=brdf_lut,
                speed=False,
                inference = True
                )

            H, W = view.image_height, view.image_width
            c2w = torch.inverse(view.world_view_transform.T)  # [4, 4]
            view_dirs = -(( F.normalize(canonical_rays[:, None, :], p=2, dim=-1)* c2w[None, :3, :3]).sum(dim=-1) #[HW,3]
                        .reshape(H, W, 3)) # direct from screen to cam center
            render_pkg_df = pbr_render_df(
                viewpoint_camera=view,
                pc=gaussians,
                light=cubemap,
                pipe=pipeline,
                bg_color=background,
                view_dirs = view_dirs,
                brdf_lut= brdf_lut,
                speed=False,
                inference = True
                )
            render_pkg = {}
            for keys in render_pkg_df.keys():
                render_pkg[keys] = (render_pkg_fw[keys] + render_pkg_df[keys])/2.
        
        elif mode == "stochastic":
            assert pipeline.fw_rate > 0 and pipeline.fw_rate < 1, "fw_rate should be in (0,1)"

            render_pkg_fw = pbr_render_fw(
                        viewpoint_camera=view,
                        pc=gaussians,
                        light=cubemap,
                        pipe=pipeline,
                        bg_color=background,
                        brdf_lut=brdf_lut,
                        speed=False,
                        inference = True
                        )
            
            H, W = view.image_height, view.image_width
            c2w = torch.inverse(view.world_view_transform.T)  # [4, 4]
            view_dirs = -(( F.normalize(canonical_rays[:, None, :], p=2, dim=-1)* c2w[None, :3, :3]).sum(dim=-1) #[HW,3]
                        .reshape(H, W, 3)) # direct from screen to cam center

            render_pkg_df = pbr_render_df(
                viewpoint_camera=view,
                pc=gaussians,
                light=cubemap,
                pipe=pipeline,
                bg_color=background,
                view_dirs = view_dirs,
                brdf_lut= brdf_lut,
                speed=False,
                inference = True
                )
            
            fw_mask = torch.rand(H, W, device="cuda") < pipeline.fw_rate # [H,W]
                
            render_pkg = render_pkg_fw
            for key in ["render", "albedo", "roughness", "metallic", 
                        "diffuse_rgb", "specular_rgb", 
                        "diffuse_light", "specular_light"]:
                if key in render_pkg_fw.keys() and key in render_pkg_df.keys():
                    if render_pkg_fw[key] is not None and render_pkg_df[key] is not None:
                        render_pkg[key] = fw_mask[None,:, :] * render_pkg_fw[key] + (~fw_mask[None,:, :]) * render_pkg_df[key]

        else:
            raise ValueError("Unknown render mode")
        
        torch.cuda.synchronize()

        alpha, rend_normal, surf_depth, normal_from_d = render_pkg["rend_alpha"], render_pkg["rend_normal"], render_pkg["surf_depth"], render_pkg["surf_normal"]
        diffuse_rgb, specular_rgb, albedo, roughness, metallic = render_pkg["diffuse_rgb"], render_pkg["specular_rgb"], render_pkg["albedo"], render_pkg["roughness"], render_pkg["metallic"]
        diffuse_light, specular_light = render_pkg["diffuse_light"], render_pkg["specular_light"]

        image = render_pkg["render"]
        torchvision.utils.save_image(image, os.path.join(render_path,'{0:05d}'.format(idx) + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        roughness = apply_depth_colormap(roughness[0][...,None], cmap="gray").permute(2,0,1)
        metallic = apply_depth_colormap(metallic[0][...,None], cmap="gray").permute(2,0,1)
        alpha = apply_depth_colormap(alpha[0][...,None], cmap="gray").permute(2,0,1)

        brdf_map = torch.cat([albedo, roughness, metallic,], dim=2,)
        torchvision.utils.save_image(brdf_map, os.path.join(brdf_path, f"{idx:05d}.png"))
        pbr_image = torch.cat([image, diffuse_rgb, specular_rgb], dim=2)  # [3, H, 3W]
        torchvision.utils.save_image(pbr_image, os.path.join(pbr_path, f"{idx:05d}.png"))

        torchvision.utils.save_image(alpha, os.path.join(alpha_path, f"{idx:05d}.png"))
        
        # world view normals
        normalR = 0.5 + (0.5*rend_normal)
        normalD = 0.5 + (0.5*normal_from_d)
        torchvision.utils.save_image(normalR, os.path.join(normalRender_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(normalD, os.path.join(normalDepth_path, f"{idx:05d}.png"))

        depth = apply_depth_colormap(-surf_depth[0][...,None], cmap="turbo").permute(2,0,1)        
        torchvision.utils.save_image(depth, os.path.join(depth_path, f"{idx:05d}.png"))

        lights = torch.cat([diffuse_light, specular_light], dim=2)
        torchvision.utils.save_image(lights, os.path.join(light_path, f"{idx:05d}.png"))

        if mode == "stochastic":
            torchvision.utils.save_image(render_pkg_fw["render"], os.path.join(render_path,'{0:05d}'.format(idx) + "_fw.png"))
            torchvision.utils.save_image(render_pkg_df["render"], os.path.join(render_path,'{0:05d}'.format(idx) + "_df.png"))

            roughness_fw = apply_depth_colormap(render_pkg_fw["roughness"][0][...,None], cmap="gray").permute(2,0,1)
            roughness_df = apply_depth_colormap(render_pkg_df["roughness"][0][...,None], cmap="gray").permute(2,0,1)
            metallic_fw = apply_depth_colormap(render_pkg_fw["metallic"][0][...,None], cmap="gray").permute(2,0,1)
            metallic_df = apply_depth_colormap(render_pkg_df["metallic"][0][...,None], cmap="gray").permute(2,0,1)

            brdf_map_fw = torch.cat([render_pkg_fw["albedo"], roughness_fw, metallic_fw,], dim=2,)
            torchvision.utils.save_image(brdf_map_fw, os.path.join(brdf_path, f"{idx:05d}_fw.png"))
            brdf_map_df = torch.cat([render_pkg_df["albedo"], roughness_df, metallic_df,], dim=2,)
            torchvision.utils.save_image(brdf_map_df, os.path.join(brdf_path, f"{idx:05d}_df.png"))

            pbr_image_fw = torch.cat([render_pkg_fw["render"], render_pkg_fw["diffuse_rgb"], render_pkg_fw["specular_rgb"]], dim=2)  # [3, H, 3W]
            torchvision.utils.save_image(pbr_image_fw, os.path.join(pbr_path, f"{idx:05d}_fw.png"))
            pbr_image_df = torch.cat([render_pkg_df["render"], render_pkg_df["diffuse_rgb"], render_pkg_df["specular_rgb"]], dim=2)
            torchvision.utils.save_image(pbr_image_df, os.path.join(pbr_path, f"{idx:05d}_df.png"))

        





def render_sets(dataset : ModelParams, chkp_path: str, pipeline : PipelineParams, skip_train : bool, skip_test : bool, mode : str):
    with torch.no_grad():
    

        (model_params, light_params, _, first_iter) = torch.load(chkp_path)

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False, load_gt_normals=False)
        gaussians.restore(model_params)
        
        cubemap = CubemapLight(base_res=256).cuda()
        cubemap.load_state_dict(light_params)
        cubemap.eval()

        print("Restored from checkpoint at iteration", first_iter)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        # bg_color = [1,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        canonical_rays = scene.get_canonical_rays()


        if not skip_train:
             render_set(dataset.model_path, "train", first_iter, scene.getTrainCameras(), 
                        gaussians, cubemap,  pipeline, background, canonical_rays, 
                        mode)

        if not skip_test:
             render_set(dataset.model_path, "test", first_iter, scene.getTestCameras(), 
                        gaussians, cubemap, pipeline, background, canonical_rays, 
                        mode)


             
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    # parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None, help="The path to the checkpoint to load.")

    parser.add_argument("--mode", type=str, default="iterative")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args),args.checkpoint, pipeline.extract(args), args.skip_train, args.skip_test, args.mode)


# python pbr_render.py -s /is/cluster/fast/pyu/data/refnerf/helmet -m /is/cluster/fast/pyu/results/helmet/iter_20_1 -w --eval --checkpoint /is/cluster/fast/pyu/results/helmet/iter_20_1/chkpnt45000.pth --mode iterative