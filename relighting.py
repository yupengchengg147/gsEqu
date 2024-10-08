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
from typing import Dict, List, Tuple


from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import pbr_render_fw, pbr_render_df, pbr_render_mixxed
import torchvision
from pbr import CubemapLight, get_brdf_lut

from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import apply_depth_colormap, turbo_cmap

import numpy as np
import cv2
import nvdiffrast.torch as dr



def render_set(model_path, name, iteration, views, gaussians, cubemap,  pipeline, background, canonical_rays, mode, light_name):

    brdf_lut = get_brdf_lut().cuda()

    # build mip for environment light
    cubemap.build_mips()
    os.makedirs(os.path.join(model_path, name), exist_ok=True)

    render_path = os.path.join(model_path, light_name, name, "ours_{}".format(iteration), "renders")
    pbr_path = os.path.join(model_path, light_name, name, "ours_{}".format(iteration), "pbr")

    makedirs(render_path, exist_ok=True)
    makedirs(pbr_path, exist_ok=True)


    
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
            for key in ["render", 
                        "diffuse_rgb", "specular_rgb", 
                        ]:
                if key in render_pkg_fw.keys() and key in render_pkg_df.keys():
                    if render_pkg_fw[key] is not None and render_pkg_df[key] is not None:
                        render_pkg[key] = fw_mask[None,:, :] * render_pkg_fw[key] + (~fw_mask[None,:, :]) * render_pkg_df[key]

        elif mode == "mixxed":
                
                # print("Mixxed rendering")
                            
                H, W = view.image_height, view.image_width
                c2w = torch.inverse(view.world_view_transform.T)  # [4, 4]
                view_dirs = -(( F.normalize(canonical_rays[:, None, :], p=2, dim=-1)* c2w[None, :3, :3]).sum(dim=-1) #[HW,3]
                            .reshape(H, W, 3)) # direct from screen to cam center
                
                render_pkg = pbr_render_mixxed(
                    viewpoint_camera=view,
                    pc=gaussians,
                    light=cubemap,
                    pipe=pipeline,
                    bg_color=background,
                    view_dirs = view_dirs,
                    brdf_lut= brdf_lut,
                    speed=False,
                    )
        else:
            raise ValueError("Unknown render mode")
        
        torch.cuda.synchronize()

        diffuse_rgb, specular_rgb = render_pkg["diffuse_rgb"], render_pkg["specular_rgb"]

        image = render_pkg["render"]
        torchvision.utils.save_image(image, os.path.join(render_path,'{0:05d}'.format(idx) + ".png"))
       
        pbr_image = torch.cat([image, diffuse_rgb, specular_rgb], dim=2)  # [3, H, 3W]
        torchvision.utils.save_image(pbr_image, os.path.join(pbr_path, f"{idx:05d}.png"))


        if mode == "stochastic":
            torchvision.utils.save_image(render_pkg_fw["render"], os.path.join(render_path,'{0:05d}'.format(idx) + "_fw.png"))
            torchvision.utils.save_image(render_pkg_df["render"], os.path.join(render_path,'{0:05d}'.format(idx) + "_df.png"))

            pbr_image_fw = torch.cat([render_pkg_fw["render"], render_pkg_fw["diffuse_rgb"], render_pkg_fw["specular_rgb"]], dim=2)  # [3, H, 3W]
            torchvision.utils.save_image(pbr_image_fw, os.path.join(pbr_path, f"{idx:05d}_fw.png"))
            pbr_image_df = torch.cat([render_pkg_df["render"], render_pkg_df["diffuse_rgb"], render_pkg_df["specular_rgb"]], dim=2)
            torchvision.utils.save_image(pbr_image_df, os.path.join(pbr_path, f"{idx:05d}_df.png"))


def read_hdr(path: str) -> np.ndarray:
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    with open(path, "rb") as h:
        buffer_ = np.frombuffer(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map: torch.Tensor, res: List[int]) -> torch.Tensor:
    cubemap = torch.zeros(
        6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device="cuda"
    )
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
            indexing="ij",
        )
        v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(
            latlong_map[None, ...], texcoord[None, ...], filter_mode="linear"
        )[0]
    return cubemap


def render_sets(dataset : ModelParams, chkp_path: str, pipeline : PipelineParams, skip_train : bool, skip_test : bool, mode : str, hdri_path: str):
    
    with torch.no_grad():
        (model_params, light_params, _, first_iter) = torch.load(chkp_path)

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False, load_gt_normals=False)
        gaussians.restore(model_params)
        

        # cubemap.load_state_dict(light_params)
        print(f"read hdri from {hdri_path}")
        hdri = read_hdr(hdri_path)
        hdri = torch.from_numpy(hdri).cuda()
        res = 256
        cubemap = CubemapLight(base_res=res).cuda()
        cubemap.base.data = latlong_to_cubemap(hdri, [res, res])

        cubemap.eval()

        light_name = os.path.basename(hdri_path).split(".")[0]
        print(f"Relighting with light {light_name}")


        print("Restored from checkpoint at iteration", first_iter)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        # bg_color = [1,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        canonical_rays = scene.get_canonical_rays()


        if not skip_train:
             render_set(dataset.model_path, "train", first_iter, scene.getTrainCameras(), 
                        gaussians, cubemap,  pipeline, background, canonical_rays, 
                        mode, light_name)

        if not skip_test:
             render_set(dataset.model_path, "test", first_iter, scene.getTestCameras(), 
                        gaussians, cubemap, pipeline, background, canonical_rays, 
                        mode, light_name)


             
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

    parser.add_argument("--hdri", type=str, default=None, help="The path to the hdri for relighting.")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args),args.checkpoint, pipeline.extract(args), args.skip_train, args.skip_test, args.mode, args.hdri)


# python relighting.py -s /is/cluster/fast/pyu/data/refnerf/helmet -m /is/cluster/fast/pyu/refnerf_results_3dgs/car/st_fwrate_0.5 
# -w --eval --checkpoint /is/cluster/fast/pyu/refnerf_results_3dgs/car/st_fwrate_0.5/chkpnt45000.pth 
# --mode stochastic --fw_rate 0.5 --hdri /is/cluster/fast/pyu/data/high_res_envmaps_1k/night.hdr