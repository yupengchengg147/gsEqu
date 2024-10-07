import os
import sys
import torch
import torch.nn.functional as F

import numpy as np

from random import randint
from tqdm import tqdm
from utils.general_utils import safe_state, get_expon_lr_func
from utils.loss_utils import l1_loss, ssim, zero_one_loss, delta_normal_loss

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from scene import Scene, GaussianModel
from pbr import CubemapLight, get_brdf_lut


from gaussian_renderer import pbr_render_fw, network_gui, render, pbr_render_df, pbr_render_st

from train import prepare_output_and_logger, training_report


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



def pbr_training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, mode, gt_mask):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, load_gt_normals=gt_mask)
    canonical_rays = scene.get_canonical_rays()

    gaussians.training_setup(opt)

    cubemap = CubemapLight(base_res=256).cuda()
    param_groups = [
        {"name": "cubemap", "params": cubemap.parameters(), "lr": opt.brdf_mlp_lr_init},
    ]
    light_optimizer = torch.optim.Adam(param_groups, lr=opt.opacity_lr)
    brdf_mlp_scheduler_args = get_expon_lr_func(lr_init=opt.brdf_mlp_lr_init,
                                        lr_final=opt.brdf_mlp_lr_final,
                                        lr_delay_mult=opt.brdf_mlp_lr_delay_mult,
                                        max_steps=opt.brdf_mlp_lr_max_steps)
    brdf_lut = get_brdf_lut().cuda()

    if checkpoint:
        (model_params, light_params, lightopt_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        cubemap.load_state_dict(light_params)
        light_optimizer.load_state_dict(lightopt_params)
        print("Restored from checkpoint at iteration", first_iter)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # bg_color = [1, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_alpha_for_log = 0.0
    ema_delta_for_log = 0.0
    ema_normal_for_log = 0.0
    
    # in warmup iterations, set envmap gradient false, set pbr related para gradient false
    cubemap.base.requires_grad = False
    gaussians.set_requires_grad("albedo", False)
    gaussians.set_requires_grad("roughness", False)
    gaussians.set_requires_grad("metallic", False)

    gaussians.set_requires_grad("normal_0", False)
    gaussians.set_requires_grad("normal_1", False)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        #dynamic lr
        gaussians.update_learning_rate(iteration)
        
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Every 1000 its we increase the levels of SH up to a maximum degree
    
        if iteration < (opt.warmup_iterations+1):
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
            
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        elif iteration == (opt.warmup_iterations+1):
            cubemap.train()
            cubemap.base.requires_grad = True
        
            gaussians.set_requires_grad("features_dc", False)
            gaussians.set_requires_grad("features_rest", False)

            gaussians._albedo.data = gaussians._features_dc.data.clone().squeeze()
            gaussians.set_requires_grad("albedo", True)
            gaussians.set_requires_grad("roughness", True)
            gaussians.set_requires_grad("metallic", True)

            gaussians.set_requires_grad("normal_0", True)
            gaussians.set_requires_grad("normal_1", True)
            continue

        else:
            for param_group in light_optimizer.param_groups:
                if param_group["name"] == "cubemap":
                    lr = brdf_mlp_scheduler_args(iteration - opt.warmup_iterations)
                    param_group['lr'] = lr
            cubemap.build_mips()
            
            render_mode = mode
            if render_mode == "fw":
                
                render_pkg = pbr_render_fw(
                    viewpoint_camera=viewpoint_cam,
                    pc=gaussians,
                    light=cubemap,
                    pipe=pipe,
                    bg_color=background,
                    brdf_lut=brdf_lut,
                    speed=True,
                    )
                
            elif render_mode == "df":

                H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
                c2w = torch.inverse(viewpoint_cam.world_view_transform.T)  # [4, 4]
                view_dirs = -(( F.normalize(canonical_rays[:, None, :], p=2, dim=-1)* c2w[None, :3, :3]).sum(dim=-1) #[HW,3]
                            .reshape(H, W, 3)) # direct from screen to cam center
                render_pkg = pbr_render_df(
                        viewpoint_camera=viewpoint_cam,
                        pc=gaussians,
                        light=cubemap,
                        pipe=pipe,
                        bg_color=background,
                        view_dirs = view_dirs,
                        brdf_lut= brdf_lut,
                        speed=True,
                        )
                
            elif render_mode == "iterative":
                assert opt.fw_iter > 0 and opt.df_iter > 0, "iterative interval should be larger than 0"
                
                if iteration % (opt.fw_iter + opt.df_iter) < opt.fw_iter:
                    # fw
                    # print("currently fw shading")
                    render_pkg = pbr_render_fw(
                        viewpoint_camera=viewpoint_cam,
                        pc=gaussians,
                        light=cubemap,
                        pipe=pipe,
                        bg_color=background,
                        brdf_lut=brdf_lut,
                        speed=True,
                        )
                else:
                    # print("currently df shading")
                    H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
                    c2w = torch.inverse(viewpoint_cam.world_view_transform.T)  # [4, 4]
                    view_dirs = -(( F.normalize(canonical_rays[:, None, :], p=2, dim=-1)* c2w[None, :3, :3]).sum(dim=-1) #[HW,3]
                                .reshape(H, W, 3)) # direct from screen to cam center

                    render_pkg = pbr_render_df(
                        viewpoint_camera=viewpoint_cam,
                        pc=gaussians,
                        light=cubemap,
                        pipe=pipe,
                        bg_color=background,
                        view_dirs = view_dirs,
                        brdf_lut= brdf_lut,
                        speed=True,
                        )

            elif render_mode == "stochastic":
                assert pipe.fw_rate > 0 and pipe.fw_rate < 1, "fw_rate should be in (0,1)"

                render_pkg_fw = pbr_render_fw(
                            viewpoint_camera=viewpoint_cam,
                            pc=gaussians,
                            light=cubemap,
                            pipe=pipe,
                            bg_color=background,
                            brdf_lut=brdf_lut,
                            speed=True,
                            )
                
                H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
                c2w = torch.inverse(viewpoint_cam.world_view_transform.T)  # [4, 4]
                view_dirs = -(( F.normalize(canonical_rays[:, None, :], p=2, dim=-1)* c2w[None, :3, :3]).sum(dim=-1) #[HW,3]
                            .reshape(H, W, 3)) # direct from screen to cam center

                render_pkg_df = pbr_render_df(
                    viewpoint_camera=viewpoint_cam,
                    pc=gaussians,
                    light=cubemap,
                    pipe=pipe,
                    bg_color=background,
                    view_dirs = view_dirs,
                    brdf_lut= brdf_lut,
                    speed=True,
                    )
                
                fw_mask = torch.rand(H, W, device="cuda") < pipe.fw_rate # [H,W]
                
                render_pkg = render_pkg_fw
                for key in ["render", "albedo", "roughness", "metallic", 
                            "diffuse_rgb", "specular_rgb", 
                            "diffuse_light", "specular_light"]:
                    if key in render_pkg_fw.keys() and key in render_pkg_df.keys():
                        if render_pkg_fw[key] is not None and render_pkg_df[key] is not None:
                            render_pkg[key] = fw_mask[None,:, :] * render_pkg_fw[key] + (~fw_mask[None,:, :]) * render_pkg_df[key]

            else:
                raise ValueError("Unknown render mode")


            image = render_pkg["render"]
            viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            alpha, rend_normal, surf_depth, normal_from_d = render_pkg["rend_alpha"], render_pkg["rend_normal"], render_pkg["surf_depth"], render_pkg["surf_normal"]

        # diffuse_rgb, specular_rgb, albedo, roughness, metallic = render_pkg["diffuse_rgb"], render_pkg["specular_rgb"], render_pkg["albedo"], render_pkg["roughness"], render_pkg["metallic"]
        # diffuse_light, specular_light = render_pkg["diffuse_light"], render_pkg["specular_light"]

        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if iteration > (opt.warmup_iterations+1):
            # regularization
            lambda_normal = opt.lambda_normal if iteration > opt.normal_ref_start_iter and iteration < opt.normal_reg_until_iter else 0.0
            normal_error = (1 - (rend_normal * normal_from_d).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()

            lambda_alpha = opt.lambda_alpha if iteration > opt.normal_ref_start_iter else 0.0 #1e-3
            alpha_loss = zero_one_loss(alpha) * lambda_alpha

            lambda_delta_n = opt.lambda_delta_n if iteration > opt.normal_ref_start_iter else 0.0 #1e-3
            delta_n_loss = delta_normal_loss(render_pkg["delta_n"], alpha) * lambda_delta_n

            total_loss = loss + alpha_loss + normal_loss + delta_n_loss
        
        else:
            total_loss = loss
            delta_n_loss = torch.tensor(0.0, device="cuda")
            normal_loss = torch.tensor(0.0, device="cuda") 
            alpha_loss = torch.tensor(0.0, device="cuda")

        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_delta_for_log = 0.4 * delta_n_loss.item() + 0.6 * ema_delta_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_alpha_for_log = 0.4 * alpha_loss.item() + 0.6 * ema_alpha_for_log
            # ema_dist_for_log = 0.0
            # norm_grad_fw = 0.0
            # norm_grad_df = 0.0
            # b = gaussians._opacity.grad.data.clone()
            # opacity_grad = torch.abs(b).cpu().mean()
            # c = gaussians._xyz.grad.data.clone() #[n,]
            # z_grad = torch.abs(c[:,2]).cpu().mean()
            # x_grad = torch.abs(c[:,0]).cpu().mean()
            # y_grad = torch.abs(c[:,1]).cpu().mean()
            # grad_dict = {"opacity_grad": opacity_grad, 
            #              "x_grad": x_grad,
            #              "y_grad": y_grad,
            #              "z_grad": z_grad}
            # if iteration > opt.warmup_iterations and iteration % (2*opt.iter_interval) < opt.iter_interval:
            #     a = render_pkg['ng_w'].grad.data.clone() #[n,3]
            #     norm_grad_fw = torch.norm(a, dim=-1).cpu().mean()
            #     grad_dict["norm_grad_fw"] = norm_grad_fw
            
            # if iteration > opt.warmup_iterations and iteration % (2*opt.iter_interval) >= opt.iter_interval:
            #     a1 = rend_normal.grad.data.clone() #[3,h,w]
            #     norm_grad_df = torch.norm(a1, dim=0).cpu().mean()
            #     grad_dict["norm_grad_df"] = norm_grad_df
            grad_dict = None

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "delta_n": f"{ema_delta_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "Alpha": f"{ema_alpha_for_log:.{5}f}",
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/delta_n_loss', ema_delta_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/alpha_loss', ema_alpha_for_log, iteration)
            
            if iteration < (opt.warmup_iterations+1):
                training_report(tb_writer, grad_dict, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            elif render_mode == "fw" or (render_mode == "iterative" and iteration % (opt.fw_iter + opt.df_iter) < opt.fw_iter):
                training_report(tb_writer, grad_dict, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                                testing_iterations, scene, pbr_render_fw, 
                                (cubemap, pipe, background, brdf_lut, False)
                                )
            elif render_mode == "df" or (render_mode == "iterative" and iteration % (opt.fw_iter + opt.df_iter) >= opt.fw_iter):
                training_report(tb_writer, grad_dict, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                                testing_iterations, scene, pbr_render_df, 
                                (cubemap, pipe, background, view_dirs, brdf_lut, False)
                                )
            elif render_mode == "stochastic":
                training_report(tb_writer, grad_dict, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                                testing_iterations, scene, pbr_render_st, 
                                (cubemap, pipe, background, view_dirs, brdf_lut, False, pipe.fw_rate)
                                )
            else:
                raise ValueError("Unknown render mode")
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if iteration > opt.warmup_iterations: 
                    light_optimizer.step()
                    light_optimizer.zero_grad(set_to_none=True)
                    cubemap.clamp_(min=0.0,max=1.0)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), cubemap.state_dict(),light_optimizer.state_dict(),iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15000, 30_000, 45000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15000, 30_000, 45000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15000, 30_000, 45000])

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--mode", type=str, default="iterative")
    # parser.add_argument("--fw_iter", type=int, default=1)
    # parser.add_argument("--df_iter", type=int, default=1)

    parser.add_argument('--gt_mask', action='store_true', default=False)


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    pbr_training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, 
                 args.mode, args.gt_mask)

    # All done
    print("\nTraining complete.")




# python pbr_train.py -s /is/cluster/fast/pyu/data/refnerf/helmet -m /is/cluster/fast/pyu/results/helmet/iter_20_1 -w --eval --warmup_iterations 1 --lambda_dist 100 --lambda_normal 0.01 --fw_iter 1 --df_iter 1 --mode iterative --gamma --tone

# python pbr_train.py -s /is/cluster/fast/pyu/data/refnerf/helmet -m /is/cluster/fast/pyu/results/test -w --eval --warmup_iterations 1 --lambda_normal 0.01  --mode stochastic --fw_rate 0.2 --gamma --tone
# --fw_iter 1 --df_iter 1

# python pbr_train.py \
#             -s "/is/cluster/fast/pyu/data/refnerf/${model}/" \
#             -m "${output_dir}" \
#             -w \
#             --eval \
#             --warmup_iterations 1 \
#             --lambda_normal 0.01 \
#             --fw_rate "${fw_rate}" \
#             --mode "stochastic" \
#             --gamma \
#             --tone
# /is/cluster/fast/pyu/data/tensorir/armadillo