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
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, add_gs_points
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
import numpy as np
import time, struct
from utils.loss_utils import ssim
from lpipsPyTorch import get_lpips_model
from scene.cameras import Camera
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    #args.save_iterations.append(opt.iterations)
    saving_iterations.append(opt.joint_start_iter)
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    gaussians.set_render_stage('init')
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    

    if 'synthetic' in dataset.source_path:
        min_cover_area = 4
    else:
        min_cover_area = 16
    opac_30_until_iter = opt.init_until_iter + opt.opac_30_iter
    opac_60_until_iter = opac_30_until_iter + opt.opac_60_iter
    gs_add_start_iter = opt.joint_start_iter + opt.gs_add_start_from_joint
    gs_add_end_iter = opt.joint_start_iter + opt.gs_add_end_from_joint
    gaussians.use_surf_reg = opt.use_surf_reg

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)
            
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        if iteration == opt.init_until_iter:
            gaussians.msaa = 2
            gaussians.set_render_stage('surfel')
            gaussians.surfel_mask = gaussians.get_opacity.flatten() > 0.8 ###############
            gaussians.surfel_opac = 30
        if iteration == opac_30_until_iter:
            gaussians.surfel_opac = 60
        if iteration == opac_60_until_iter:
            gaussians.surfel_opac = 90
        if iteration == opt.joint_start_iter: 
            gaussians.set_render_stage('joint')
            gaussians.change_scale2D_2_3D()
            gaussians.calc_mod_depth()
            gaussians.adjust_learning_rate('opacity', 0.005)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if opt.use_surf_reg and gaussians.render_stage == 'surfel':
            rend_normal = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = 0.05 * (normal_error).mean()
            loss = loss + normal_loss


        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), opt.iterations, scene, render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats2(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % 100 == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune2(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration < opt.reset_opac_until_iter and (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()


            if iteration == opt.joint_start_iter + 1:
                print('Render points')
                for cam in scene.getTrainCameras():
                    rets = render(cam, gaussians, pipe, background, ret_points=True)
                    cam.points_map = rets["surf_points"].cpu()
                    #cam.s_idxs = rets["s_idxs"]
                    #cam.s_radii = rets["s_radii"]
                    #cam.s_mod_depth = rets["s_mod_depth"]
                print('Render completed')
                gs_add_num_each = int((~gaussians.surfel_mask).sum().item() /5 /10)

            if opac_30_until_iter < iteration < opt.joint_start_iter:
                # Here we use max_radii record cover_area
                cover_area = render_pkg['cover_area']
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], cover_area[visibility_filter])
                if iteration % 1000 == 0:
                    #gaussians.prune_surfel(scene.cameras_extent, min_cover_area, 0.44)
                    gaussians.prune_surfel(scene.cameras_extent, min_cover_area, None)

            if opt.joint_start_iter < iteration < opt.joint_start_iter + 2_000:
                max_contrib = render_pkg['g_max_contrib']
                gaussians.add_gsprune_stats(max_contrib)
                if iteration % 500 == 0:
                    gaussians.prune_gs(min_contrib=0.02)

            if gs_add_start_iter <= iteration < gs_add_end_iter: #23_000 <= iteration < 33_000:
                if iteration % 1000 == 0:
                    print('Add points')
                    add_gs_points(gs_add_num_each, scene.getTrainCameras(), gaussians, pipe, background)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):    

    args.model_path = os.path.join("./output/", os.path.basename(args.source_path))
    if os.path.exists(args.model_path):
        sim_idx = 1
        while True:
            tpath = args.model_path+f'_{sim_idx}'
            if not os.path.exists(tpath):
                break
            sim_idx += 1
        args.model_path = tpath
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

LPIPM = None
@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, last_iter, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_scalar('points_num/total', scene.gaussians.get_xyz.shape[0], iteration)
        surfel_mask = scene.gaussians.surfel_mask
        if surfel_mask is not None:
            tb_writer.add_scalar('points_num/Gaussians', (~surfel_mask).sum().item(), iteration)
            tb_writer.add_scalar('points_num/Surfels', surfel_mask.sum().item(), iteration)

    global LPIPM
    if LPIPM is None:
        LPIPM = get_lpips_model(net_type='vgg').cuda()

    ReportIntv = 6000
    
    # Report test and samples of training set
    if iteration % ReportIntv == 0 or iteration == last_iter:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    error_map = ((image - gt_image)**2).sum(dim=0, keepdim=True)
                    error_map = torch.pow(error_map, 0.5)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/#2_render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/#3_error".format(viewpoint.image_name), error_map[None], global_step=iteration)
                        if "surfel_render" in render_pkg.keys():
                            s_color = torch.clamp(render_pkg["surfel_render"], 0.0, 1.0)
                            tb_writer.add_images(config['name'] + "_view_{}/#5_surfel_color".format(viewpoint.image_name), s_color[None], global_step=iteration)
                        if "rend_normal" in render_pkg.keys():
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/#4_rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                        if iteration == ReportIntv:
                            tb_writer.add_images(config['name'] + "_view_{}/#1_ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        
                        #tb_writer.add_scalar(config['name'] + "_view_{}/PSNR".format(viewpoint.image_name), psnr(image, gt_image).mean().item(), iteration)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image[None], gt_image[None]).item()
                    lpips_test += LPIPM(image[None], gt_image[None]).item() #######################

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: PSNR {}, SSIM {}, LPIPS {}".format(iteration, config['name'], psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            opac = scene.gaussians.get_opacity.flatten()
            surfel_mask = scene.gaussians.surfel_mask
            if surfel_mask is not None:
                surfel_opac = opac[surfel_mask]
                gauss_opac = opac[~surfel_mask]
                if gauss_opac.shape[0] > 0:
                    tb_writer.add_histogram("scene/gauss_opac", gauss_opac, iteration)
                if surfel_opac.shape[0] > 0:
                    tb_writer.add_histogram("scene/surfel_opac", surfel_opac, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--open_server', action='store_true', default=False)

    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
