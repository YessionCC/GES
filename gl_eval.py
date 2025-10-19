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
import torch
from scene import Scene
from scene.cameras import Camera
import cv2, sys
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.system_utils import searchForMaxIteration
from gaussian_renderer import render

from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from PIL import Image
from utils.general_utils import PILtoTorch_
import numpy as np

from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import get_lpips_model


def render_network(dataset : ModelParams, iteration : int, pipeline : PipelineParams, save_ims: bool):
    with torch.no_grad():

        scene = Scene(dataset, None, load_iteration=iteration, shuffle=False, nees_load_gs=False)
        test_views = scene.getTestCameras()
        
        sample_num = 4
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        W,H = test_views[0].image_width, test_views[0].image_height # 1920, 1080
        
        outfile_name = os.path.abspath('./temp.txt')
        with open(outfile_name, 'w') as f:
            for view in test_views:
                proj_mat = view.projection_matrix.cpu().numpy().flatten()
                view_mat = view.world_view_transform.cpu().numpy().flatten()
                proj_mat_str = [str(n) for n in proj_mat]
                view_mat_str = [str(n) for n in view_mat]
                f.write(' '.join(list(proj_mat_str))+'\n')
                f.write(' '.join(list(view_mat_str))+'\n')

        iter_path = os.path.abspath(os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(scene.loaded_iter)))
        model_path = os.path.abspath(os.path.join(iter_path, "point_cloud.ply"))
        if save_ims:
            save_path = os.path.abspath(os.path.join(iter_path, "renders"))
            os.makedirs(save_path, exist_ok=True)
        else:
            save_path = "NOT_SAVE"

        exe_path = 'ges_viewer/bin'
        cmd_str = f'cd {exe_path}; ./eval {sample_num} {W} {H} {bg_color[0]} {bg_color[1]} {bg_color[2]} {model_path} {outfile_name} {save_path}'
        print(f'Cmd:\n{cmd_str}')
        os.system(cmd_str)

        if save_ims:
            LPIPS = get_lpips_model(net_type='vgg').cuda()
            ssims = []
            psnrs = []
            lpipss = []
            i = 0
            for impath in sorted(os.listdir(save_path)):
                print(f"Evaluate: {impath}")
                im = Image.open(os.path.join(save_path, impath))
                render_im = PILtoTorch_(im).cuda()[None,:3,...]
                gt_image = test_views[i].original_image.cuda()[None]

                if render_im.shape[-1] != gt_image.shape[-1]:
                    print('CHECK RESOL FAILED, only report FPS')
                    exit()

                ssims.append(ssim(render_im, gt_image).item())
                psnrs.append(psnr(render_im, gt_image).item())
                lpipss.append(LPIPS(render_im, gt_image).item())
                i+=1
            
            ssim_v = np.array(ssims).mean()
            psnr_v = np.array(psnrs).mean()
            lpip_v = np.array(lpipss).mean()

            print('psnr:{},ssim:{},lpips:{}'.format(psnr_v, ssim_v, lpip_v))
            dump_path = os.path.join(iter_path, 'metric.txt')
            with open(dump_path, 'w') as f:
                f.write('psnr:{},ssim:{},lpips:{}'.format(psnr_v, ssim_v, lpip_v))

        
            

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_ims", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_network(model.extract(args), args.iteration, pipeline.extract(args), args.save_ims)

    