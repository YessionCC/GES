
import torch
from scene import Scene
import os, time
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import get_lpips_model

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, save_ims):
    if save_ims:
        save_dir = os.path.join(model_path, name, "ours_{}".format(iteration))
        os.makedirs(save_dir, exist_ok=True)

    LPIPS = get_lpips_model(net_type='vgg').cuda()
    psnr_test = 0
    ssim_test = 0
    lpips_test = 0

    for view in tqdm(views, desc="Rendering progress"):
        render_pkg = render(view, gaussians, pipeline, background)
        image = render_pkg["render"].clip(0,1)
        gt_image = view.original_image[0:3]

        psnr_test += psnr(image, gt_image).mean().double()
        ssim_test += ssim(image[None], gt_image[None]).item()
        lpips_test += LPIPS(image[None], gt_image[None]).item()

        if save_ims:
            torchvision.utils.save_image(image, os.path.join(save_dir, f'{view.image_name}.png'))
    
    ssim_v = ssim_test / len(views)
    psnr_v = psnr_test / len(views)
    lpip_v = lpips_test / len(views)
    print('psnr:{},ssim:{},lpips:{}'.format(psnr_v, ssim_v, lpip_v))
    dump_path = os.path.join(model_path, 'metric.txt')
    with open(dump_path, 'w') as f:
        f.write('psnr:{},ssim:{},lpips:{}'.format(psnr_v, ssim_v, lpip_v))

@torch.no_grad
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, save_ims : bool):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians.set_render_stage('joint')
    gaussians.msaa = 2
    gaussians.surfel_mask = gaussians.get_opacity.flatten() > 900
    gaussians.mod_depth = gaussians.get_opacity[gaussians.surfel_mask] - 1000

    render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, save_ims)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.save_images)