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
from tqdm import tqdm
import ges_rasterization_init2D
import ges_rasterization_surfel
import ges_rasterization_surfel_reg
import ges_rasterization_joint_s
import ges_rasterization_joint_g

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
from torch.nn import AvgPool2d, MaxPool2d

def add_gs_points(gen_num: int, cameras, pc: GaussianModel, pipe, bg_color : torch.Tensor):
    gen_num_each_view = int(gen_num / len(cameras))
    all_pts = []
    with torch.no_grad():
        for cam in cameras:
            rets = render(cam, pc, pipe, bg_color, ret_points=False)
            color = rets["render"]
            points = cam.points_map.cuda() 
            gt_image = cam.original_image.cuda()
            error = ((color - gt_image)**2).sum(0).flatten()
            pdf = error / error.sum()
            #idxs = torch.multinomial(pdf, gen_num_each_view)
            cdf = torch.cumsum(pdf, dim = 0)
            samples = torch.rand(gen_num_each_view, device='cuda', dtype=torch.float)
            idxs = torch.searchsorted(cdf, samples)
            new_pts = points[idxs]
            all_pts.append(new_pts)
    all_pts = torch.cat(all_pts, dim = 0)
    invalid_msk = torch.all(all_pts == 0, dim=-1)
    all_pts = all_pts[~invalid_msk]
    pc.add_gs_points(all_pts)

####################################################


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, ret_points = False):
    stage = pc.render_stage
    if stage == 'init':
        render_func = render_init
    elif stage == 'surfel':
        render_func = render_surfel_reg if pc.use_surf_reg else render_surfel
    elif stage == 'joint':
        render_func = render_joint
    return render_func(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, ret_points)


def render_init(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, ret_points = False):
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

    GaussianRasterizationSettings = ges_rasterization_init2D.GaussianRasterizationSettings
    GaussianRasterizer = ges_rasterization_init2D.GaussianRasterizer

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
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features

    rasargs = {
        "means3D" : means3D,
        "means2D" : means2D,
        "shs" : shs,
        "colors_precomp" : None,
        "opacities" : opacity,
        "scales" : scales,
        "rotations" : rotations,
        "cov3D_precomp" : None
    }

    rets = rasterizer(**rasargs)
    rendered_image, radii = rets[0], rets[1]

    return {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }


def render_surfel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, ret_points = False):
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

    msaa = int(pc.msaa)

    GaussianRasterizationSettings = ges_rasterization_surfel.GaussianRasterizationSettings
    GaussianRasterizer = ges_rasterization_surfel.GaussianRasterizer

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height*msaa),
        image_width=int(viewpoint_camera.image_width*msaa),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        use_filter=False, ##
        disable_sh_means_grad_opac_thr = pc.means_sh_grad_thr,
        need_clamp_color = True,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.surfel_opac * torch.ones_like(pc.get_opacity)
    scales = pc.get_scaling[:,:2]
    rotations = pc.get_rotation
    shs = pc.get_features

    rasargs = {
        "means3D" : means3D,
        "means2D" : means2D,
        "shs" : shs,
        "colors_precomp" : None,
        "opacities" : opacity,
        "scales" : scales,
        "rotations" : rotations,
        "cov3D_precomp" : None
    }

    render_msk = pc.surfel_mask
    for key in rasargs.keys():
        val = rasargs[key]
        if val is None: continue
        rasargs[key] = val[render_msk]

    rendered_image, radii_, cover_area_ = rasterizer(**rasargs)

    radii = torch.zeros(means3D.shape[0], device='cuda', dtype=torch.int32)
    radii[render_msk] = radii_
    cover_area = torch.zeros(means3D.shape[0], device='cuda', dtype=torch.int32)
    cover_area[render_msk] = cover_area_

    if msaa != 1:
        avgp = AvgPool2d(msaa, stride=msaa)
        rendered_image = avgp(rendered_image)
        radii = torch.ceil(radii.float() / msaa)
        cover_area = (cover_area / msaa**2).int()

    x_pad = torch.nn.functional.pad(rendered_image, (1, 1, 1, 1), mode='replicate')
    rendered_image = torch.nn.functional.avg_pool2d(x_pad, kernel_size=3, stride=1)

    return  {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "cover_area": cover_area,
        "radii": radii,
    }

def render_joint(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, ret_points = False):
    if viewpoint_camera.s_idxs is None:
        s_rets = render_joint_s(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, ret_points)
        s_color = s_rets["s_color"]
        s_mod_depth = s_rets["s_mod_depth"]
        s_radii = s_rets["s_radii"]
    else:
        s_color = render_joint_s_with_pre_idxs(viewpoint_camera, viewpoint_camera.s_idxs.cuda(), pc, bg_color)
        s_mod_depth = viewpoint_camera.s_mod_depth.cuda()
        s_radii = viewpoint_camera.s_radii.cuda()

    g_rets = render_joint_g(viewpoint_camera, pc, pipe, bg_color, s_mod_depth, scaling_modifier, override_color)
    weight_color = g_rets['weight_color']
    weightsum = g_rets['weightsum']
    g_radii = g_rets['g_radii']
    g_max_contrib = g_rets['max_contrib']
    means2D = g_rets['viewspace_points']

    rets = {}
    
    s_weight = pc.surfel_weight * torch.exp(-torch.clamp_max(s_mod_depth, 5) / 3) ####
    final_color = (s_color*s_weight + weight_color) / (s_weight + weightsum)

    final_radii = torch.zeros(pc.get_xyz.shape[0], device='cuda', dtype=torch.int)
    final_radii[pc.surfel_mask] = s_radii
    final_radii[~pc.surfel_mask] = g_radii

    if ret_points:
        # just use for add_points
        maxp_depth = s_rets['s_maxp_depth']
        invalid_pts = (maxp_depth == 0).flatten()
        _, points_w = depth_to_normal(viewpoint_camera, maxp_depth, True)
        points_w = points_w.reshape(-1, 3)
        if invalid_pts.sum() > 0:
            points_w[invalid_pts] = torch.tensor([0,0,0], dtype=torch.float, device='cuda').reshape(1,3) # we use 000 to mark outside pts
        rets.update({
            'surf_points': points_w,
            's_idxs': s_rets['s_idxs'],
            's_radii': s_radii,
            's_mod_depth': s_mod_depth
        })

    rets.update({
        "render": final_color,
        'viewspace_points': means2D,
        "visibility_filter" : final_radii > 0,
        "radii": final_radii,
        'g_max_contrib': g_max_contrib,
    })
    return rets

def render_joint_s_with_pre_idxs(viewpoint_camera, s_idxs: torch.Tensor, pc : GaussianModel, bg_color : torch.Tensor):
    msaa = int(pc.msaa)
    msH, msW = s_idxs.shape[0], s_idxs.shape[1]
    surfel_mask = pc.surfel_mask

    shs = pc.get_features[surfel_mask]
    xyz = pc.get_xyz[surfel_mask]

    colors_precomp = ges_rasterization_joint_s.ComputeSH(
        xyz, shs, viewpoint_camera.camera_center, pc.active_sh_degree)
    
    colors_precomp = torch.cat([bg_color[None], colors_precomp], dim=0)
    s_idxs = (s_idxs + 1).flatten() # -1 means bg_color, now 0->bg_color
    colors = colors_precomp[s_idxs].reshape(msH, msW, 3).permute(2,0,1)

    if msaa != 1:
        avgp = AvgPool2d(msaa, stride=msaa)
        colors = avgp(colors)
    return colors


def render_joint_s(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, ret_max_pool_dep = False):

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    msaa = int(pc.msaa)

    GaussianRasterizationSettings = ges_rasterization_joint_s.GaussianRasterizationSettings
    GaussianRasterizer = ges_rasterization_joint_s.GaussianRasterizer

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height*msaa),
        image_width=int(viewpoint_camera.image_width*msaa),
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
    opacity = pc.get_opacity
    surfel_mask = pc.surfel_mask ###
    scales = pc.get_scaling[:,:2]
    rotations = pc.get_rotation
    shs = pc.get_features

    mod_depth = pc.get_moddepth

    rasargs = {
        "means3D" : means3D,
        "shs" : shs,
        "mod_depth": mod_depth,
        "colors_precomp" : None,
        "opacities" : opacity,
        "scales" : scales,
        "rotations" : rotations,
        "cov3D_precomp" : None
    }
    for key in rasargs.keys():
        val = rasargs[key]
        if val is None: continue
        rasargs[key] = val[surfel_mask]

    s_color, radii, allmap = rasterizer(**rasargs)

    ret = {}

    s_depth = allmap[0:1]
    s_mod_depth = allmap[1:2]
    s_idxs = allmap[2:3].long()[0] # H,W

    if msaa != 1:
        avgp = AvgPool2d(msaa, stride=msaa)
        s_color = avgp(s_color)
        s_mod_depth = avgp(s_mod_depth)
        radii = (radii / msaa).int()

    if ret_max_pool_dep:
        if msaa != 1:
            raw_depth = s_depth.detach().clone()
            maxp = MaxPool2d(msaa, stride = msaa)
            maxp_depth = maxp(raw_depth)
            ret.update({'s_maxp_depth': maxp_depth})
        else:
            raw_depth = s_depth.detach().clone()
            ret.update({'s_maxp_depth': raw_depth})

    s_c_pad = torch.nn.functional.pad(s_color, (1, 1, 1, 1), mode='replicate')
    #s_d_pad = torch.nn.functional.pad(s_mod_depth, (1, 1, 1, 1), mode='replicate')
    s_color = torch.nn.functional.avg_pool2d(s_c_pad, kernel_size=3, stride=1)
    #s_mod_depth = torch.nn.functional.avg_pool2d(s_d_pad, kernel_size=3, stride=1)
    
    ret.update({
        "s_color": s_color,
        "s_radii": radii,
        's_mod_depth': s_mod_depth,
        "s_idxs": s_idxs # NO MSAA
    })
    return ret

def render_joint_g(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, depth_map: torch.Tensor, scaling_modifier = 1.0, override_color = None):

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    max_contrib_ret = torch.zeros_like(pc.get_opacity, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        max_contrib_ret.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    GaussianRasterizationSettings = ges_rasterization_joint_g.GaussianRasterizationSettings
    GaussianRasterizer = ges_rasterization_joint_g.GaussianRasterizer

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
    surfel_mask = pc.surfel_mask ###
    rotations = pc.get_rotation
    shs = pc.get_features
    scales = pc.get_scaling
    opacity = pc.get_opacity 

    rasargs = {
        "means3D" : means3D,
        "means2D" : screenspace_points,
        "max_contrib_ret": max_contrib_ret,
        "shs" : shs,
        "colors_precomp" : None,
        "opacities" : opacity,
        "scales" : scales,
        "rotations" : rotations,
        "cov3D_precomp" : None
    }
    for key in rasargs.keys():
        val = rasargs[key]
        if val is None: continue
        rasargs[key] = val[~surfel_mask]
    rasargs.update({"depth_map": depth_map})

    weight_color, radii, weightsum = rasterizer(**rasargs)

    return {
        "weight_color": weight_color,
        "viewspace_points": screenspace_points,
        "g_radii": radii,
        'weightsum': weightsum,
        "max_contrib": max_contrib_ret,
    }

def render_surfel_reg(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, ret_points = False):
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

    msaa = int(pc.msaa)

    GaussianRasterizationSettings = ges_rasterization_surfel_reg.GaussianRasterizationSettings
    GaussianRasterizer = ges_rasterization_surfel_reg.GaussianRasterizer

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height*msaa),
        image_width=int(viewpoint_camera.image_width*msaa),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        use_filter=False,
        disable_sh_means_grad_opac_thr = pc.means_sh_grad_thr,
        need_clamp_color = True,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.surfel_opac * torch.ones_like(pc.get_opacity)
    scales = pc.get_scaling[:,:2]
    rotations = pc.get_rotation
    shs = pc.get_features

    rasargs = {
        "means3D" : means3D,
        "means2D" : means2D,
        "shs" : shs,
        "colors_precomp" : None,
        "opacities" : opacity,
        "scales" : scales,
        "rotations" : rotations,
        "cov3D_precomp" : None
    }

    render_msk = pc.surfel_mask
    for key in rasargs.keys():
        val = rasargs[key]
        if val is None: continue
        rasargs[key] = val[render_msk]

    rendered_image, radii_, cover_area_, allmap = rasterizer(**rasargs)

    radii = torch.zeros(means3D.shape[0], device='cuda', dtype=torch.int32)
    radii[render_msk] = radii_
    cover_area = torch.zeros(means3D.shape[0], device='cuda', dtype=torch.int32)
    cover_area[render_msk] = cover_area_

    if msaa != 1:
        avgp = AvgPool2d(msaa, stride=msaa)
        rendered_image = avgp(rendered_image)
        radii = torch.ceil(radii.float() / msaa)
        cover_area = (cover_area / msaa**2).int()

    rets =  {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "cover_area": cover_area,
        "radii": radii,
    }

    # additional regularizations
    render_alpha = allmap[1:2]

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)

    surf_depth = render_depth_expected

    if msaa != 1:
        render_normal = avgp(render_normal)
        surf_depth = avgp(surf_depth)
        render_alpha = avgp(render_alpha)
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    if ret_points:
        surf_normal, points_w = depth_to_normal(viewpoint_camera, surf_depth, ret_points)
        rets.update({'surf_points': points_w})
    else:
        surf_normal = depth_to_normal(viewpoint_camera, surf_depth, ret_points)
    surf_normal = surf_normal.permute(2,0,1)
    surf_normal = surf_normal * (render_alpha).detach()
    
    rets.update({
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
    })

    return rets