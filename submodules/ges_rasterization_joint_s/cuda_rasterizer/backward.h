/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, const dim3 block,
		int W, int H,
		const int* n_contrib,
		const float* dL_dpixels,
		float* dL_dcolors);

	void preprocess(
		int P, int D, int M,
		const float3* means3D,
		const int* radii,
		const float* shs,
		const glm::vec3* campos, 
		float* dL_dcolors,
		float* dL_dshs);

	void computeSH(
		int P, int D, int M,
		const float* means3D,
		const float* shs,
		const float* campos, 
		const float* dL_dout_colors,
		float* dL_dshs);
}

#endif
