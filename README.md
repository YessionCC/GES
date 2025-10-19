# When Gaussian Meets Surfel: Ultra-fast High-fidelity Radiance Field Rendering
Keyang Ye, Tianjia Shao, Kun Zhou<br>
| [Webpage](https://dl.acm.org/doi/10.1145/3730925) | [Full Paper](https://arxiv.org/pdf/2504.17545) 

This repository contains the code implementation associated with the paper "When Gaussian Meets Surfel: Ultra-fast High-fidelity Radiance Field Rendering".

Abstract: *We introduce Gaussian-enhanced Surfels (GESs), a bi-scale representation for radiance field rendering, wherein a set of 2D opaque surfels with view-dependent colors represent the coarse-scale geometry and appearance of scenes, and a few 3D Gaussians surrounding the surfels supplement fine-scale appearance details. The rendering with GESs consists of two passes -- surfels are first rasterized through a standard graphics pipeline to produce depth and color maps, and then Gaussians are splatted with depth testing and color accumulation on each pixel order independently. The optimization of GESs from multi-view images is performed through an elaborate coarse-to-fine procedure, faithfully capturing rich scene appearance. The entirely sorting-free rendering of GESs not only achieves very fast rates, but also produces view-consistent images, successfully avoiding popping artifacts under view changes. The basic GES representation can be easily extended to achieve anti-aliasing in rendering (Mip-GES), boosted rendering speeds (Speedy-GES) and compact storage (Compact-GES), and reconstruct better scene geometries by replacing 3D Gaussians with 2D Gaussians (2D-GES). Experimental results show that GESs advance the state-of-the-arts as a compelling representation for ultra-fast high-fidelity radiance field rendering.*

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{ges25ye,
author = {Ye, Keyang and Shao, Tianjia and Zhou, Kun},
title = {When Gaussian Meets Surfel: Ultra-fast High-fidelity Radiance Field Rendering},
year = {2025},
issue_date = {August 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {44},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3730925},
doi = {10.1145/3730925},
month = jul,
articleno = {113},
numpages = {15},
}</code></pre>
  </div>
</section>


## Installation
First, please refer to the setup tutorial of vanilla [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). Then, install the following submodules.

```
pip install submodules/ges_rasterization_init2
pip install submodules/ges_rasterization_surfel
pip install submodules/ges_rasterization_surfel_reg
pip install submodules/ges_rasterization_joint_s
pip install submodules/ges_rasterization_joint_g
```
We may integrate these submodules in the future and simplify the installation process.

## Datasets
We mainly test our method on *MipNeRF360*, *DeepBlending*, *Tank and Temples* and *NeRF Synthetic* datasets, and use the same test settings as vanilla 3DGS.

You can create links as follows:
```
mkdir data
ln -s PATH_TO_DATASET data
```

## Running
We provide the script to test our code on each scene of datasets. Just run:
```
sh train_all.sh
```
You may need to modify the path in `train_all.sh`

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --use_surf_reg
  Apply depth-normal consistency loss during the surfel optimization to avoid hole or spiking artifacts.

  #### --converge
  Optimize surfels with more iterations to make results more stable. We observed that surfel optimization exhibits sensitivity to randomness. To ensure high-quality results in most scenes, we recommend always adding this command.
  
</details>
<br>

## Viewer
We provide a simple gui viewer based on OpenGL. Please Ensure your system supports OpenGL 4.5 or higher version and has the OpenGL development environment installed. Our viewer uses [GLFW](https://www.glfw.org/download) for window creation. Please install them first.
```
sudo apt-get install build-essential
sudo apt-get install libgl1-mesa-dev
sudo apt-get install libglfw3
```
Then run the following commands to install our viewer:
```
cd ges_viewer; mkdir build; cd build;
cmake ..; make -j
```
The executable file is then generated in `ges_viewer\bin\viewer`. Run `.\viewer` to launch the viewer and you can drag `*.ply` files into the window to freely view the scene. Press and hold the middle mouse button to translate the view; Press and hold the right mouse button to rotate the view; Use the mouse wheel to move forward or backward. You may check out the code for more options and controls.

## Updates
We add a simple blur post-processing shader to the rendered surfel color map. Experiments show that this alleviates aliasing, especially on synthetic datasets. In addition, some training parameters are adjusted to reduce the instability of surfel optimization and improve reproducibility.

## Acknowledgments

Our Pytorch code is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [2DGS](https://github.com/hbb1/2d-gaussian-splatting) and [MipSplatting](https://github.com/autonomousvision/mip-splatting). Our OpenGL viewer implementation incorporates code segments from  [Splatapult](https://github.com/hyperlogic/splatapult). We sincerely appreciate their contributions.