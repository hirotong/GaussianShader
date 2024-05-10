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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_lighting
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import apply_depth_colormap, srgb2linear, linear2srgb
from utils.general_utils import get_minimum_axis
from scene.NVDIFFREC.util import save_image_raw
import numpy as np
import torch.nn.functional as F
from scene.mesh import safe_normalize
from utils.grid_utils import mipmap_linear_grid_put_2d


def render_lightings(model_path, name, iteration, gaussians, sample_num):
    lighting_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(lighting_path, exist_ok=True)
    # sampled_indicies = torch.randperm(gaussians.get_xyz.shape[0])[:sample_num]
    sampled_indicies = torch.arange(gaussians.get_xyz.shape[0], dtype=torch.long)[:sample_num]
    for sampled_index in tqdm(sampled_indicies, desc="Rendering lighting progress"):
        lighting = render_lighting(gaussians, sampled_index=sampled_index)
        torchvision.utils.save_image(lighting, os.path.join(lighting_path, "{0:05d}".format(sampled_index) + ".png"))
        save_image_raw(
            os.path.join(lighting_path, "{0:05d}".format(sampled_index) + ".hdr"),
            lighting.permute(1, 2, 0).detach().cpu().numpy(),
        )


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, linear=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()

        render_pkg = render(view, gaussians, pipeline, background, debug=True)

        torch.cuda.synchronize()

        gt = view.original_image[0:3, :, :]
        if linear:
            torchvision.utils.save_image(
                linear2srgb(render_pkg["render"]), os.path.join(render_path, "{0:05d}".format(idx) + ".png")
            )
            torchvision.utils.save_image(linear2srgb(gt), os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))
        else:
            torchvision.utils.save_image(
                render_pkg["render"], os.path.join(render_path, "{0:05d}".format(idx) + ".png")
            )
            torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))
        for k in render_pkg.keys():
            if render_pkg[k].dim() < 3 or k == "render" or k == "delta_normal_norm":
                continue
            save_path = os.path.join(model_path, name, "ours_{}".format(iteration), k)
            makedirs(save_path, exist_ok=True)
            if k == "alpha":
                render_pkg[k] = apply_depth_colormap(render_pkg["alpha"][0][..., None], min=0.0, max=1.0).permute(
                    2, 0, 1
                )
            if k == "depth":
                render_pkg[k] = apply_depth_colormap(-render_pkg["depth"][0][..., None]).permute(2, 0, 1)
            elif k in ["diffuse", "specular_color", "albedo", "diffuse_color"]:
                render_pkg[k] = linear2srgb(render_pkg[k])
            elif "normal" in k:
                render_pkg[k] = 0.5 + (0.5 * render_pkg[k])
            torchvision.utils.save_image(render_pkg[k], os.path.join(save_path, "{0:05d}".format(idx) + ".png"))
        keys = ["render", "diffuse", "diffuse_color", "specular", "specular_color", "roughness", "albedo", "metallic"]
        concat_image = [gt]
        for key in keys:
            if key in render_pkg.keys():
                concat_image.append(render_pkg[key])
        concat_image = torchvision.utils.make_grid(concat_image, nrow=1)
        save_path = os.path.join(model_path, name, "ours_{}".format(iteration), "compare")
        makedirs(save_path, exist_ok=True)
        torchvision.utils.save_image(
            concat_image,
            os.path.join(save_path, "{0:05d}".format(idx) + ".png"),
        )


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_dim, pipeline.brdf_mode, dataset.brdf_envmap_res)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                dataset.linear,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                dataset.linear,
            )

        if pipeline.brdf:
            render_lightings(dataset.model_path, "lighting", scene.loaded_iter, gaussians, sample_num=1)


def extract_mesh(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    mode: str,
    density_thresh: float,
    texture_size: int = 1024,
):
    save_path = os.path.join(dataset.model_path, "mesh", f"iteration_{iteration}")

    makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_dim, pipeline.brdf_mode, dataset.brdf_envmap_res)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        if mode == "geo":
            mesh = scene.gaussians.extract_mesh(save_path, density_thresh)
            mesh.write_ply(os.path.join(save_path, "mesh.ply"))

        elif mode == "geo+tex":
            mesh = scene.gaussians.extract_mesh(save_path, density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv ...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device="cuda", dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device="cuda", dtype=torch.float32)
            
            render_resolution = 800

            import nvdiffrast.torch as dr

            glctx = dr.RasterizeCudaContext()

            # render from the trained cameras
            views = scene.getTrainCameras()
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
                torch.cuda.synchronize()

                render_pkg = render(view, gaussians, pipeline, background, debug=True)

                rgbs = render_pkg["render"]

                # get coordinate in texture image
                w2c = view.world_view_transform.transpose(0, 1)  # The matrix is saved transposed
                proj = view.projection_matrix.transpose(0, 1)  # The matrix is saved transposed

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode="constant", value=1.0), w2c.T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f)  # [1, H, W, 1]
                depth = depth.squeeze(0)  # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, H, W, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)  # [1, H, W, 3]
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ w2c[:3, :3].T
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()

                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w, uvs[..., [1, 0]] * 2 - 1, rgbs, min_resolution=256, return_count=True
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]
            
            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)
            
            mask = mask.view(h, w)
            
            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
            
            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion
            
            inpaint_region = binary_dilation(mask, iterations=5)
            inpaint_region[mask] = 0
            
            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0
            
            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)
            
            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]
            
            mesh.albedo = torch.from_numpy(albedo).to("cuda")
            mesh.write(os.path.join(save_path, "mesh.obj"))
        else:
            scene.gaussians.save_ply(os.path.join(save_path, "mesh.ply"))
            


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", type=str, default="geo")
    parser.add_argument("--density_thresh", type=float, default=0.5)
    parser.add_argument("--texture_size", type=int, default=1024)
    args = get_combined_args(parser)
    print("Extracting mesh from " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    extract_mesh(model.extract(args), args.iteration, pipeline.extract(args), args.mode, args.density_thresh)
