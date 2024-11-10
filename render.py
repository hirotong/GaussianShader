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
from argparse import ArgumentParser
from copy import deepcopy
from glob import glob
from os import makedirs
from scene.NVDIFFREC.light import load_env
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render, render_lighting
from scene import Scene
from scene.NVDIFFREC.util import save_image_raw
from utils.general_utils import get_minimum_axis, read_pickle, safe_state
from utils.image_utils import apply_depth_colormap, linear2srgb, srgb2linear

NEROSYNC_ENVNAME2PATH = {
    "corridor": "large_corridor_4k",
    "golf": "limpopo_golf_course_4k",
    "neon": "neon_photostudio_4k",
}


def nerosync_relighting_poses(scene: Scene, env_map: str, scale_factor: float = 1.0):
    source_dir = scene.args.source_path
    case_name = os.path.basename(source_dir).replace("_blender", "")
    if case_name == "tbell":
        case_name = "table_bell"
    elif case_name == "teapot":
        case_name = "utah_teapot"
    gt_dir = os.path.join(os.path.dirname(os.path.dirname(source_dir)), "relight_gt", f"{case_name}_{env_map}")
    all_cams = []
    img_num = len(glob(os.path.join(gt_dir, "*.pkl")))

    for i in range(img_num):
        cam = deepcopy(scene.getTestCameras()[0])
        w2c = read_pickle(os.path.join(gt_dir, f"{i}-camera.pkl"))[0]
        w2c = np.concatenate([w2c, np.ones((1, 4))], axis=0)
        cam.world_view_transform = torch.from_numpy(w2c.T).float().cuda()
        cam.full_proj_transform = (
            cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
        all_cams.append(cam)
    return all_cams


def render_lightings(model_path, name, iteration, gaussians, sample_num):
    lighting_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(lighting_path, exist_ok=True)
    # sampled_indicies = torch.randperm(gaussians.get_xyz.shape[0])[:sample_num]
    sampled_indicies = torch.arange(gaussians.get_xyz.shape[0], dtype=torch.long)[:sample_num]
    for sampled_index in tqdm(sampled_indicies, desc="Rendering lighting progress"):
        lighting = render_lighting(gaussians, sampled_index=sampled_index)
        torchvision.utils.save_image(
            lighting,
            os.path.join(lighting_path, "{0:05d}".format(sampled_index) + ".png"),
        )
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
                linear2srgb(render_pkg["render"]),
                os.path.join(render_path, "{0:05d}".format(idx) + ".png"),
            )
            torchvision.utils.save_image(linear2srgb(gt), os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))
        else:
            torchvision.utils.save_image(
                render_pkg["render"],
                os.path.join(render_path, "{0:05d}".format(idx) + ".png"),
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
        keys = [
            "render",
            "diffuse",
            "diffuse_color",
            "specular",
            "specular_color",
            "roughness",
            "albedo",
            "metallic",
        ]
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


def render_relighting(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, "test", "ours_{}".format(iteration), "relighting")
    # gts_path = os.path.join(model_path, "test", "ours_{}".format(iteration), "relighting_gt")

    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()

        render_pkg = render(view, gaussians, pipeline, background, debug=False)

        torch.cuda.synchronize()

        torchvision.utils.save_image(
            render_pkg["render"],
            os.path.join(render_path, "{0:05d}".format(idx) + ".png"),
        )


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    skip_relighting: bool,
):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.sh_degree,
            dataset.brdf_dim,
            pipeline.brdf_mode,
            dataset.brdf_envmap_res,
        )
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

        if not skip_relighting:
            if "nerosync" in dataset.model_path:
                env_maps = ["corridor", "golf", "neon"]

                for env_map in env_maps:
                    views = nerosync_relighting_poses(scene, env_map)

                    env_map_path = os.path.join(
                        os.path.dirname(dataset.source_path), "hdr", f"{NEROSYNC_ENVNAME2PATH[env_map]}.exr"
                    )

                    envmap_res = dataset.brdf_envmap_res
                    env_light = load_env(env_map_path, res=[envmap_res, envmap_res])
                    env_light.eval()
                    gaussians.brdf_mlp = env_light

                    render_relighting(
                        dataset.model_path,
                        env_map,
                        scene.loaded_iter,
                        views,
                        gaussians,
                        pipeline,
                        background,
                    )

        if pipeline.brdf:
            render_lightings(
                dataset.model_path,
                "lighting",
                scene.loaded_iter,
                gaussians,
                sample_num=1,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_relighting", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.skip_relighting,
    )
