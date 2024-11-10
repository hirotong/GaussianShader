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

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torchvision.transforms.functional as tf
import trimesh
from PIL import Image
from tqdm import tqdm

from arguments import get_config_args
from lpipsPyTorch import lpips
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.mesh_utils import MeshEvaluator, to_cam_open3d
from utils.point_utils import mask_depth_to_pts, pose_apply, pose_inverse, project_points


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def rasterize_depth_map(mesh: trimesh.Trimesh, camera: o3d.camera.PinholeCameraParameters):
    import nvdiffrast.torch as dr

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    pts, depth = project_points(vertices, camera.extrinsic[:3], camera.intrinsic.intrinsic_matrix)
    # normalize to projection
    h, w = camera.intrinsic.height, camera.intrinsic.width
    pts[:, 0] = (pts[:, 0] * 2 - w) / w
    pts[:, 1] = (pts[:, 1] * 2 - h) / h
    # pts[:, 0] = (pts[:, 0] * 2 - w + 1) / (w - 1)
    # pts[:, 1] = (pts[:, 1] * 2 - h + 1) / (h - 1)
    near, far = 5e-1, 1e2
    z = (depth - near) / (far - near)
    z = z * 2 - 1
    pts_clip = np.concatenate([pts, z[:, None]], axis=1)

    pts_clip = torch.from_numpy(pts_clip.astype(np.float32)).cuda()
    indices = torch.from_numpy(faces.astype(np.int32)).cuda()
    pts_clip = torch.cat([pts_clip, torch.ones_like(pts_clip[..., 0:1])], 1).unsqueeze(0)

    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, pts_clip, indices, (h, w))  # [1,h,w,4]
    depth = (rast[0, :, :, 2] + 1) / 2 * (far - near) + near
    mask = rast[0, :, :, -1] != 0
    return depth.cpu().numpy(), mask.cpu().numpy().astype(bool)


def get_mesh_eval_points(scene: Scene, mesh_path, num_points=100_000):
    viewpoints = scene.getTestCameras()

    mesh = trimesh.load_mesh(mesh_path, process=False, force="mesh")
    # if False:
    cameras = to_cam_open3d(viewpoints)
    pbar = tqdm(len(cameras))
    pts_pr = []
    for idx, cam in enumerate(cameras):
        if viewpoints[idx].gt_alpha_mask is not None:
            gt_alpha_mask = viewpoints[idx].gt_alpha_mask.squeeze().cpu().numpy().astype(bool)
        depth_pr, mask_pr = rasterize_depth_map(mesh, cam)
        mask_pr = mask_pr & gt_alpha_mask
        pts_ = mask_depth_to_pts(depth_pr, mask_pr, cam)
        pose = pose_inverse(cam.extrinsic)
        pts_pr.append(pose_apply(pose, pts_))
        pbar.update(1)

    pts_pr = np.concatenate(pts_pr, axis=0).astype(np.float32)
    eval_pcd = o3d.geometry.PointCloud()
    eval_pcd.points = o3d.utility.Vector3dVector(pts_pr)
    o3d.io.write_point_cloud((mesh_path.parent / "eval_pts_ori.ply").as_posix(), eval_pcd)
    eval_pcd = eval_pcd.voxel_down_sample(voxel_size=0.01)
    o3d.io.write_point_cloud((mesh_path.parent / "eval_pts.ply").as_posix(), eval_pcd)
    eval_pcd_post, idx = eval_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=5)
    print(f"\nRemoving {len(eval_pcd.points) - len(eval_pcd_post.points)} points")
    o3d.io.write_point_cloud((mesh_path.parent / "eval_pts_post.ply").as_posix(), eval_pcd_post)
    print(f"eval_pcd_post: {len(eval_pcd_post.points)}")
    eval_pcd = eval_pcd_post

    return eval_pcd


def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        # try:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg"))

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
            })
            per_view_dict[scene_dir][method].update({
                "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
            })

        with open(scene_dir + "/results.json", "w") as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", "w") as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)
        # except:
        #     print("Unable to compute metrics for model", scene_dir)


def convert_to_serializable(obj):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    return obj


def evaluate_mesh(model_paths, n_points, visualize_pcd=True):
    full_dict = {}
    mesh_evaluator = MeshEvaluator(n_points)

    for scene_dir in model_paths:
        full_dict[scene_dir] = {}
        print("Scene:", scene_dir)

        cfg_args = get_config_args(scene_dir)
        train_dir = Path(scene_dir) / "train"
        source_path = Path(cfg_args.source_path)

        gaussians = GaussianModel(cfg_args.sh_degree)
        scene = Scene(cfg_args, gaussians, shuffle=False)

        pcd_gt = o3d.io.read_point_cloud((source_path / "eval_points.ply").as_posix())

        for method in os.listdir(train_dir):
            print("Method:", method)
            method_dir = train_dir / method

            # mesh_pred = trimesh.load(method_dir / "fuse_post.ply", process=False, force="mesh")
            # pcd_pred = o3d.io.read_point_cloud((method_dir / "pred_points.ply").as_posix())
            pcd_pred = get_mesh_eval_points(scene, method_dir / "fuse_mesh.ply")
            # pcd_pred = get_mesh_eval_points(scene, method_dir / "fuse.ply")

            pointcloud_pred = np.asarray(pcd_pred.points)
            pointcloud_tgt = np.asarray(pcd_gt.points)

            # mesh_eval_dict, pred2gt_pcd, gt2pred_pcd = mesh_evaluator.eval_mesh(
            #     mesh_pred, pointcloud_tgt, None, visualize_pcd=visualize_pcd
            # )
            mesh_eval_dict, pred2gt_pcd, gt2pred_pcd = mesh_evaluator.eval_pointcloud(
                pointcloud_pred, pointcloud_tgt, None, visualize_pcd=visualize_pcd
            )

            if visualize_pcd:
                save_dir = method_dir / "vis_pcd"

                if not save_dir.exists():
                    save_dir.mkdir()

                o3d.io.write_point_cloud((save_dir / "pred2gt.ply").as_posix(), pred2gt_pcd)
                o3d.io.write_point_cloud((save_dir / "gt2pred.ply").as_posix(), gt2pred_pcd)

            mesh_eval_dict["n_points"] = n_points

            full_dict[scene_dir][method] = mesh_eval_dict

            print("  Mesh evaluation results:")
            print("    Chamfer distance L1: {:>12.7f}".format(mesh_eval_dict["chamfer-L1"]))

        serializable_data = convert_to_serializable(full_dict[scene_dir])

        with open(scene_dir + "/results_mesh.json", "w") as fp:
            json.dump(serializable_data, fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)

    evaluate_mesh(args.model_paths, 100_000, visualize_pcd=True)
