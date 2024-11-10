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
import sys
from glob import glob
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import k2focal
from utils.general_utils import read_pickle
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.image_utils import linear2srgb, srgb2linear
from utils.sh_utils import SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    R: np.array  # w2c but transposed
    T: np.array  # w2c
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    normal_image: np.array
    alpha_mask: np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, (
                "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            )

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            normal_image=None,
            alpha_mask=None,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", linear=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if "camera_angle_x" not in contents.keys():
            fovx = None
        else:
            fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            # R = -np.transpose(matrix[:3,:3])
            # R[:,0] = -R[:,0]
            # T = -matrix[:3, 3]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            if linear:
                norm_data = srgb2linear(norm_data)
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            alpha_mask = norm_data[:, :, 3]
            alpha_mask = Image.fromarray(np.array(alpha_mask * 255.0, dtype=np.byte), "L")
            # arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=-1)
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")

            normal_cam_name = os.path.join(path, frame["file_path"] + "_normal" + extension)
            normal_image_path = os.path.join(path, normal_cam_name)
            if os.path.exists(normal_image_path):
                normal_image = Image.open(normal_image_path)

                normal_im_data = np.array(normal_image.convert("RGBA"))
                normal_bg_mask = (normal_im_data == 128).sum(-1) == 3
                normal_norm_data = normal_im_data / 255.0
                normal_arr = normal_norm_data[:, :, :3] * normal_norm_data[:, :, 3:4] + bg * (
                    1 - normal_norm_data[:, :, 3:4]
                )
                normal_arr[normal_bg_mask] = 0
                normal_image = Image.fromarray(np.array(normal_arr * 255.0, dtype=np.byte), "RGB")
            else:
                normal_image = None

            if fovx == None:
                focal_length = contents["fl_x"]
                FovY = focal2fov(focal_length, image.size[1])
                FovX = focal2fov(focal_length, image.size[0])
            else:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovx
                FovX = fovy

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                    normal_image=normal_image,
                    alpha_mask=alpha_mask,
                )
            )

    return cam_infos


def readCamerasFromNeROSync(path, white_background, extension=".png", load_priors=False) -> List[CameraInfo]:
    dataset_name = os.path.basename(path)
    dataset_folder = os.path.dirname(path)

    num_imgs = len(glob(os.path.join(path, "*.pkl")))

    cam_infos = []

    for i in tqdm(range(num_imgs), desc="Reading NeROSync Cameras"):
        pkl_path = os.path.join(path, f"{i}-camera.pkl")
        cam = read_pickle(pkl_path)

        w2c = cam[0]
        K = cam[1]

        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        fx, fy, cx, cy = k2focal(K)

        image_path = os.path.join(path, f"{i}.png")
        image_name = Path(image_path).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        norm_data = im_data / 255.0

        depth_path = os.path.join(path, f"{i}-depth.png")
        depth = Image.open(depth_path)
        depth_data = np.array(depth.convert("I;16")).astype(np.float32) / 65535.0 * 15.0
        mask = (depth_data < 14.5).astype(np.float32)

        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        arr = norm_data[:, :, :3] * mask[..., None] + bg * (1 - mask[..., None])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
        alpha_mask = Image.fromarray(np.array(mask * 255.0, dtype=np.byte), "L")

        normal_image = None

        if load_priors:
            depth_prior_dir = os.path.join(path, "mldepth_priors")
            normal_prior_dir = os.path.join(path, "marigold_priors")
            assert os.path.exists(depth_prior_dir), f"Depth prior directory {depth_prior_dir} does not exist"
            assert os.path.exists(normal_prior_dir), f"Normal prior directory {normal_prior_dir} does not exist"

            depth_prior_path = os.path.join(depth_prior_dir, "depth_npy", f"{i}_pred.npy")
            normal_prior_path = os.path.join(normal_prior_dir, "normal_npy", f"{i}_pred.npy")

            depth_prior = np.load(depth_prior_path)
            normal_prior = np.load(normal_prior_path)

            depth_prior = depth_prior * mask
            normal_prior *= -1
            normal_prior = normal_prior.transpose(1, 2, 0) @ R.T
            normal_prior = normal_prior * mask[..., None]
        else:
            depth_prior = None
            normal_prior = None
        FovX = focal2fov(fx, image.size[0])
        FovY = focal2fov(fy, image.size[1])

        cam_infos.append(
            CameraInfo(
                uid=i,
                R=R,
                T=T,
                FovX=FovX,
                FovY=FovY,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
                normal_image=normal_image,
                alpha_mask=alpha_mask,
            )
        )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png", linear=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, linear)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, linear)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")

    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readNeROSyncInfo(
    path, white_background, eval, extension=".png", load_priors=False, train_num_images=-1
) -> SceneInfo:
    print("Reading NeRO Synthetic Info.")

    cam_infos = readCamerasFromNeROSync(path, white_background, extension, load_priors)

    dataset_folder = os.path.dirname(path)
    test_ids, train_ids = read_pickle(os.path.join(dataset_folder, "synthetic_split_128.pkl"))
    train_cam_infos = cam_infos
    test_cam_infos = [cam_infos[int(i)] for i in test_ids]

    if train_num_images > 0:
        sample_indices_dict = read_pickle(os.path.join(dataset_folder, "sample_indices.pkl"))
        train_cam_infos = [train_cam_infos[int(i)] for i in sample_indices_dict[train_num_images]]

    print(f"NeROSync Data Set has {len(train_cam_infos)} training and {len(test_cam_infos)} test cameras.")
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "NeROSync": readNeROSyncInfo,
}
