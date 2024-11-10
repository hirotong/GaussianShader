import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scene.cameras import Camera


def depths_to_points(view: Camera, depthmap):
    W, H = view.image_width, view.image_height
    intrinsics, extrinsics = view.get_calib_matrix_nerf()
    c2w = torch.inverse(extrinsics)

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device="cuda").float(), torch.arange(H, device="cuda").float(), indexing="xy"
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrinsics.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(view, depth):
    """
    view: view camera
    depth: depthmap"""
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


# https://github.com/liuyuan-pal/NeRO/blob/main/utils/base_utils.py
def project_points(pts, RT, K):
    pts = np.matmul(pts, RT[:, :3].transpose()) + RT[:, 3:].transpose()
    pts = np.matmul(pts, K.transpose())
    dpt = pts[:, 2]
    mask0 = (np.abs(dpt) < 1e-4) & (np.abs(dpt) > 0)
    if np.sum(mask0) > 0:
        dpt[mask0] = 1e-4
    mask1 = (np.abs(dpt) > -1e-4) & (np.abs(dpt) < 0)
    if np.sum(mask1) > 0:
        dpt[mask1] = -1e-4
    pts2d = pts[:, :2] / dpt[:, None]
    return pts2d, dpt


def mask_depth_to_pts(depth, mask, camera, rgb=None):
    """
    Convert masked depth map to 3D points.

    Args:
        depth (np.ndarray): Depth map.
        mask (np.ndarray): Binary mask of the depth map.
        camera (Camera): Camera object containing intrinsic parameters.
        rgb (np.ndarray, optional): RGB image corresponding to the depth map.

    Returns:
        np.ndarray: 3D points in camera coordinate system.
        np.ndarray: RGB values of the points (if rgb is provided).
    """
    K = camera.intrinsic.intrinsic_matrix
    hs, ws = np.nonzero(mask)
    depth = depth[hs, ws]
    pts = np.asarray([ws, hs, depth], np.float32).transpose()
    pts[:, :2] *= pts[:, 2:]
    if rgb is not None:
        return np.dot(pts, np.linalg.inv(K).transpose()), rgb[hs, ws]
    else:
        return np.dot(pts, np.linalg.inv(K).transpose())


def pose_inverse(pose):
    R = pose[:3, :3].T
    t = -R @ pose[:3, 3]
    pose_inv = np.eye(4)
    pose_inv[:3, :3] = R
    pose_inv[:3, 3] = t
    return pose_inv


def transform_points_pose(pts, pose):
    R, t = pose[:3, :3], pose[:3, 3]
    if len(pts.shape) == 1:
        return (R @ pts[:, None] + t[:, None])[:, 0]
    return pts @ R.T + t[None, :]


def pose_apply(pose, pts):
    return transform_points_pose(pts, pose)
