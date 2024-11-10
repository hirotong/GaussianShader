import numpy as np
import open3d as o3d

# import pymeshlab as pml
import sklearn.neighbors as skln
import torch
import trimesh


def poisson_mesh_reconstruction(points, normals=None):
    # points/normals: [N, 3] np.ndarray

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # outlier removal
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=10)

    # normals
    if normals is None:
        pcd.estimate_normals()
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals[ind])

    # visualize
    o3d.visualization.draw_geometries([pcd], point_show_normal=False)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # visualize
    o3d.visualization.draw_geometries([mesh])

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    print(f"[INFO] poisson mesh reconstruction: {points.shape} --> {vertices.shape} / {triangles.shape}")

    return vertices, triangles


def decimate_mesh(verts, faces, target, backend="pymeshlab", remesh=False, optimalplacement=True):
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == "pyfqmr":
        import pyfqmr

        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
        verts, faces, normals = solver.getMesh()
    else:
        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, "mesh")  # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.PercentageValue(1))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target), optimalplacement=optimalplacement)

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.PercentageValue(1))

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    print(f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}")

    return verts, faces


def clean_mesh(
    verts,
    faces,
    v_pct=1,
    min_f=64,
    min_d=20,
    repair=True,
    remesh=True,
    remesh_size=0.01,
):
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, "mesh")  # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(threshold=pml.PercentageValue(v_pct))  # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pml.PercentageValue(min_d))

    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.PureValue(remesh_size))

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}")

    return verts, faces


# Maximum values for bounding box [-1, 1]^3
EMPTY_PCL_DICT = {
    "completeness": np.sqrt(3),
    "accuracy": np.sqrt(3),
    "completeness2": 3,
    "accuracy2": 3,
    "chamfer": 6,
}

EMPTY_PCL_DICT_NORMALS = {
    "normals completeness": -1.0,
    "normals accuracy": -1.0,
    "normals": -1.0,
}


class MeshEvaluator:
    """Mesh evaluation class.

    It handles the mesh evaluation process.

    Args:
        n_points (int): number of points to be used for evaluation
    """

    def __init__(self, n_points=100_000):
        self.n_points = n_points

    def eval_mesh(
        self, mesh: trimesh.Trimesh, pointcloud_tgt, normals_tgt=None, remove_wall=False, visualize_pcd=True
    ):
        """Evaluates a mesh.

        Args:
            mesh (trimesh): mesh which should be evaluated
            pointcloud_tgt (numpy array): target point cloud
            normals_tgt (numpy array): target normals
            points_iou (numpy_array): points tensor for IoU evaluation
            occ_tgt (numpy_array): GT occupancy values for IoU points
        """
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            if remove_wall:  #! Remove walls and floors
                pointcloud, idx = mesh.sample(2 * self.n_points, return_index=True)
                eps = 0.007
                x_max, x_min = pointcloud_tgt[:, 0].max(), pointcloud_tgt[:, 0].min()
                y_max, y_min = pointcloud_tgt[:, 1].max(), pointcloud_tgt[:, 1].min()
                z_max, z_min = pointcloud_tgt[:, 2].max(), pointcloud_tgt[:, 2].min()

                # add small offsets
                x_max, x_min = x_max + eps, x_min - eps
                y_max, y_min = y_max + eps, y_min - eps
                z_max, z_min = z_max + eps, z_min - eps

                mask_x = (pointcloud[:, 0] <= x_max) & (pointcloud[:, 0] >= x_min)
                mask_y = pointcloud[:, 1] >= y_min  # floor
                mask_z = (pointcloud[:, 2] <= z_max) & (pointcloud[:, 2] >= z_min)

                mask = mask_x & mask_y & mask_z
                pointcloud_new = pointcloud[mask]
                # Subsample
                idx_new = np.random.randint(pointcloud_new.shape[0], size=self.n_points)
                pointcloud = pointcloud_new[idx_new]
                idx = idx[mask][idx_new]
            else:
                pointcloud, idx = mesh.sample(self.n_points, return_index=True)

            pointcloud = pointcloud.astype(np.float32)
            normals = mesh.face_normals[idx]

        else:
            pointcloud = np.empty((0, 3))
            normals = np.empty((0, 3))

        out_dict = self.eval_pointcloud(pointcloud, pointcloud_tgt, normals, normals_tgt, visualize_pcd=visualize_pcd)

        # if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
        #     occ = check_mesh_contains(mesh, points_iou)
        #     out_dict["iou"] = compute_iou(occ, occ_tgt)
        # else:
        #     out_dict["iou"] = 0.0

        return out_dict

    def eval_pointcloud(
        self,
        pointcloud,
        pointcloud_tgt,
        normals=None,
        normals_tgt=None,
        thresholds=np.linspace(1.0 / 1000, 1, 1000),
        visualize_pcd=True,
    ):
        """Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
            thresholds (numpy array): threshold values for the F-score calculation
        """
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            print("Empty pointcloud / mesh detected!")
            out_dict = EMPTY_PCL_DICT.copy()
            if normals is not None and normals_tgt is not None:
                out_dict.update(EMPTY_PCL_DICT_NORMALS)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from the predicted point cloud
        completeness, completeness_normals = distance_p2p(pointcloud_tgt, normals_tgt, pointcloud, normals)
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness**2

        completeness_mean = completeness.mean()
        completeness2_mean = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(pointcloud, normals, pointcloud_tgt, normals_tgt)
        precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy**2

        accuracy_mean = accuracy.mean()
        accuracy2_mean = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2_mean + accuracy2_mean)
        normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals
        chamferL1 = 0.5 * (completeness_mean + accuracy_mean)

        # F-Score
        F = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision))]

        out_dict = {
            "completeness": completeness_mean,
            "accuracy": accuracy_mean,
            "normals completeness": completeness_normals,
            "normals accuracy": accuracy_normals,
            "normals": normals_correctness,
            "completeness2": completeness2_mean,
            "accuracy2": accuracy2_mean,
            "chamfer-L2": chamferL2,
            "chamfer-L1": chamferL1,
            "f-score": F[9],  # threshold = 1.0%
            "f-score-15": F[14],  # threshold = 1.5%
            "f-score-20": F[19],  # threshold = 2.0%
        }

        # visualize error
        if visualize_pcd:
            # vis_dis = 0.05
            # R = np.array([[1, 0, 0]], dtype=np.float64)
            # G = np.array([[0, 1, 0]], dtype=np.float64)
            # B = np.array([[0, 0, 1]], dtype=np.float64)
            # W = np.array([[1, 1, 1]], dtype=np.float64)
            # data_color = np.tile(B, (pointcloud.shape[0], 1))
            # data_alpha = accuracy.clip(max=vis_dis) / vis_dis
            # data_color = R * data_alpha + W * (1 - data_alpha)
            # data_color[np.where(accuracy[:, 0] >= vis_dis)] = G
            # d2s_pcd = o3d.geometry.PointCloud()
            # d2s_pcd.points = o3d.utility.Vector3dVector(pointcloud)
            # d2s_pcd.colors = o3d.utility.Vector3dVector(data_color)

            # gt_color = np.tile(B, (pointcloud_tgt.shape[0], 1))
            # gt_alpha = completeness.clip(max=vis_dis) / vis_dis
            # gt_color = R * gt_alpha + W * (1 - gt_alpha)
            # gt_color[np.where(completeness[:, 0] >= vis_dis)] = G
            # s2d_pcd = o3d.geometry.PointCloud()
            # s2d_pcd.points = o3d.utility.Vector3dVector(pointcloud_tgt)
            # s2d_pcd.colors = o3d.utility.Vector3dVector(gt_color)

            import open3d as o3d
            from matplotlib import cm

            colormap = cm.get_cmap("jet")
            # colormap = np.array(colormap.colors)

            error_thresh = (np.percentile(accuracy, 95) + np.percentile(completeness, 95)) / 2
            error_thresh = 0.02
            print(error_thresh)
            error_d2s = accuracy.copy()
            error_d2s = error_d2s.clip(max=error_thresh) / error_thresh  # Normalize to [0, 1]
            error_d2s = (error_d2s * 255).astype(np.uint8)
            error_d2s_vis = colormap(error_d2s[..., 0])
            d2s_pcd = o3d.geometry.PointCloud()
            d2s_pcd.points = o3d.utility.Vector3dVector(pointcloud)
            d2s_pcd.colors = o3d.utility.Vector3dVector(error_d2s_vis[..., :3])

            error_s2d = completeness.copy()
            error_s2d = error_s2d.clip(max=error_thresh) / error_thresh  # Normalize to [0, 1]
            error_s2d = (error_s2d * 255).astype(np.uint8)
            error_s2d_vis = colormap(error_s2d[..., 0])
            s2d_pcd = o3d.geometry.PointCloud()
            s2d_pcd.points = o3d.utility.Vector3dVector(pointcloud_tgt)
            s2d_pcd.colors = o3d.utility.Vector3dVector(error_s2d_vis[..., :3])

            return out_dict, d2s_pcd, s2d_pcd

        return out_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = skln.KDTree(points_tgt)
    dist, idx = kdtree.query(points_src, k=1)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def distance_p2m(points, mesh):
    """Compute minimal distances of each point in points to mesh.

    Args:
        points (numpy array): points array
        mesh (trimesh): mesh

    """
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist


def get_threshold_percentage(dist, thresholds):
    """Evaluates a point cloud.

    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [(dist <= t).mean() for t in thresholds]
    return in_threshold


def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([[W / 2, 0, 0, (W - 1) / 2], [0, H / 2, 0, (H - 1) / 2], [0, 0, 0, 1]]).float().cuda().T
        intrins = (viewpoint_cam.projection_matrix @ ndc2pix)[:3, :3].T
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx=intrins[0, 2].item(),
            cy=intrins[1, 2].item(),
            fx=intrins[0, 0].item(),
            fy=intrins[1, 1].item(),
        )
        extrinsic = np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj
