import numpy as np
import open3d as o3d
import quaternion
import time
import torch
import cv2

# def get_pointcloud_from_depth(rgb: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray):
#     if len(depth.shape) == 3:
#         depth = depth[:, :, 0]
#     filter_z, filter_x = np.where(depth > 0)
#     depth_values = depth[filter_z, filter_x]
#     pixel_z = (depth.shape[0] - filter_z - intrinsic[1][2]) * depth_values / intrinsic[1][1]
#     pixel_x = (filter_x - intrinsic[0][2]) * depth_values / intrinsic[0][0]
#     pixel_y = depth_values
#     color_values = rgb[filter_z, filter_x]
#     point_values = np.stack([pixel_x, pixel_y, pixel_z], axis=-1)
#     return point_values, color_values


def get_pointcloud_from_depth(rgb: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray):
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    filter_z, filter_x = np.where(depth > 0)
    depth_values = depth[filter_z, filter_x]
    pixel_z = (depth.shape[0] - 1 - filter_z - intrinsic[1][2]) * depth_values / intrinsic[1][1]
    pixel_x = (filter_x - intrinsic[0][2]) * depth_values / intrinsic[0][0]
    pixel_y = depth_values
    color_values = rgb[filter_z, filter_x]
    point_values = np.stack([pixel_x, pixel_z, -pixel_y], axis=-1)
    return point_values, color_values


# def get_pointcloud_from_depth_mask(depth: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray):
#     if len(depth.shape) == 3:
#         depth = depth[:, :, 0]
#     if len(mask.shape) == 3:
#         mask = mask[:, :, 0]
#     filter_z, filter_x = np.where((depth > 0) & (mask > 0))
#     depth_values = depth[filter_z, filter_x]
#     pixel_z = (depth.shape[0] - filter_z - intrinsic[1][2]) * depth_values / intrinsic[1][1]
#     pixel_x = (filter_x - intrinsic[0][2]) * depth_values / intrinsic[0][0]
#     pixel_y = depth_values
#     point_values = np.stack([pixel_x, pixel_y, pixel_z], axis=-1)
#     return point_values


def get_pointcloud_from_depth_mask(depth: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray):
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    filter_z, filter_x = np.where((depth > 0) & (mask > 0))
    depth_values = depth[filter_z, filter_x]
    pixel_z = (depth.shape[0] - 1 - filter_z - intrinsic[1][2]) * depth_values / intrinsic[1][1]
    pixel_x = (filter_x - intrinsic[0][2]) * depth_values / intrinsic[0][0]
    pixel_y = depth_values
    point_values = np.stack([pixel_x, pixel_z, -pixel_y], axis=-1)
    return point_values


# def translate_to_world(pointcloud, position, rotation):
#     extrinsic = np.eye(4)
#     extrinsic[0:3, 0:3] = np.linalg.inv(quaternion.as_rotation_matrix(rotation))
#     extrinsic[0:3, 3] = position
#     world_points = np.matmul(extrinsic, np.concatenate((pointcloud, np.ones((pointcloud.shape[0], 1))), axis=-1).T).T
#     return world_points[:, 0:3]


def translate_to_world(pointcloud, position, rotation):
    extrinsic = np.eye(4)
    extrinsic[0:3, 0:3] = rotation
    extrinsic[0:3, 3] = position
    world_points = np.matmul(extrinsic, np.concatenate((pointcloud, np.ones((pointcloud.shape[0], 1))), axis=-1).T).T
    return world_points[:, 0:3]


def pointcloud_distance(pcdA, pcdB, device='cpu'):
    pointsA = torch.tensor(pcdA.point["positions"].cpu().numpy(), device=device)
    pointsB = torch.tensor(pcdB.point["positions"].cpu().numpy(), device=device)
    cdist = torch.cdist(pointsA, pointsB)
    min_distances1, _ = cdist.min(dim=1)
    return min_distances1


def cpu_pointcloud_from_array(points, colors):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)
    return pointcloud


def gpu_pointcloud_from_array(points, colors, device):
    pointcloud = o3d.t.geometry.PointCloud()
    pointcloud.point["positions"] = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32, device=device)
    pointcloud.point["colors"] = o3d.core.Tensor(colors.astype(np.float32) / 255.0, dtype=o3d.core.Dtype.Float32, device=device)
    return pointcloud


def gpu_pointcloud(pointcloud, device):
    new_pointcloud = o3d.t.geometry.PointCloud()
    new_pointcloud.point["positions"] = o3d.core.Tensor(pointcloud.points, device=device)
    new_pointcloud.point["colors"] = o3d.core.Tensor(pointcloud.colors, device=device)
    return new_pointcloud


def cpu_pointcloud(pointcloud):
    new_pointcloud = o3d.geometry.PointCloud()
    new_pointcloud.points = o3d.utility.Vector3dVector(pointcloud.point["positions"].cpu().numpy())
    new_pointcloud.colors = o3d.utility.Vector3dVector(pointcloud.point["colors"].cpu().numpy())
    return new_pointcloud


def cpu_merge_pointcloud(pcdA, pcdB):
    return pcdA + pcdB


def gpu_merge_pointcloud(pcdA, pcdB):
    if pcdA.is_empty():
        return pcdB
    if pcdB.is_empty():
        return pcdA
    return pcdA + pcdB


def gpu_cluster_filter(pointcloud, eps=0.3, min_points=20):
    labels = pointcloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    numpy_labels = labels.cpu().numpy()
    unique_labels = np.unique(numpy_labels)
    largest_cluster_label = max(unique_labels, key=lambda x: np.sum(numpy_labels == x))
    largest_cluster_pc = pointcloud.select_by_index((labels == largest_cluster_label).nonzero()[0])
    return largest_cluster_pc


def cpu_cluster_filter(pointcloud, eps=0.3, min_points=20):
    labels = pointcloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    unique_labels = np.unique(labels)
    largest_cluster_label = max(unique_labels, key=lambda x: np.sum(labels == x))
    largest_cluster_pc = pointcloud.select_by_index((labels == largest_cluster_label).nonzero()[0])
    return largest_cluster_pc
