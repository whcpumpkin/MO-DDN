from mapping_utils.geometry import get_pointcloud_from_depth, get_pointcloud_from_depth_mask, translate_to_world, pointcloud_distance, cpu_pointcloud_from_array, gpu_cluster_filter, gpu_pointcloud_from_array, gpu_merge_pointcloud
from mapping_utils.preprocess import preprocess_depth, preprocess_image
from mapping_utils.projection import project_frontier, translate_grid_to_point, translate_point_to_grid, project_costmap
from mapping_utils.transform import habitat_camera_intrinsic, habitat_translation, habitat_rotation, realworld_translation, realworld_rotation
from mapping_utils.path_planning import visualize_path, path_planning
from cv_utils.image_percevior import GLEE_Percevior
# from matplotlib import colormaps
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import time
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from PIL import Image
import os
from habitat.utils.common import draw_single_bounding_boxes_and_labels
import requests
import json


def _select_tensor_pointcloud_by_mask(tpcd, mask_indices, device):
    """Select points from a tensor `PointCloud` using integer indices.

    This helper converts to a legacy CPU `o3d.geometry.PointCloud` for
    selection (which is more portable across Open3D builds), then converts
    the selected points back to a tensor pointcloud on `device`.
    """
    from mapping_utils.geometry import cpu_pointcloud, gpu_pointcloud_from_array

    if len(mask_indices) == 0:
        # return an empty tensor pointcloud
        return o3d.t.geometry.PointCloud()

    legacy = cpu_pointcloud(tpcd)
    sel = legacy.select_by_index(mask_indices.tolist())
    pts = np.asarray(sel.points)
    cols = np.asarray(sel.colors) * 255.0
    return gpu_pointcloud_from_array(pts, cols, device)


def bbox_to_mask(bbox, width, height):
    mask = np.zeros((height, width))
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    mask[y1:y2, x1:x2] = 1
    return mask


def visualize_segmentation(image, classes, masks):
    copy_image = image.copy()
    label_classes = np.unique(classes)
    for cls, mask in zip(classes, masks):
        if len(np.unique(mask)) != 2:
            continue
        copy_image[np.where(mask == 1)] = d3_40_colors_rgb[label_classes.tolist().index(cls) % 40]
        x, y = int(np.mean(np.where(mask)[1])), int(np.mean(np.where(mask)[0]))
        cv2.putText(copy_image.astype(np.uint8), str(cls), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    ret_image = cv2.addWeighted(image, 0.2, copy_image, 0.8, 0)
    return ret_image


def ask_for_percevior(image, label=None):
    header = {
        "Content-Type": "application/json",
        "Authorization": "whcpumpkin",
        # "Organization": "org-PmiFO2CwwRvqBO0UpCYfKqgs"
    }
    width, height, channel = image.shape
    if channel == 4:
        image = image[:, :, :3]
    if isinstance(image, list) is False:
        image = image.tolist()
    if label is None:
        with open("data/datasets/ddnplus/hssd-hab_v0.2.5/train/train.json", "r") as f:
            label = json.load(f)["category_to_task_category_id"].keys()
    label = [obj.split(".n")[0].replace("_", " ") for obj in label]
    label = list(set(label))
    label = ','.join(label)
    texts = label.split(",")
    idx_to_label = {i: l for i, l in enumerate(texts)}
    post_dict = {
        "messages": [{
            "rgb": image,
            "confidence_threshold": 0.25,
            "area_threshold": 100,
        }],
        # "temperature": 1.5,
    }
    session = requests.Session()
    session.proxies = {}
    while True:
        try:
            r = session.post("http://127.0.0.1:56342/generate/", json=post_dict, headers=header)
            response = r.json()
            labels = response['classes']
            mask_array = np.array(response['masks'])
            confidences = response['confidences']
            visualization = np.array(response['visualization'])
            break
        except Exception as e:
            print(e)
            time.sleep(1)
            continue
    return labels, mask_array, confidences, visualization


class Mapper:

    def __init__(self,
                 camera_intrinsic,
                 pcd_resolution=0.05,
                 grid_resolution=0.1,
                 grid_size=5,
                 floor_height=-0.8,
                 ceiling_height=0.8,
                 translation_func=habitat_translation,
                 rotation_func=habitat_rotation,
                 device='cuda:0',
                 block_size=2,
                 detector_runner=None,
                 args=None):
        self.camera_intrinsic = camera_intrinsic
        self.pcd_resolution = pcd_resolution
        self.grid_resolution = grid_resolution
        self.grid_size = grid_size
        self.floor_height = floor_height
        self.ceiling_height = ceiling_height
        self.translation_func = translation_func
        self.rotation_func = rotation_func
        self.object_percevior = GLEE_Percevior(device='cuda:0')
        # self.object_percevior = ask_for_percevior
        # self.detector_runner = detector_runner
        self.pcd_device = o3d.core.Device('cpu:0'.upper())
        self.block_map = {}
        self.block_size = block_size
        self.args = args

    def reset(self, position, rotation):
        self.update_iterations = 0
        self.initial_position = self.translation_func(position)
        self.current_position = self.translation_func(position) - self.initial_position
        self.current_rotation = self.rotation_func(rotation)
        self.scene_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.object_entities = []
        self.room_entities = []
        self.trajectory_position = []
        self.navigable_pcd_idx_in_useful_pcd = []
        self.project_block = {}
        self.block_map = {}
        self.block_navigation_map = {}

    def update(self, rgb, depth, position, rotation, object_category_idx=None):
        self.current_position = self.translation_func(position) - self.initial_position
        self.current_rotation = self.rotation_func(rotation)
        self.current_depth = preprocess_depth(depth, upper_bound=9.9)
        self.current_rgb = preprocess_image(rgb)
        self.trajectory_position.append(self.current_position)
        if np.sum(self.current_depth) > 0:
            camera_points, camera_colors = get_pointcloud_from_depth(self.current_rgb, self.current_depth, self.camera_intrinsic)
            world_points = translate_to_world(camera_points, self.current_position, self.current_rotation)
            self.current_pcd = gpu_pointcloud_from_array(world_points, camera_colors, self.pcd_device).voxel_down_sample(self.pcd_resolution)

        # select points above a height threshold without relying on tensor select_by_index
        positions = self.current_pcd.point["positions"].cpu().numpy()
        mask_idx = np.where(positions[:, 2] > self.floor_height - 0.3)[0]
        filter_pcd = _select_tensor_pointcloud_by_mask(self.current_pcd, mask_idx, self.pcd_device)
        try:
            mask_idx2 = np.where(filter_pcd.point["positions"].cpu().numpy()[:, 2] < self.ceiling_height)[0]
            filter_pcd = _select_tensor_pointcloud_by_mask(filter_pcd, mask_idx2, self.pcd_device)
            # self.is_navigable(filter_pcd, self.floor_height, 0.8)
            self.scene_pcd = gpu_merge_pointcloud(self.current_pcd, self.scene_pcd).voxel_down_sample(self.pcd_resolution)
            scene_pos = self.scene_pcd.point["positions"].cpu().numpy()
            scene_mask_idx = np.where(scene_pos[:, 2] > self.floor_height - 0.3)[0]
            self.scene_pcd = _select_tensor_pointcloud_by_mask(self.scene_pcd, scene_mask_idx, self.pcd_device)
            scene_pos = self.scene_pcd.point["positions"].cpu().numpy()
            useful_mask_idx = np.where(scene_pos[:, 2] < self.ceiling_height)[0]
            self.useful_pcd = _select_tensor_pointcloud_by_mask(self.scene_pcd, useful_mask_idx, self.pcd_device)

            navigable_mask_idx = np.where(self.useful_pcd.point["positions"].cpu().numpy()[:, 2] < self.floor_height)[0]
            self.navigable_pcd = _select_tensor_pointcloud_by_mask(self.useful_pcd, navigable_mask_idx, self.pcd_device)
            # self.navigable_pcd = self.is_navigable(self.current_position, self.useful_pcd, self.floor_height, 0.8)
            obstacle_mask_idx = np.where(self.useful_pcd.point["positions"].cpu().numpy()[:, 2] > self.floor_height)[0]
            self.obstacle_pcd = _select_tensor_pointcloud_by_mask(self.useful_pcd, obstacle_mask_idx, self.pcd_device)
            self.trajectory_pcd = gpu_pointcloud_from_array(np.array(self.trajectory_position), np.zeros((len(self.trajectory_position), 3)), self.pcd_device)

            # classes, masks, confidences, visualization = self.object_percevior(self.current_rgb)
            classes, masks, confidences, visualization = self.object_percevior.perceive(self.current_rgb, confidence_threshold=0.5)
            self.segmentation = visualization[0]
            current_object_entities = self.get_object_entities(self.current_depth, classes, masks, confidences)
            self.object_entities = self.associate_object_entities(self.object_entities, current_object_entities)

            # if bool(self.obstacle_pcd.point) and bool(self.navigable_pcd.point):
            if self.args.FBE > 0:
                self.frontier_pcd = project_frontier(self.obstacle_pcd, self.navigable_pcd, self.floor_height + 0.2, self.grid_resolution)
                self.frontier_pcd = gpu_pointcloud_from_array(self.frontier_pcd, np.ones((self.frontier_pcd.shape[0], 3)) * np.array([[255, 0, 0]]), self.pcd_device)

            self.update_iterations += 1
        except Exception as e:
            print(e)

    def is_navigable(self, pcd, floor_height, max_height):
        """  
        判断一个点是否在其上方一定高度内没有其他点云。  
        
        :param point: 要检查的点  
        :param pcd: 整体的点云数据  
        :param floor_height: 地板的高度阈值  
        :param max_height: 可导航空间的最大高度  
        :return: 如果点上方没有点云则返回True，否则返回False  
        """
        # 如果点的高度大于floor_height，直接返回False
        self.project_block = {}
        positions = pcd.point["positions"].cpu().numpy()
        for i in range(len(positions)):
            block_x = math.ceil(positions[i][0] / 0.06)
            block_y = math.ceil(positions[i][1] / 0.06)
            if (block_x, block_y) not in self.project_block.keys():
                self.project_block[(block_x, block_y)] = []
            self.project_block[(block_x, block_y)].append(i)

        for key in self.project_block.keys():
            flag = 1
            for idx in self.project_block[key]:
                if positions[idx][2] > floor_height:
                    flag = 0
                    break
            if flag == 1:
                sel = _select_tensor_pointcloud_by_mask(pcd, np.array(self.project_block[key], dtype=np.int64), self.pcd_device)
                self.navigable_pcd = gpu_merge_pointcloud(self.navigable_pcd, sel)

    def get_block_map(self):
        self.block_map = {}
        self.block_navigation_map = {}
        if bool(self.object_entities):
            for object_entity in self.object_entities:
                average_position = object_entity['pcd'].get_center().cpu().numpy()
                block_x = math.ceil(average_position[0] / self.block_size)
                block_y = math.ceil(average_position[1] / self.block_size)
                if (block_x, block_y) not in self.block_map.keys():
                    self.block_map[(block_x, block_y)] = []
                self.block_map[(block_x, block_y)].append(object_entity)
        try:
            pos = self.navigable_pcd.point["positions"].cpu().numpy()
        except Exception:
            pos = None
        if pos is not None and pos.shape[0] > 0:
            for point in pos:
                block_x = math.ceil(point[0] / self.block_size)
                block_y = math.ceil(point[1] / self.block_size)
                if (block_x, block_y) not in self.block_navigation_map.keys():
                    self.block_navigation_map[(block_x, block_y)] = []
                self.block_navigation_map[(block_x, block_y)].append(point)

    def visualize_block_map(
        self,
        path=None,
    ):
        self.get_block_map()
        max_x = -99999
        max_y = -99999
        min_x = 99999
        min_y = 99999
        for key in self.block_map.keys():
            max_x = max(max_x, key[0])
            max_y = max(max_y, key[1])
            min_x = min(min_x, key[0])
            min_y = min(min_y, key[1])

        fig, ax = plt.subplots()
        for (x, y), items_list in self.block_map.items():
            for items in items_list:
                label = items['class']
                ax.text(x, y, label, ha='center', va='center')

        # 设置坐标轴的范围，以确保所有格子都可见
        min_x = min(min_x, -10)
        max_x = max(max_x, 10)
        min_y = min(min_y, -10)
        max_y = max(max_y, 10)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        # 显示网格
        ax.grid(True)
        if path is not None:
            plt.savefig(path)

    def get_view_pointcloud(self, rgb, depth, translation, rotation):
        current_position = self.translation_func(translation) - self.initial_position
        current_rotation = self.rotation_func(rotation)
        current_depth = preprocess_depth(depth)
        current_rgb = preprocess_image(rgb)
        camera_points, camera_colors = get_pointcloud_from_depth(current_rgb, current_depth, self.camera_intrinsic)
        world_points = translate_to_world(camera_points, current_position, current_rotation)
        current_pcd = gpu_pointcloud_from_array(world_points, camera_colors, self.pcd_device).voxel_down_sample(self.pcd_resolution)
        return current_pcd

    def get_object_entities(self, depth, classes, masks, confidences):
        entities = []
        for cls, mask, score in zip(classes, masks, confidences):
            if depth[mask > 0].min() < 1.0 and score < 0.5:
                continue
            exist_objects = np.unique([ent['class'] for ent in self.object_entities]).tolist()
            if cls not in exist_objects:
                exist_objects.append(cls)
            camera_points = get_pointcloud_from_depth_mask(depth, mask, self.camera_intrinsic)
            world_points = translate_to_world(camera_points, self.current_position, self.current_rotation)
            point_colors = np.array([d3_40_colors_rgb[exist_objects.index(cls) % 40]] * world_points.shape[0])
            if world_points.shape[0] < 10:
                continue
            object_pcd = gpu_pointcloud_from_array(world_points, point_colors, self.pcd_device).voxel_down_sample(self.pcd_resolution)
            object_pcd = gpu_cluster_filter(object_pcd)
            try:
                obj_pos_shape = object_pcd.point["positions"].cpu().numpy().shape[0]
            except Exception:
                obj_pos_shape = 0
            if obj_pos_shape < 10:
                continue
            entity = {'class': cls, 'pcd': object_pcd, 'confidence': score}
            entities.append(entity)
        return entities

    def associate_object_entities(self, ref_entities, eval_entities):
        for entity in eval_entities:
            if len(ref_entities) == 0:
                ref_entities.append(entity)
                continue
            overlap_score = []
            eval_pcd = entity['pcd']
            for ref_entity in ref_entities:
                try:
                    eval_count = eval_pcd.point["positions"].cpu().numpy().shape[0]
                except Exception:
                    eval_count = 0
                if eval_count == 0:
                    break
                cdist = pointcloud_distance(eval_pcd, ref_entity['pcd'])
                overlap_condition = (cdist < 0.1)
                nonoverlap_mask = (~overlap_condition.cpu().numpy()).astype(bool)
                # select remaining points
                remaining_idx = np.where(nonoverlap_mask)[0]
                eval_pcd = _select_tensor_pointcloud_by_mask(eval_pcd, remaining_idx, self.pcd_device)
                overlap_score.append((overlap_condition.sum() / (overlap_condition.shape[0] + 1e-6)).cpu().numpy())
            max_overlap_score = np.max(overlap_score)
            arg_overlap_index = np.argmax(overlap_score)
            if max_overlap_score < 0.25:
                entity['pcd'] = eval_pcd
                ref_entities.append(entity)
            else:
                argmax_entity = ref_entities[arg_overlap_index]
                argmax_entity['pcd'] = gpu_merge_pointcloud(argmax_entity['pcd'], eval_pcd)
                ref_entities[arg_overlap_index] = argmax_entity
        return ref_entities

    def get_appeared_objects(self):
        return [entity['class'] for entity in self.object_entities]

    def get_appeared_rooms(self):
        return [entity['class'] for entity in self.room_entities]

    def save_pointcloud_debug(self, path="./"):
        save_pcd = o3d.geometry.PointCloud()
        try:
            assert self.useful_pcd.point["positions"].cpu().numpy().shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(self.useful_pcd.point["positions"].cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(self.useful_pcd.point["colors"].cpu().numpy())
            o3d.io.write_point_cloud(path + "scene.ply", save_pcd)
        except:
            pass
        try:
            assert self.navigable_pcd.point["positions"].cpu().numpy().shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(self.navigable_pcd.point["positions"].cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(self.navigable_pcd.point["colors"].cpu().numpy())
            o3d.io.write_point_cloud(path + "navigable.ply", save_pcd)
        except:
            pass
        object_pcd = o3d.geometry.PointCloud()
        for entity in self.object_entities:
            points = entity['pcd'].point["positions"].cpu().numpy()
            colors = entity['pcd'].point["colors"].cpu().numpy()
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points)
            new_pcd.colors = o3d.utility.Vector3dVector(colors)
            object_pcd = object_pcd + new_pcd
        if len(object_pcd.points) > 0:
            o3d.io.write_point_cloud(path + "object.ply", object_pcd)
