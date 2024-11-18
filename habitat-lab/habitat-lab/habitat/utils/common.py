#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from os import makedirs
from os import path as osp
from typing import Any, Dict, List, Tuple
import numpy as np
from habitat.core.logging import logger
from scipy.ndimage import label, generate_binary_structure
import cv2
import itertools
import math
import json
from copy import deepcopy


def check_make_dir(directory_path: str) -> bool:
    """
    Check for the existence of the provided directory_path and create it if not found.
    """
    # if output directory doesn't exist, create it
    if not osp.exists(directory_path):
        try:
            makedirs(directory_path)
        except OSError:
            logger.error(f"check_make_dir: Failed to create the specified directory_path: {directory_path}")
            return False
        logger.info(f"check_make_dir: directory_path did not exist and was created: {directory_path}")
    return True


def cull_string_list_by_substrings(
    full_list: List[str],
    included_substrings: List[str],
    excluded_substrings: List[str],
) -> List[str]:
    """
    Cull a list of strings to the subset of strings containing any of the "included_substrings" and none of the "excluded_substrings".
    Returns the culled list, does not modify the input list.
    """
    culled_list: List[str] = []
    for string in full_list:
        excluded = False
        for excluded_substring in excluded_substrings:
            if excluded_substring in string:
                excluded = True
                break
        if not excluded:
            for included_substring in included_substrings:
                if included_substring in string:
                    culled_list.append(string)
                    break
    return culled_list


def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    r"""Flattens nested dict.

    Source: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys

    :param d: Nested dict.
    :param parent_key: Parameter to set parent dict key.
    :param sep: Nested keys separator.
    :return: Flattened dict.
    """
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def compute_instance_bounding_boxes(semantic_map, depth_map, min_area=1, dilation_kernel_size=3, depth_threshold=1):
    bounding_boxes = {}

    # 获取唯一的对象ID
    object_ids = np.unique(semantic_map)

    for object_id in object_ids:
        # 忽略背景或无效值
        if object_id == 0 or object_id == 519:
            continue

        # 创建针对当前object_id的二值图像
        binary_map = np.where(semantic_map == object_id, 1, 0).astype(np.uint8)

        # 使用OpenCV计算连通组件，直接使用未经膨胀的二值图
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)

        for label in range(1, num_labels):  # label=0 是背景
            # 连通区域的面积
            area = stats[label, cv2.CC_STAT_AREA]

            # 如果连通区域过小，则忽略
            if area < min_area:
                continue

            # 仅考虑该标签的深度值
            depth_values = depth_map[labels == label]

            # 检查深度范围是否超过阈值
            if depth_values.min() > depth_threshold:
                continue  # 如果超过，忽略这个连通组件

            # 边界框：左上角的x和y，宽度，高度
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]

            # 将边界框转换为(min_row, min_col, max_row, max_col)格式
            bbox = (y, x, y + h, x + w)

            # 将边界框添加到结果中，使用object_id和实例编号作为键
            bounding_boxes.setdefault(object_id, []).append(bbox)

    return bounding_boxes


def draw_bounding_boxes_and_labels(rgb_image, bounding_boxes, labels):
    """
    Draw bounding boxes and labels on the rgb image.
    :param rgb_image: RGB image as a numpy array.
    :param bounding_boxes: Dictionary of bounding boxes for each object_id.
    :param labels: Dictionary of labels for each object_id.
    """
    for object_id, boxes in bounding_boxes.items():
        label = labels[object_id]  # Use the object_id as label if not found in labels dictionary
        for box in boxes:
            start_point = (box[1], box[0])  # (min_y, min_x)
            end_point = (box[3], box[2])  # (max_y, max_x)
            color = (255, 0, 0)  # Blue color in BGR
            thickness = 2
            # Draw rectangle around each object instance
            cv2.rectangle(rgb_image, start_point, end_point, color, thickness)
            # Put the label of the object
            cv2.putText(rgb_image, label, (box[1], box[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return rgb_image


def draw_single_bounding_boxes_and_labels(rgb_image, box, target_name):
    """
    Draw bounding boxes and labels on the rgb image.
    :param rgb_image: RGB image as a numpy array.
    :param bounding_boxes: Dictionary of bounding boxes for each object_id.
    :param labels: Dictionary of labels for each object_id.
    """

    start_point = (box[0], box[1])  # (min_x, min_y)
    start_point = np.floor(start_point).astype(np.uint32)
    end_point = (box[2], box[3])  # (max_x, max_y)
    end_point = np.ceil(end_point).astype(np.uint32)
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 2
    # Draw rectangle around each object instance
    rgb_image = np.ascontiguousarray(rgb_image)
    cv2.rectangle(rgb_image, start_point, end_point, color, thickness)
    put_position = np.mean(np.array([start_point, end_point]), axis=0).astype(np.int32)
    # Put the label of the object
    cv2.putText(rgb_image, target_name, put_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return rgb_image


def bbox_iou(bbox, bboxes):
    """
    Calculate IoU of `bbox` with each bbox in `bboxes`.
    
    :param bbox: A single bounding box, format: [x_min, y_min, x_max, y_max]
    :param bboxes: A dict of bounding boxes, key is object_id, value is a list of bounding boxes
    :return: A list of IoU values
    """
    ious = {}
    for key in bboxes:
        ious[key] = []
        for box in bboxes[key]:
            # Calculate intersection coordinates
            x_min_inter = max(bbox[0], box[0])
            y_min_inter = max(bbox[1], box[1])
            x_max_inter = min(bbox[2], box[2])
            y_max_inter = min(bbox[3], box[3])

            # Calculate intersection area
            inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

            # Calculate union area
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            union_area = bbox_area + box_area - inter_area

            # Calculate IoU
            iou = inter_area / union_area if union_area > 0 else 0
            ious[key].append(iou)

    return ious


def calculate_distance(p1, p2, sim):
    """计算两点之间的欧式距离"""
    if p1 is None or p2 is None:
        return 9999999
    # output = sim.get_straight_shortest_path_points(p1, p2)
    # each_etep_distance = []
    # for i in range(len(output) - 1):
    #     each_etep_distance.append(np.linalg.norm(np.array(output[i]) - np.array(output[i + 1])))
    return sim.geodesic_distance(p1, p2)


def find_closest_point(start_point, points, sim):
    """找到距离起点最近的点"""
    min_distance = 9999999
    closest_point = None
    for point in points:
        distance = calculate_distance(start_point, point[0], sim)
        if distance < min_distance:
            min_distance = distance
            closest_point = point[0]
    if min_distance == 9999999:
        # print("Warning: no navigable point found bewteen {} and {}".format(start_point, points))
        pass
    return closest_point, min_distance


def find_shortest_path(start_point, categories, sim):
    """找到从起点出发访问所有物品种类的最短路径"""
    min_path_distance = float('inf')
    min_path = None
    # 生成所有可能的种类访问序列
    for sequence in itertools.permutations(categories.keys()):
        current_point = start_point
        total_distance = 0
        path = []
        # 对于每个序列，计算依次访问每个种类的最短路径
        for category in sequence:
            closest_point, distance = find_closest_point(current_point, categories[category], sim)
            total_distance += distance
            current_point = closest_point
            path.append((category, closest_point))
        # 更新最短路径
        if total_distance < min_path_distance:
            min_path_distance = total_distance
            min_path = path
    return min_path, min_path_distance


def sample_object_navigable_point(_sim, _task, agent_position, distance=None):
    with open(_sim._current_scene, 'r') as f:
        _scene_config = json.load(f)
    with open(_sim.habitat_config.scene_object_string_to_name, 'r') as f:
        _scene_object_string_to_name = json.load(f)
    if distance is None:
        distance = _task._config.measurements.ddnplus_basic_success['ddnplus_success_distance']
    _object_to_positions = {}
    for object_string in _scene_config["object_instances"]:
        object_name = str(_scene_object_string_to_name[object_string['template_name'].split("_part")[0]])
        if object_name == "nan":
            continue
        translation = object_string['translation']
        ori_height = deepcopy(translation[1])
        translation[1] = agent_position[1]
        if _sim.pathfinder.is_navigable(translation):
            if object_name not in _object_to_positions.keys():
                _object_to_positions[object_name] = []
            _object_to_positions[object_name].append([np.array(translation).tolist(), ori_height])
        else:
            flag = 0
            for i in range(20):
                near_position = _sim.pathfinder.get_random_navigable_point_near(translation, distance / 10 * (i + 1), 5000)
                if not np.isnan(near_position).any():
                    if object_name not in _object_to_positions.keys():
                        _object_to_positions[object_name] = []
                    _object_to_positions[object_name].append([np.array(near_position).tolist(), ori_height])
                    flag = 1
                    break
            if flag == 0:
                print("Object {} is not navigable, near position is {}".format(object_name, near_position))
    return _object_to_positions
