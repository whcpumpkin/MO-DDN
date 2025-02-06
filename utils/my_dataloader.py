import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import json
import random
import os
from PIL import Image
import numpy as np

import time


def my_collate_fn(batch):
    ins_list = []
    ins_attributes_list = []
    object_attributes_list = []
    for ins, ins_attributes, object_attributes in batch:
        ins_list.append(ins)
        ins_attributes_list.append(ins_attributes)
        object_attributes_list.append(object_attributes)
    return ins_list, ins_attributes_list, object_attributes_list


def ddnp_collate_fn(batch):
    ins_list = []
    basic_object_list = []
    preference_object_list = []
    basic_demand_instruction_list = []
    preference_demand_instruction_list = []
    for ins, basic_object, preference_object, basic_demand_instruction, preference_demand_instruction in batch:
        ins_list.append(ins)
        basic_object_list.append(basic_object)
        preference_object_list.append(preference_object)
        basic_demand_instruction_list.append(basic_demand_instruction)
        preference_demand_instruction_list.append(preference_demand_instruction)
    return ins_list, basic_object_list, preference_object_list, basic_demand_instruction_list, preference_demand_instruction_list


def traj_collate_fn(batch):
    rgb_list = []
    depth_list = []
    metadata_action_list = []
    task_instruction_list = []
    # mask_list = []
    gps_compass_list = []
    for rgb, depth, metadata_action, task_instruction, gps_compass in batch:
        rgb_list.append(rgb)
        depth_list.append(depth)
        metadata_action_list.append(metadata_action)
        task_instruction_list.append(task_instruction)
        # mask_list.append(mask)
        gps_compass_list.append(gps_compass)
    return rgb_list, depth_list, metadata_action_list, task_instruction_list, gps_compass_list


class Attribute_Dataset(Dataset):

    def __init__(self, args, mode="train"):
        super(Attribute_Dataset, self).__init__()
        self.args = args
        with open(args.path_to_attribute_ins_file, 'r') as f:
            self.ins_attributes = json.load(f)
        with open(args.path_to_attribute_object_file, 'r') as f:
            self.object_attributes = json.load(f)
        self.ins_list = list(self.ins_attributes.keys())

    def __len__(self):
        return len(self.ins_list)

    def __getitem__(self, idx):
        ins = self.ins_list[idx]
        ins_attributes = self.ins_attributes[ins][0]
        if len(ins_attributes) < self.args.attribute_k1:
            # repeat the  attribute to make it have length of attribute_k1
            ins_attributes = ins_attributes + ins_attributes[:self.args.attribute_k1 - len(ins_attributes)]
        elif len(ins_attributes) > self.args.attribute_k1:
            # randomly select attribute_k1 attributes from the ins_attributes
            ins_attributes = ins_attributes[:self.args.attribute_k1]

        solutions = self.ins_attributes[ins][1]
        object_attributes = []
        for solution in solutions:
            for object_id in solution:
                obj_list = self.object_attributes[object_id]
                if len(obj_list) < self.args.attribute_k2:
                    # repeat the  attribute to make it have length of attribute_k2
                    obj_list = obj_list + obj_list[:self.args.attribute_k2 - len(obj_list)]
                elif len(obj_list) > self.args.attribute_k2:
                    obj_list = obj_list[:self.args.attribute_k2]
                object_attributes.append([object_id, obj_list])

        if len(object_attributes) < self.args.attribute_k3:
            # repeat the attribute to make it have length of attribute_k3
            object_attributes = object_attributes + object_attributes[:self.args.attribute_k3 - len(object_attributes)]

        elif len(object_attributes) > self.args.attribute_k3:
            object_attributes = object_attributes[:self.args.attribute_k3]
        if len(object_attributes) != self.args.attribute_k3:
            print("Error: object_attributes_k3 length is not equal to attribute_k3")
            print("the length of object_attributes_k3 is: {}".format(len(object_attributes)))
        return ins, ins_attributes, object_attributes


class DDNP_Dataset(Dataset):

    def __init__(self, args, mode="train"):
        super(DDNP_Dataset, self).__init__()
        self.args = args
        with open(args.task_file, 'r') as f:
            self.task_list = json.load(f)

    def __len__(self):
        return len(self.task_list)

    def __getitem__(self, idx):
        ins = self.task_list[idx]['task_instruction']
        basic_object = self.task_list[idx]['basic_solution']
        preference_object = self.task_list[idx]['preferred_solution']
        basic_demand_instruction = self.task_list[idx]['basic_demand_instruction']
        preference_demand_instruction = self.task_list[idx]['preferred_demand_instruction']
        return ins, basic_object, preference_object, basic_demand_instruction, preference_demand_instruction


class Traj_Dataset(Dataset):

    def __init__(self, args, mode="train"):
        super(Traj_Dataset, self).__init__()
        self.args = args
        self.mode = mode
        if mode == "train":
            with open(args.path_to_train_traj_file, 'r') as f:
                self.traj_list = json.load(f)
        elif mode == "val":
            with open(args.path_to_val_traj_file, 'r') as f:
                self.traj_list = json.load(f)

    def __len__(self):
        if self.mode == "train":
            return len(self.traj_list)
        elif self.mode == "val":
            return len(self.traj_list) // 5

    def __getitem__(self, idx):
        traj_path = self.traj_list[idx]
        rgb_path = os.path.join(traj_path, 'rgb')
        depth_path = os.path.join(traj_path, 'depth')
        metadata_path = os.path.join(traj_path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        rgb_list = []
        depth_list = []
        mask_list = []
        gps_compass_list = []
        w, h = 0, 0
        # metadata['action_list'] = metadata['action_list'][1:]
        for i in range(len(metadata['action_list'])):
            current_rgb = Image.open(os.path.join(rgb_path, str(i) + '.png')).convert('RGB')
            current_rgb = torch.tensor(np.array(current_rgb))
            rgb_list.append(current_rgb)
            current_depth = torch.tensor(np.load(os.path.join(depth_path, str(i) + '.npy')))
            depth_list.append(current_depth)
            w, h, c = current_rgb.shape
            # time_start = time.time()
            if self.args.use_gps_compass:
                gps_compass_list.append(metadata['gps'][i] + metadata['compass'][i])
            else:
                gps_compass_list.append([0, 0, 0])
            # print("load gps {}th frame cost {}s".format(i, time.time() - time_start))
            mask_list.append(1)

        # mask_list = mask_list + [0] * (self.args.max_seq_len - len(mask_list))
        metadata_action_list = metadata['action_list']
        depth_list = torch.stack(depth_list, dim=0)
        rgb_list = torch.stack(rgb_list, dim=0)
        metadata_action_list = [x - 1 for x in metadata_action_list]

        return rgb_list, depth_list, metadata_action_list, metadata['current_episode']['task_instruction'], gps_compass_list
