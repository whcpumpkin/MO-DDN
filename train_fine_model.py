import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.sim_utilities import get_all_object_ids
import cv2
import os
import random
import json
from habitat.core.env import DDNPlusEnv, RLEnv, Fine_Grained_Env
from vector_env import VectorEnv
from tqdm import tqdm
from PIL import Image
import time
import numpy as np
from habitat.utils.common import compute_instance_bounding_boxes, draw_bounding_boxes_and_labels
from habitat.tasks.nav.shortest_path_follower import DDNPlusShortestPathFollower
from models_code.DDNP_agent import AgentModel, InstructionEncoder
from utils.args import parse_arguments
import torch
from copy import deepcopy
import torch.optim as optim
from attributebook.attribute_model import AttributeModel
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from habitat.utils.common import draw_single_bounding_boxes_and_labels
import torch.multiprocessing as multiprocessing
from utils.my_dataloader import Traj_Dataset, traj_collate_fn
from torch.utils.data import DataLoader
import torch.nn as nn

multiprocessing.set_start_method('spawn', force=True)


def fine_training(args, time_now):
    args.time_now = time_now
    with open("dataset/instruction_feature_{}.json".format(args.ins_model_name), "r") as f:
        instruction_feature = json.load(f)
    writer = SummaryWriter(log_dir='{}/{}/{}'.format(args.save_dir, args.save_name, time_now))
    train_traj_dataset = Traj_Dataset(args, mode='train')
    train_traj_dataloader = DataLoader(train_traj_dataset, batch_size=args.il_batch_size, shuffle=True, num_workers=args.workers, collate_fn=traj_collate_fn)

    val_traj_dataset = Traj_Dataset(args, mode='val')
    val_traj_dataloader = DataLoader(val_traj_dataset, batch_size=args.il_batch_size, shuffle=False, num_workers=args.workers, collate_fn=traj_collate_fn)
    action_mapping = {0: 'move_forward', 1: 'turn_left', 2: 'turn_right', 3: 'look_up', 4: 'look_down', 5: 'find', 6: 'leave'}
    f = open("{}/{}/{}/{}.txt".format(args.save_dir, args.save_name, time_now, "eval_result.txt"), "a")

    @torch.no_grad()
    def eval_model(agent, val_traj_dataloader, args, epoch):
        agent.eval()
        accuracy_num = 0
        all_num = 0
        action_dist_dict = {}
        f = open("{}/{}/{}/{}.txt".format(args.save_dir, args.save_name, time_now, "eval_result.txt"), "a")
        for idx, batch in tqdm(enumerate(val_traj_dataloader), total=len(val_traj_dataloader), desc='Val Epoch {}/{}'.format(epoch + 1, args.epoch)):
            batch_size = 1
            seq_len = batch[0][0].shape[0]
            rgb = batch[0]
            rgb = torch.stack(rgb, dim=0).to(args.device).squeeze(0)

            depth = batch[1]
            depth = torch.stack(depth, dim=0).to(args.device).squeeze(0)

            action = batch[2]
            action = torch.tensor(action).squeeze(0)

            task_instructions = batch[3]
            instruction_features = [instruction_feature[task_instruction] for task_instruction in task_instructions]
            instruction_features = torch.tensor(instruction_features).to(args.device).squeeze(0)

            batch_size = rgb.shape[0]

            if args.use_gps_compass:
                gps_compass = batch[4]
                gps_compass = torch.tensor(gps_compass).to(args.device).squeeze(0)
            # time_start = time.time()
            action_dist = agent.forward_il(rgb, depth, instruction_features, action, gps_compass)
            # print("Forward Time: {}".format(time.time() - time_start))
            acton_gt = action.reshape(seq_len, 1).to(args.device)
            action_logits = action_dist.logits.reshape(seq_len, -1)

            accuracy_num += (action_logits.argmax(dim=1) == acton_gt.squeeze(1)).sum()
            all_num += seq_len
            # time_start = time.time()
            for action_idx, action_argmax in enumerate(action_logits.argmax(dim=1).tolist()):
                if action_argmax not in action_dist_dict:
                    action_dist_dict[action_argmax] = 1
                else:
                    action_dist_dict[action_argmax] += 1
            # print("Dict Time: {}".format(time.time() - time_start))
        accuracy = accuracy_num / all_num
        writer.add_scalar('val/il_accuracy', accuracy, epoch)
        print("Epoch: {}, Val Accuracy: {}".format(epoch + 1, accuracy))
        f.write("-------------------------------------------------------------\n")
        f.write("Model: {}\n".format(epoch))
        for key in action_dist_dict.keys():
            print("Action: {}, Count: {}, Percentage: {}".format(action_mapping[key], action_dist_dict[key], action_dist_dict[key] / all_num))
            f.write("Action: {}, Count: {}, Percentage: {}\n".format(action_mapping[key], action_dist_dict[key], action_dist_dict[key] / all_num))

        print("Model: {}, Val Accuracy: {}".format(epoch, accuracy_num / all_num))
        f.write("Model: {}, Val Accuracy: {}\n".format(epoch, accuracy_num / all_num))

    task_name = "ddnplus"
    habitat_config = habitat.get_config("benchmark/nav/{}/hssd-200_{}_hssd-hab_with_semantic.yaml".format(task_name, task_name))
    agent = AgentModel(image_size=habitat_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.width, args=args, attribute_feature_dim=768)
    agent.optimizer = optim.Adam(agent.parameters(), lr=1e-5, eps=1e-5)
    agent.attribute_model.load_model(args.attribute_model_path)
    agent.to(args.device)
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0]).to(args.device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, reduction='none')
    # eval_model_in_env(agent, args, writer, f, 0)
    # eval_model(agent, val_traj_dataloader, args, 0)

    if os.path.exists('{}/{}/{}'.format(args.save_dir, args.save_name, time_now)) is False:
        os.makedirs('{}/{}/{}'.format(args.save_dir, args.save_name, time_now))
    agent.save_model(os.path.join('{}/{}/{}'.format(args.save_dir, args.save_name, time_now), "fine_model_{}.pth".format(0)))

    with open(os.path.join('{}/{}/{}'.format(args.save_dir, args.save_name, time_now), "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    for epoch in range(args.epoch):
        agent.train()
        accuracy_num = 0
        all_num = 0
        all_loss = 0

        step_num = 0
        step_loss = 0
        step_accuracy = 0
        agent.optimizer.zero_grad()
        for idx, batch in tqdm(enumerate(train_traj_dataloader), total=len(train_traj_dataloader), desc='Train Epoch {}/{}'.format(epoch + 1, args.epoch)):
            batch_size = 1
            seq_len = batch[0][0].shape[0]
            rgb = batch[0]
            rgb = torch.stack(rgb, dim=0).to(args.device).squeeze(0)

            depth = batch[1]
            depth = torch.stack(depth, dim=0).to(args.device).squeeze(0)

            action = batch[2]
            action = torch.tensor(action).squeeze(0)

            task_instructions = batch[3]
            instruction_features = [instruction_feature[task_instruction] for task_instruction in task_instructions]
            instruction_features = torch.tensor(instruction_features).to(args.device).squeeze(0)

            batch_size = rgb.shape[0]

            if args.use_gps_compass:
                gps_compass = batch[4]
                gps_compass = torch.tensor(gps_compass).to(args.device).squeeze(0)
            # time_start = time.time()
            action_dist = agent.forward_il(rgb, depth, instruction_features, action, gps_compass)
            acton_gt = action.reshape(seq_len, 1).to(args.device)
            action_logits = action_dist.logits.reshape(seq_len, -1)
            loss = loss_fn(action_logits, acton_gt.squeeze(1))
            loss = loss.sum() / seq_len / args.grad_acc_steps
            all_loss += loss.item() * args.grad_acc_steps
            step_loss += loss.item() * args.grad_acc_steps
            loss.backward()
            if (idx + 1) % args.grad_acc_steps == 0:
                agent.optimizer.step()
                agent.optimizer.zero_grad()

            accuracy_num += (action_logits.argmax(dim=1) == acton_gt.squeeze(1)).sum()
            step_accuracy += (action_logits.argmax(dim=1) == acton_gt.squeeze(1)).sum()
            all_num += seq_len
            step_num += seq_len
            if (idx + 1) % 100 == 0:
                writer.add_scalar('train/step_il_loss', step_loss, epoch * len(train_traj_dataloader) + idx)
                writer.add_scalar('train/step_il_accuracy', step_accuracy / step_num, epoch * len(train_traj_dataloader) + idx)
                print("Epoch: {}, Step: {}, Loss: {}, Accuracy: {}".format(epoch + 1, idx + 1, step_loss / step_num, step_accuracy / step_num))
                step_num = 0
                step_loss = 0
                step_accuracy = 0
        accuracy = accuracy_num / all_num
        writer.add_scalar('train/il_accuracy', accuracy, epoch + 1)
        writer.add_scalar('train/il_loss', all_loss / len(train_traj_dataloader), epoch + 1)
        eval_model(agent, val_traj_dataloader, args, epoch + 1)
        # eval_model_in_env(agent, args, writer, f, epoch + 1)
        agent.save_model(os.path.join('{}/{}/{}'.format(args.save_dir, args.save_name, time_now), "fine_model_{}.pth".format(epoch + 1)))


if __name__ == "__main__":
    args = parse_arguments()
    time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    fine_training(args, time_now)
