import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.sim_utilities import get_all_object_ids
import os
import random
import json
from habitat.core.env import DDNPlusEnv, RLEnv, Fine_Grained_Env
from tqdm import tqdm
from PIL import Image
import time
import numpy as np
from habitat.utils.common import compute_instance_bounding_boxes, draw_bounding_boxes_and_labels
# from map import Mapper, habitat_camera_intrinsic
from habitat.tasks.nav.shortest_path_follower import DDNPlusShortestPathFollower
from utils.args import parse_arguments
import torch
from habitat.utils.common import draw_single_bounding_boxes_and_labels
import torch.multiprocessing as multiprocessing
from utils.common import ArgsObject

multiprocessing.set_start_method('spawn', force=True)


def find_bounding_box(semantic_map, target_label):
    semantic_map = semantic_map[:, :, 0]
    mask = (semantic_map == target_label)
    if not mask.any():
        return None
    rows, cols = np.where(mask)
    x_min, y_min = cols.min(), rows.min()
    x_max, y_max = cols.max(), rows.max()
    return (x_min, y_min, x_max, y_max)


def local_data_collection(args, time_now, rank=0):

    device = "cuda"
    with open("data/datasets/ddnplus/hssd-hab_v0.2.5/train/train.json", "r") as f:
        object_name_to_idx = json.load(f)
    # roberta_text_encoder = InstructionEncoder()
    # roberta_text_encoder.to(device)
    # with open("dataset/instruction_feature_{}.json".format(args.ins_model_name), "r") as f:
    #     instruction_feature = json.load(f)
    save_path = '{}/{}/{}/{}'.format(args.save_dir, args.save_name, time_now, rank)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    def save_observation(observations, metadata, step, action, image):
        rgb = observations['rgb']
        depth = observations['depth']
        semantic = observations['semantic']
        gps = observations['gps']
        compsss = observations['ddnpluscompass']
        # rgb = Image.fromarray(rgb)
        # rgb.save('traj_save/{}/{}/rgb/{}.png'.format(args.save_name, time_now, step))
        # np.save('traj_save/{}/{}/depth/{}.npy'.format(args.save_name, time_now, step), depth.astype(np.float16))

        # # semantic save as npy
        # semantic_path = 'traj_save/{}/{}/semantic/{}.npy'.format(args.save_name, time_now, step)
        # np.save(semantic_path, semantic.astype(np.uint16))
        image['rgb_list'].append(rgb)
        image['depth_list'].append(depth.astype(np.float16))
        image['semantic_list'].append(semantic.astype(np.uint16))

        metadata['action_list'].append(action)
        metadata['gps'].append(gps.tolist())
        metadata['compass'].append(compsss.tolist())

    def save_metadata(metadata, save_name, time_now, j, image, bounding_box, save_path):
        with open(os.path.join(save_path, str(j), 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        for i in range(len(image['rgb_list'])):
            rgb = Image.fromarray(image['rgb_list'][i])
            rgb.save(os.path.join(save_path, str(j), 'rgb', '{}.png'.format(i)))
            np.save(os.path.join(save_path, str(j), 'depth', '{}.npy'.format(i)), image['depth_list'][i])
            np.save(os.path.join(save_path, str(j), 'semantic', '{}.npy'.format(i)), image['semantic_list'][i])
        bb_rgb_image = draw_single_bounding_boxes_and_labels(image['rgb_list'][-1], bounding_box, metadata['target_object_name'])
        bb_rgb_image = Image.fromarray(bb_rgb_image)
        bb_rgb_image.save(os.path.join(save_path, str(j), 'bb_rgb.png'))

    # instruction_feature = {k: torch.tensor(v).float().to(device) for k, v in instruction_feature.items()}
    object_name_to_idx = object_name_to_idx["category_to_task_category_id"]
    object_idx_to_name = {v: k for k, v in object_name_to_idx.items()}

    task_name = "ddnplus"
    habitat_config = habitat.get_config("benchmark/nav/{}/hssd-200_{}_hssd-hab_with_semantic_figure.yaml".format(task_name, task_name), overrides=["habitat.dataset.split={}".format(args.task_mode)])
    envs = Fine_Grained_Env(config=habitat_config, block_size=2, object_name_to_idx=object_name_to_idx, args=args)
    # HabitatSimActions.extend_action_space("find")
    # HabitatSimActions.extend_action_space("leave")
    action_mapping = {0: HabitatSimActions.find, 1: HabitatSimActions.move_forward, 2: HabitatSimActions.turn_left, 3: HabitatSimActions.turn_right}
    greedy_follower = DDNPlusShortestPathFollower(envs.sim, 1, False)
    j = 0
    pbar = tqdm(total=args.epoch, desc='Epoch {}/{} from {}'.format(j + 1, args.epoch, rank))
    f = open(os.path.join("{}/{}/{}/{}.txt".format(args.save_dir, args.save_name, time_now, rank)), 'w')
    while j < args.epoch:
        try:
            observations = envs.reset(True)

            metadata = {}
            image = {}
            metadata['target_object_name'] = envs.target_object_name
            metadata['target_object_position'] = envs.target_object_position
            metadata['current_episode'] = {}
            metadata['current_episode']['episode_id'] = envs.current_episode.episode_id
            metadata['current_episode']['scene_id'] = envs.current_episode.scene_id
            metadata['current_episode']['basic_demand_instruction'] = envs.current_episode.basic_demand_instruction
            metadata['current_episode']['basic_solution'] = envs.current_episode.basic_solution
            metadata['current_episode']['preferred_demand_instruction'] = envs.current_episode.preferred_demand_instruction
            metadata['current_episode']['preferred_solution'] = envs.current_episode.preferred_solution
            metadata['current_episode']['task_instruction'] = envs.current_episode.task_instruction
            metadata['agent_position'] = envs._sim.get_agent_state().position.tolist()

            f.write(metadata['current_episode']['scene_id'])
            f.write('\n')
            f.write(metadata['current_episode']['task_instruction'])
            f.write('\n--------------------------\n\n')
            rotation = envs._sim.get_agent_state().rotation
            metadata['agent_rotation'] = [rotation.w, rotation.x, rotation.y, rotation.z]
            # with open('traj_save/{}/{}/{}/metadata.json'.format(args.save_name, time_now, j), 'w') as f:
            #     json.dump(metadata, f)
            metadata['action_list'] = []
            metadata['gps'] = []
            metadata['compass'] = []
            image['rgb_list'] = []
            image['depth_list'] = []
            image['semantic_list'] = []

            image['rgb_list'].append(observations['rgb'])
            image['depth_list'].append(observations['depth'].astype(np.float16))
            image['semantic_list'].append(observations['semantic'].astype(np.uint16))
            metadata['gps'].append(observations['gps'].tolist())
            metadata['compass'].append(observations['ddnpluscompass'].tolist())

            target_object_position = envs.target_object_position
            target_object_height = envs.target_object_height
            action_greedy = 1
            step = 0
            while action_greedy > 0:
                choose_bbox = (100, 100, 150, 150)
                action_greedy = greedy_follower.get_next_action(target_object_position)

                action = {"action": action_mapping[action_greedy], "action_args": {"bbox": choose_bbox, "semantic_map": observations['semantic'], "depth_map": observations['depth']}}

                if action_greedy == 0:
                    agent_position = envs._sim.get_agent_state().position.tolist()
                    look_flag = 0
                    if target_object_height > 1.5:
                        action = {"action": HabitatSimActions.look_up, "action_args": {"bbox": choose_bbox, "semantic_map": observations['semantic'], "depth_map": observations['depth']}}
                        observations, reward, done, info = envs.step(action)
                        save_observation(observations, metadata, step, action['action'], image)
                        step += 1
                    elif target_object_height < 0.75:
                        action = {"action": HabitatSimActions.look_down, "action_args": {"bbox": choose_bbox, "semantic_map": observations['semantic'], "depth_map": observations['depth']}}
                        observations, reward, done, info = envs.step(action)
                        save_observation(observations, metadata, step, action['action'], image)
                        step += 1

                    visible = envs.object_name_to_idx[envs.target_object_name] in observations["semantic"]
                    turn_right = random.random() < 0.5
                    num_of_turn = 0
                    while visible is False:
                        action_move = HabitatSimActions.turn_right if turn_right else HabitatSimActions.turn_left
                        action = {"action": action_move, "action_args": {"bbox": choose_bbox, "semantic_map": observations['semantic'], "depth_map": observations['depth']}}
                        observations, reward, done, info = envs.step(action)
                        num_of_turn += 1
                        save_observation(observations, metadata, step, action['action'], image)
                        step += 1
                        visible = envs.object_name_to_idx[envs.target_object_name] in observations["semantic"]
                        if num_of_turn > 12:
                            break
                    action = {"action": action_mapping[action_greedy], "action_args": {"bbox": choose_bbox, "semantic_map": observations['semantic'], "depth_map": observations['depth']}}
                    bounding_box = find_bounding_box(observations['semantic'], envs.object_name_to_idx[envs.target_object_name])
                observations, reward, done, info = envs.step(action)
                # print("agent position: ", envs._sim.get_agent_state().position.tolist())
                save_observation(observations, metadata, step, action['action'], image)
                if done:
                    if 'find_success' in info:
                        if info['find_success'] > 0:
                            if os.path.exists(os.path.join(save_path, str(j))) is False:
                                os.makedirs(os.path.join(save_path, str(j)))
                                os.makedirs(os.path.join(save_path, str(j), 'rgb'))
                                os.makedirs(os.path.join(save_path, str(j), 'depth'))
                                os.makedirs(os.path.join(save_path, str(j), 'semantic'))

                            save_metadata(metadata, args.save_name, time_now, j, image, bounding_box, save_path)
                            j += 1
                            pbar.update(1)
                        else:
                            # print("Find failed")
                            pass
                    break
        except Exception as e:
            print(e)
            f.write("error: ")
            f.write(str(e) + '\n')


def full_data_collection(args, time_now, rank=0):
    device = "cuda"
    with open("data/datasets/ddnplus/hssd-hab_v0.2.5/train/train.json", "r") as f:
        object_name_to_idx = json.load(f)
    # roberta_text_encoder = InstructionEncoder()
    # roberta_text_encoder.to(device)
    # with open("dataset/instruction_feature_{}.json".format(args.ins_model_name), "r") as f:
    #     instruction_feature = json.load(f)
    save_path = '{}/{}/{}/{}'.format(args.save_dir, args.save_name, time_now, rank)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    def save_observation(observations, metadata, step, action, image):
        rgb = observations['rgb']
        depth = observations['depth']
        semantic = observations['semantic']
        gps = observations['gps']
        compsss = observations['ddnpluscompass']
        # rgb = Image.fromarray(rgb)
        # rgb.save('traj_save/{}/{}/rgb/{}.png'.format(args.save_name, time_now, step))
        # np.save('traj_save/{}/{}/depth/{}.npy'.format(args.save_name, time_now, step), depth.astype(np.float16))

        # # semantic save as npy
        # semantic_path = 'traj_save/{}/{}/semantic/{}.npy'.format(args.save_name, time_now, step)
        # np.save(semantic_path, semantic.astype(np.uint16))
        image['rgb_list'].append(rgb)
        image['depth_list'].append(depth.astype(np.float16))
        image['semantic_list'].append(semantic.astype(np.uint16))

        metadata['action_list'].append(action)
        metadata['gps'].append(gps.tolist())
        metadata['compass'].append(compsss.tolist())

    def save_metadata(metadata, save_name, time_now, j, image, bounding_box, save_path, idx, target_object_name):
        with open(os.path.join(save_path, str(j), 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        for i in range(len(image['rgb_list'])):
            rgb = Image.fromarray(image['rgb_list'][i])
            if os.path.exists(os.path.join(save_path, str(j), 'rgb', '{}.png'.format(i))) is False:
                rgb.save(os.path.join(save_path, str(j), 'rgb', '{}.png'.format(i)))
            else:
                pass
            if os.path.exists(os.path.join(save_path, str(j), 'depth', '{}.npy'.format(i))) is False:
                np.save(os.path.join(save_path, str(j), 'depth', '{}.npy'.format(i)), image['depth_list'][i])
            else:
                pass
            if os.path.exists(os.path.join(save_path, str(j), 'semantic', '{}.npy'.format(i))) is False:
                np.save(os.path.join(save_path, str(j), 'semantic', '{}.npy'.format(i)), image['semantic_list'][i])
            else:
                pass
            semantic_map = image['semantic_list'][i]
            semantic_map = Image.fromarray((semantic_map / 518.0 * 255).astype('uint8').squeeze())
            semantic_map.save(os.path.join(save_path, str(j), 'semantic', '{}.png'.format(i)))
        bb_rgb_image = draw_single_bounding_boxes_and_labels(image['rgb_list'][-1], bounding_box, target_object_name)
        bb_rgb_image = Image.fromarray(bb_rgb_image)
        bb_rgb_image.save(os.path.join(save_path, str(j), 'bb_rgb_{}.png'.format(idx)))

    # instruction_feature = {k: torch.tensor(v).float().to(device) for k, v in instruction_feature.items()}
    object_name_to_idx = object_name_to_idx["category_to_task_category_id"]
    object_idx_to_name = {v: k for k, v in object_name_to_idx.items()}

    task_name = "ddnplus"
    habitat_config = habitat.get_config("benchmark/nav/{}/hssd-200_{}_hssd-hab_with_semantic.yaml".format(task_name, task_name), overrides=["habitat.dataset.split={}".format(args.task_mode)])

    envs = habitat.DDNPlusEnv(config=habitat_config, object_name_to_idx=object_name_to_idx, args=args)
    HabitatSimActions.extend_action_space("find")
    action_mapping = {0: HabitatSimActions.find, 1: HabitatSimActions.move_forward, 2: HabitatSimActions.turn_left, 3: HabitatSimActions.turn_right}
    greedy_follower = DDNPlusShortestPathFollower(envs.sim, 1, False)
    j = 0
    pbar = tqdm(total=args.epoch, desc='Epoch {}/{} from {}'.format(j + 1, args.epoch, rank))
    f = open(os.path.join("{}/{}/{}/{}.txt".format(args.save_dir, args.save_name, time_now, rank)), 'w')

    while j < args.epoch:
        try:
            observations = envs.reset()
            metadata = {}
            image = {}
            metadata['current_episode'] = {}
            metadata['current_episode']['episode_id'] = envs.current_episode.episode_id
            metadata['current_episode']['scene_id'] = envs.current_episode.scene_id
            metadata['current_episode']['basic_demand_instruction'] = envs.current_episode.basic_demand_instruction
            metadata['current_episode']['basic_solution'] = envs.current_episode.basic_solution
            metadata['current_episode']['preferred_demand_instruction'] = envs.current_episode.preferred_demand_instruction
            metadata['current_episode']['preferred_solution'] = envs.current_episode.preferred_solution
            metadata['current_episode']['task_instruction'] = envs.current_episode.task_instruction
            metadata['agent_position'] = envs._sim.get_agent_state().position.tolist()

            f.write(metadata['current_episode']['scene_id'])
            f.write('\n')
            f.write(metadata['current_episode']['task_instruction'])
            f.write('\n--------------------------\n\n')
            rotation = envs._sim.get_agent_state().rotation
            metadata['agent_rotation'] = [rotation.w, rotation.x, rotation.y, rotation.z]
            # with open('traj_save/{}/{}/{}/metadata.json'.format(args.save_name, time_now, j), 'w') as f:
            #     json.dump(metadata, f)
            metadata['action_list'] = []
            metadata['gps'] = []
            metadata['compass'] = []
            image['rgb_list'] = []
            image['depth_list'] = []
            image['semantic_list'] = []

            image['rgb_list'].append(observations['rgb'])
            image['depth_list'].append(observations['depth'].astype(np.float16))
            image['semantic_list'].append(observations['semantic'].astype(np.uint16))
            metadata['gps'].append(observations['gps'].tolist())
            metadata['compass'].append(observations['ddnpluscompass'].tolist())

            flag = 0
            step = 0
            look_flag = 0
            for idx in range(len(envs.task.measurements.measures['ddnplus_distance_to_goal'].min_path)):
                if look_flag == 1:
                    action = {"action": HabitatSimActions.look_down, "action_args": {"bbox": choose_bbox, "semantic_map": observations['semantic'], "depth_map": observations['depth']}}
                    observations, reward, done, info = envs.step(action)
                    save_observation(observations, metadata, step, action['action'], image)
                    step += 1
                    look_flag = 0
                elif look_flag == 2:
                    action = {"action": HabitatSimActions.look_up, "action_args": {"bbox": choose_bbox, "semantic_map": observations['semantic'], "depth_map": observations['depth']}}
                    observations, reward, done, info = envs.step(action)
                    save_observation(observations, metadata, step, action['action'], image)
                    step += 1
                    look_flag = 0
                (target_object_name, target_object_position) = envs.task.measurements.measures['ddnplus_distance_to_goal'].min_path[idx]
                target_object_position_list = envs.task.measurements.measures['ddnplus_distance_to_goal']._object_to_positions[target_object_name]
                for target_object_position_potential in target_object_position_list:
                    if abs(target_object_position_potential[0][0] - target_object_position[0]) < 1e-3 and abs(target_object_position_potential[0][2] - target_object_position[2]) < 1e-3:
                        target_object_height = target_object_position_potential[1]
                        break
                action_greedy = 1

                while action_greedy > 0:
                    choose_bbox = (100, 100, 150, 150)
                    action_greedy = greedy_follower.get_next_action(target_object_position)
                    if action_greedy == -1:
                        flag = -1
                        break

                    action = {"action": action_mapping[action_greedy], "action_args": {"bbox": choose_bbox, "semantic_map": observations['semantic'], "depth_map": observations['depth']}}

                    if action_greedy == 0:
                        agent_position = envs._sim.get_agent_state().position.tolist()

                        if target_object_height > 1.5 and look_flag == 0:
                            action = {"action": HabitatSimActions.look_up, "action_args": {"bbox": choose_bbox, "semantic_map": observations['semantic'], "depth_map": observations['depth']}}
                            observations, reward, done, info = envs.step(action)
                            save_observation(observations, metadata, step, action['action'], image)
                            step += 1
                            look_flag = 1
                        elif target_object_height < 0.75 and look_flag == 0:
                            action = {"action": HabitatSimActions.look_down, "action_args": {"bbox": choose_bbox, "semantic_map": observations['semantic'], "depth_map": observations['depth']}}
                            observations, reward, done, info = envs.step(action)
                            save_observation(observations, metadata, step, action['action'], image)
                            step += 1
                            look_flag = 2

                        visible = envs.object_name_to_idx[target_object_name] in observations["semantic"]
                        turn_right = random.random() < 0.5
                        num_of_turn = 0
                        while visible is False:
                            action_move = HabitatSimActions.turn_right if turn_right else HabitatSimActions.turn_left
                            action = {"action": action_move, "action_args": {"bbox": choose_bbox, "semantic_map": observations['semantic'], "depth_map": observations['depth']}}
                            observations, reward, done, info = envs.step(action)
                            num_of_turn += 1
                            save_observation(observations, metadata, step, action['action'], image)
                            step += 1
                            visible = envs.object_name_to_idx[target_object_name] in observations["semantic"]
                            if num_of_turn > 12:
                                break
                        action = {"action": action_mapping[action_greedy], "action_args": {"bbox": choose_bbox, "semantic_map": observations['semantic'], "depth_map": observations['depth']}}

                    elif action_greedy == -1:
                        flag = -1
                        break
                    observations, reward, done, info = envs.step(action)
                    # print("agent position: ", envs._sim.get_agent_state().position.tolist())
                    save_observation(observations, metadata, step, action['action'], image)
                    if action['action'] == HabitatSimActions.find:
                        if envs.object_name_to_idx[target_object_name] in observations["semantic"]:
                            if os.path.exists(os.path.join(save_path, str(j))) is False:
                                os.makedirs(os.path.join(save_path, str(j)))
                                os.makedirs(os.path.join(save_path, str(j), 'rgb'))
                                os.makedirs(os.path.join(save_path, str(j), 'depth'))
                                os.makedirs(os.path.join(save_path, str(j), 'semantic'))
                            bounding_box = find_bounding_box(observations['semantic'], envs.object_name_to_idx[target_object_name])
                            save_metadata(metadata, args.save_name, time_now, j, image, bounding_box, save_path, idx, target_object_name)
                            flag = 1

                        elif idx == 0:
                            flag = -1
                        break
                if flag == -1:
                    break
            if flag == 1:
                j += 1
                pbar.update(1)
        except Exception as e:
            print(e)
            f.write("error")
            f.write(str(e))
            continue


def run_in_parallel(args, time_now, target_fn):

    num_processes = args.workers
    processes = []
    if os.path.exists('{}/{}/{}'.format(args.save_dir, args.save_name, time_now)) is False:
        os.makedirs('{}/{}/{}'.format(args.save_dir, args.save_name, time_now))

    for rank in range(num_processes):
        # 每个进程调用 local_data_collection 函数，传递不同的 rank 值
        p = multiprocessing.Process(target=target_fn, args=(args, time_now, rank))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # 等待所有进程完成


if __name__ == "__main__":
    args = parse_arguments()
    time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    args.time_now = time_now
    # if args.running_mode == "local_data_collection":
    #     run_in_parallel(args, time_now, target_fn=local_data_collection)
    # if args.running_mode == "full_data_collection":
    #     run_in_parallel(args, time_now, target_fn=full_data_collection)
    # local_data_collection(args, time_now, rank=0)
    full_data_collection(args, time_now, rank=0)
