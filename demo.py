import random
import torch
import habitat
from habitat.core.env import DDNPlusEnv
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from utils.args import parse_arguments
import json
from tqdm import tqdm


def main():
    task_name = "ddnplus"
    habitat_config = habitat.get_config("benchmark/nav/{}/hssd-200_{}_hssd-hab_with_semantic.yaml".format(task_name, task_name), overrides=["habitat.dataset.split={}".format(args.task_mode)])
    with open("data/datasets/ddnplus/hssd-hab_v0.2.5/train/train.json", "r") as f:
        object_name_to_idx = json.load(f)
    object_name_to_idx = object_name_to_idx["category_to_task_category_id"]
    object_idx_to_name = {v: k for k, v in object_name_to_idx.items()}
    envs = habitat.DDNPlusEnv(config=habitat_config, object_name_to_idx=object_name_to_idx, args=args)
    HabitatSimActions.extend_action_space("find")

    def get_action(action_out, rgb_input, depth_input, semantic_input):
        if type(action_out) == torch.Tensor:
            action_out = action_out.item()
        # {"action": ac_random, "action_args": {"bbox": choose_bbox, "semantic_map": observations[i]['semantic'], "depth_map": observations[i]['depth']}} for i in range(workers)
        action = {"action": action_out, "action_args": {"semantic_map": semantic_input, "depth_map": depth_input}}
        return action

    observation = envs.reset()
    for step in tqdm(range(100)):
        action_out = random.randint(0, len(HabitatSimActions) - 1)
        action_to_env = get_action(action_out, observation['rgb'], observation['depth'], observation['semantic'])
        # we designed a reward function but did not use it in our method.
        observation, reward, done, info = envs.step(action_to_env)


if __name__ == "__main__":
    args = parse_arguments()
    main()
