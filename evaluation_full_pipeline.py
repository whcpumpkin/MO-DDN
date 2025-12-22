import glob
import json
import os
import random
import time
from typing import Dict, Optional

import habitat
import numpy as np
import torch
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from tqdm import tqdm

from map import Mapper, habitat_camera_intrinsic
from models_code.DDNP_agent import AgentModel, FullAgent
from utils.args import parse_arguments


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_object_mappings(train_json: str) -> Dict:
    with open(train_json, "r") as f:
        object_name_to_idx = json.load(f)["category_to_task_category_id"]
    return object_name_to_idx


def ensure_output_dirs(save_dir: str, run_id: str) -> str:
    target_dir = os.path.join(save_dir, run_id)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def resolve_agent_checkpoint(args) -> Optional[str]:
    candidates = []
    if args.eval_model_path:
        candidates.append(args.eval_model_path)
    candidates.append(os.path.join("pre_trained_models", "il_agent.pth"))

    for path in candidates:
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            for name in ["il_agent.pth", "agent.pth", "attribute_model_best.pth"]:
                candidate = os.path.join(path, name)
                if os.path.isfile(candidate):
                    return candidate
            pth_files = sorted(glob.glob(os.path.join(path, "*.pth")))
            if pth_files:
                return pth_files[-1]
    return None


def build_env(args, object_name_to_idx):
    task_name = "ddnplus"
    habitat_config = habitat.get_config(
        f"benchmark/nav/{task_name}/hssd-200_{task_name}_hssd-hab_with_semantic.yaml",
        overrides=[f"habitat.dataset.split={args.task_mode}",],
    )
    envs = habitat.DDNPlusEnv(config=habitat_config, object_name_to_idx=object_name_to_idx, args=args)
    HabitatSimActions.extend_action_space("find")
    return envs, habitat_config


def build_agent(args, habitat_config, checkpoint_path: Optional[str]):
    if args.random_fine:
        return None
    if checkpoint_path is None:
        raise FileNotFoundError("No agent checkpoint found; set --eval_model_path to a valid .pth file.")
    agent = AgentModel(
        image_size=habitat_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.width,
        args=args,
    )
    agent.load_model(checkpoint_path)
    return agent


@torch.no_grad()
def full_pipeline(args):
    seed_everything(args.seed)
    run_id = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    object_name_to_idx = load_object_mappings("data/datasets/ddnplus/hssd-hab_v0.2.5/train/train.json")
    output_dir = ensure_output_dirs(args.save_dir, run_id)

    envs, habitat_config = build_env(args, object_name_to_idx)
    mapper = Mapper(
        habitat_camera_intrinsic(habitat_config),
        pcd_resolution=0.05,
        grid_resolution=0.2,
        grid_size=5,
        floor_height=-1,
        ceiling_height=0.8,
        device="cpu:0",
        args=args,
    )

    checkpoint_path = resolve_agent_checkpoint(args)
    agent = build_agent(args, habitat_config, checkpoint_path)
    full_agent = FullAgent(args, agent, mapper, envs)
    full_agent.to(args.device)

    basic_success = 0
    preferred_success = 0
    basic_spl = 0
    preferred_spl = 0

    def get_action(action_out, observation):
        if isinstance(action_out, torch.Tensor):
            action_out = action_out.item()
        return {
            "action": action_out,
            "action_args": {
                "semantic_map": observation["semantic"],
                "depth_map": observation["depth"],
            },
        }

    for epoch in range(args.epoch):
        observation = envs.reset()
        agent_state = envs.sim.get_agent_state().sensor_states["rgb"]
        full_agent.reset(agent_state.position, agent_state.rotation)
        full_agent.update(observation, agent_state.position, agent_state.rotation)

        for _ in tqdm(range(args.max_step), desc=f"Epoch {epoch}"):
            if envs.task.find_called_times == 5:
                action_out = 0
            else:
                if args.add_noise:
                    observation["depth"] = observation["depth"] + np.random.normal(0, 0.03, observation["depth"].shape)
                    observation["depth"] = np.clip(observation["depth"], 0, 10)
                action_out = full_agent.act(observation)

            action_to_env = get_action(action_out, observation)
            observation, _, done, _ = envs.step(action_to_env)

            agent_state = envs.sim.get_agent_state().sensor_states["rgb"]
            full_agent.update(observation, agent_state.position, agent_state.rotation)

            if action_out == 0:
                basic_success += envs.task.measurements.measures["ddnplus_basic_success"].get_metric()
                preferred_success += envs.task.measurements.measures["ddnplus_preference_success"].get_metric()
                basic_spl += envs.task.measurements.measures["ddnplus_basic_spl"].get_metric()
                preferred_spl += envs.task.measurements.measures["ddnplus_preference_spl"].get_metric()
                break
            if done:
                break

        print(f"basic_success: {basic_success / (epoch + 1):.3f}")
        print(f"preferred_success: {preferred_success / (epoch + 1):.3f}")
        print(f"basic_SPL: {basic_spl / (epoch + 1):.3f}")
        print(f"preferred_SPL: {preferred_spl / (epoch + 1):.3f}")

    metrics = {
        "basic_success": basic_success / max(args.epoch, 1),
        "preferred_success": preferred_success / max(args.epoch, 1),
        "basic_spl": basic_spl / max(args.epoch, 1),
        "preferred_spl": preferred_spl / max(args.epoch, 1),
        "run_id": run_id,
        "output_dir": output_dir,
        "checkpoint": checkpoint_path,
    }

    with open(os.path.join(output_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    args = parse_arguments()
    stats = full_pipeline(args)
    print("Evaluation finished. Metrics saved to", stats["output_dir"])
