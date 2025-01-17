#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import gym
import numba
import numpy as np
from gym import spaces
from habitat.config import read_write
from habitat.core.dataset import BaseEpisode, Dataset, Episode, EpisodeIterator
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks.registration import make_task
from habitat.utils import profiling_wrapper
import math

if TYPE_CHECKING:
    from omegaconf import DictConfig


class Fine_Grained_Env():
    r"""Fundamental environment class for :ref:`habitat`.

    :data observation_space: ``SpaceDict`` object corresponding to sensor in
        sim and task.
    :data action_space: ``gym.space`` object corresponding to valid actions.

    All the information  needed for working on embodied task with simulator
    is abstracted inside :ref:`Env`. Acts as a base for other derived
    environment classes. :ref:`Env` consists of three major components:
    ``dataset`` (`episodes`), ``simulator`` (:ref:`sim`) and :ref:`task` and
    connects all the three components together.
    """

    observation_space: spaces.Dict
    action_space: spaces.Dict
    _config: "DictConfig"
    _dataset: Optional[Dataset[Episode]]
    number_of_episodes: Optional[int]
    _current_episode: Optional[Episode]
    _episode_iterator: Optional[Iterator[Episode]]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool
    _episode_from_iter_on_reset: bool
    _episode_force_changed: bool

    def __init__(self, config: "DictConfig", block_size=2, object_name_to_idx=None, args=None) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """

        if "habitat" in config:
            config = config.habitat
        self.args = args
        HabitatSimActions.extend_action_space("find")
        HabitatSimActions.extend_action_space("leave")
        self.block_size = block_size
        self.object_name_to_idx = object_name_to_idx
        self._config = config
        self._dataset = None
        if self._dataset is None and config.dataset.type:
            self._dataset = make_dataset(id_dataset=config.dataset.type, config=config.dataset)

        self._current_episode = None
        self._episode_iterator = None
        self._episode_from_iter_on_reset = True
        self._episode_force_changed = False

        # load the first scene if dataset is present
        if self._dataset:
            assert (len(self._dataset.episodes) > 0), "dataset should have non-empty episodes list"
            # self._setup_episode_iterator()
            # self.current_episode = next(self.episode_iterator)
            self._current_episode = random.choice(self.episodes)
            while len(self._current_episode.scene_ids[self.args.scene_mode]) == 0:
                self._current_episode = random.choice(self.episodes)
            self._current_episode.scene_id = random.choice(self._current_episode.scene_ids[self.args.scene_mode])

            with read_write(self._config):
                self._config.simulator.scene_dataset = (self.current_episode.scene_dataset_config)
                self._config.simulator.scene = self.current_episode.scene_id

            self.number_of_episodes = len(self.episodes)
        else:
            self.number_of_episodes = None

        self._sim = make_sim(id_sim=self._config.simulator.type, config=self._config.simulator)

        self._task = make_task(
            self._config.task.type,
            config=self._config.task,
            sim=self._sim,
            dataset=self._dataset,
        )
        self.observation_space = spaces.Dict({
            **self._sim.sensor_suite.observation_spaces.spaces,
            **self._task.sensor_suite.observation_spaces.spaces,
        })
        self.action_space = self._task.action_space
        self._max_episode_seconds = (self._config.environment.max_episode_seconds)
        self._max_episode_steps = self._config.environment.max_episode_steps
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False

    def _setup_episode_iterator(self):
        assert self._dataset is not None
        iter_option_dict = {k.lower(): v for k, v in self._config.environment.iterator_options.items()}
        iter_option_dict["seed"] = self._config.seed
        self._episode_iterator = self._dataset.get_episode_iterator(**iter_option_dict)

    @property
    def current_episode(self) -> Episode:
        assert self._current_episode is not None
        return self._current_episode

    @current_episode.setter
    def current_episode(self, episode: Episode) -> None:
        self._current_episode = episode
        # This allows the current episode to be set here
        # and then reset be called without the episode changing
        self._episode_from_iter_on_reset = False
        self._episode_force_changed = True

    @property
    def episode_iterator(self) -> Iterator[Episode]:
        return self._episode_iterator

    @episode_iterator.setter
    def episode_iterator(self, new_iter: Iterator[Episode]) -> None:
        self._episode_iterator = new_iter
        self._episode_force_changed = True
        self._episode_from_iter_on_reset = True

    @property
    def episodes(self) -> List[Episode]:
        return (self._dataset.episodes if self._dataset else cast(List[Episode], []))

    @episodes.setter
    def episodes(self, episodes: List[Episode]) -> None:
        assert (len(episodes) > 0), "Environment doesn't accept empty episodes list."
        assert (self._dataset is not None), "Environment must have a dataset to set episodes"
        self._dataset.episodes = episodes
        self._setup_episode_iterator()
        self._current_episode = None
        self._episode_force_changed = True
        self._episode_from_iter_on_reset = True

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def task(self) -> EmbodiedTask:
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (self._episode_start_time), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        return (self._max_episode_steps != 0 and self._max_episode_steps <= self._elapsed_steps) or (self._max_episode_seconds != 0 and self._max_episode_seconds <= self._elapsed_seconds)

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False

    def reset(self, change_episode=False) -> Observations:
        r"""Resets the environments and returns the initial observations.

        :return: initial observations from the environment.
        """
        while True:
            try:
                self._reset_stats()

                # Delete the shortest path cache of the current episode
                # Caching it for the next time we see this episode isn't really worth
                # it
                if self._current_episode is not None:
                    self._current_episode._shortest_path_cache = None

                # if (self._episode_iterator is not None
                #         and self._episode_from_iter_on_reset):
                # self._current_episode = next(self._episode_iterator)
                if change_episode:
                    self._current_episode = random.choice(self.episodes)
                    while len(self._current_episode.scene_ids[self.args.scene_mode]) == 0:
                        self._current_episode = random.choice(self.episodes)
                    self._current_episode.scene_id = random.choice(self._current_episode.scene_ids[self.args.scene_mode])
                    # self._current_episode.scene_id = "data/scene_datasets/hssd-hab/scenes/106879023_174887148.scene_instance.json"
                    # print("change episode is True, reset scene_id: ", self._current_episode.scene_id)
                    # print("change is True, reset task instruction for sample_navigable_point", self._current_episode.task_instruction)
                else:
                    # print("change episode is False, reset scene_id: ", self._current_episode.scene_id)
                    # print("change is False, reset task instruction for sample_navigable_point", self._current_episode.task_instruction)
                    pass

                # This is always set to true after a reset that way
                # on the next reset an new episode is taken (if possible)
                self._episode_from_iter_on_reset = True
                self._episode_force_changed = False

                assert self._current_episode is not None, "Reset requires an episode"

                # reset scene_id
                self._current_episode.info = {}
                self.reconfigure(self._config)
                self.exploration_map = set()
                self.already_found = []
                if random.random() < 0.0:
                    self.target_object_in_block = False
                else:
                    self.target_object_in_block = True

                while True:
                    num_try_to_sample_navigable_point = 0
                    max_try_to_sample_navigable_point = 100
                    while num_try_to_sample_navigable_point < max_try_to_sample_navigable_point:
                        # print("num_try_to_sample_navigable_point: ", num_try_to_sample_navigable_point)
                        if self.target_object_in_block:
                            self.reset_agent()

                            observations = self.task.reset(episode=self.current_episode)
                            self._task.measurements.reset_measures(
                                episode=self.current_episode,
                                task=self.task,
                                observations=observations,
                            )
                            if self._task.measurements.measures['ddnplus_distance_to_goal']._metric < 99999:
                                self.target_object_name = random.choice(list(self.task.measurements.measures['ddnplus_distance_to_goal'].target_cateogry.keys()))
                                # self.target_object_position = self._task.measurements.measures['ddnplus_distance_to_one_goal'].min_obj_position
                                # self.target_object_name = self._task.measurements.measures['ddnplus_distance_to_one_goal'].min_obj_name
                                self.target_object_position = random.choice(self.task.measurements.measures['ddnplus_distance_to_one_goal']._object_to_positions[self.target_object_name])
                                self.target_object_height = self.target_object_position[1]
                                self.target_object_position = self.target_object_position[0]
                                self.reset_agent(self.target_object_position)
                                observations = self.task.reset(episode=self.current_episode)
                                self._task.measurements.reset_measures(
                                    episode=self.current_episode,
                                    task=self.task,
                                    observations=observations,
                                )
                                current_position = self._sim.get_agent_state().position
                                self.distance = self._sim.geodesic_distance(current_position, self.target_object_position)
                                # found = list(self.task.measurements.measures['ddnplus_distance_to_goal'].target_cateogry.keys()).remove(self.target_object_name)
                                # observations["already_found"] = random.sample(found, random.randint(1, len(found)))
                                # self.already_found = observations["already_found"]
                                print("reset scene_id: ", self._current_episode.scene_id)
                                return observations
                        else:
                            self.reset_agent()
                            observations = self.task.reset(episode=self.current_episode)
                            self._task.measurements.reset_measures(
                                episode=self.current_episode,
                                task=self.task,
                                observations=observations,
                            )
                            dis_to_min_goal = self._task.measurements.measures['ddnplus_distance_to_one_goal']._metric
                            if dis_to_min_goal > self.block_size * 1.3 and dis_to_min_goal < 99999:
                                # found = list(self.task.measurements.measures['ddnplus_distance_to_goal'].target_cateogry.keys())
                                # observations["already_found"] = random.sample(found, random.randint(1, len(found)))
                                # self.already_found = observations["already_found"]
                                print("reset scene_id: ", self._current_episode.scene_id)
                                return observations
                        num_try_to_sample_navigable_point += 1

                    self._current_episode = random.choice(self.episodes)
                    while len(self._current_episode.scene_ids[self.args.scene_mode]) == 0:
                        self._current_episode = random.choice(self.episodes)
                    self._current_episode.scene_id = random.choice(self._current_episode.scene_ids[self.args.scene_mode])
                    # self._current_episode.scene_id = "data/scene_datasets/hssd-hab/scenes/106879023_174887148.scene_instance.json"
                    # print("change episode for sample_navigable_point, reset scene_id: ", self._current_episode.scene_id)
                    # print("change task instruction for sample_navigable_point", self._current_episode.task_instruction)
                    self.reconfigure(self._config)
            except Exception as e:
                f = open('./DDNP_save/{}/{}/error.txt'.format(self.args.save_name, self.args.time_now), "a")
                f.write("----------\n")
                f.write(self.args.time_now)
                f.write(str(e))
                f.write(self._current_episode.task_instruction)
                f.write("----------\n")
                f.close()
                print("--------------------------------------------------------------")
                print("Reset failed")
                print(e)
                print(self._current_episode.task_instruction)
                change_episode = True

        # except Exception as e:
        #     print("--------------------------------------------------------------")
        #     print("Reset failed")
        #     print(e)
        #     f = open('./DDNP_save/{}/{}/error.txt'.format(self.args.save_name, self.args.time_now), "a")
        #     f.write("----------\n")
        #     f.write(self.args.time_now)
        #     f.write(str(e))
        #     f.write("----------\n")
        #     f.close()
        #     print("--------------------------------------------------------------")

    def reset_agent(self, target_object_position=None):
        if target_object_position is None:
            self.current_episode.start_position = self._sim.sample_navigable_point()
            while self.current_episode.start_position[1] > 0.5:
                self.current_episode.start_position = self._sim.sample_navigable_point()
            self.current_episode.start_rotation = random.choice(self._task.rotation)
            self.reconfigure(self._config)
        else:
            self.navigable_target_position = target_object_position
            self.current_episode.start_position = self._sim.pathfinder.get_random_navigable_point_near(self.navigable_target_position, self.block_size, 5000).astype(float).tolist()
            self.current_episode.start_rotation = random.choice(self._task.rotation)
            self.reconfigure(self._config)

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active
        if self._past_limit():
            self._episode_over = True

        if self.episode_iterator is not None and isinstance(self.episode_iterator, EpisodeIterator):
            self.episode_iterator.step_taken()

    def step(self, action: Union[int, str, Dict[str, Any]], **kwargs) -> Observations:
        r"""Perform an action in the environment and return observations.

        :param action: action (belonging to :ref:`action_space`) to be
            performed inside the environment. Action is a name or index of
            allowed task's action and action arguments (belonging to action's
            :ref:`action_space`) to support parametrized and continuous
            actions.
        :return: observations after taking action in environment.
        """

        assert (self._episode_start_time is not None), "Cannot call step before calling reset"
        # assert (self._episode_over is False), "Episode over, call reset before calling step"
        assert (not self._episode_force_changed), "Episode was changed either by setting current_episode or changing the episodes list. Call reset before stepping the environment again."
        if self._episode_over:
            return self.prev_observation, 0, True, self.current_episode.info
        if "action_args" not in action.keys():
            action = action["action"]
        # Support simpler interface as well
        if isinstance(action, (str, int, np.integer)):
            action = {"action": action}

        observations = self.task.step(action=action, episode=self.current_episode)

        self._task.measurements.update_measures(
            episode=self.current_episode,
            action=action,
            task=self.task,
            observations=observations,
        )

        self._update_step_stats()
        reward, done = self.reward_function(observations, action)
        done = done or self.episode_over
        self.prev_observation = observations
        # observations['already_found'] = self.already_found
        return observations, reward, done, self.current_episode.info

    def reward_function(self, observations, action):
        done = False
        dense_reward = {'leave_success_reward': 0, 'find_success_reward': 0, 'delta_distance_reward': 0, 'exploration_reward': 0, 'block_penalty': 0}
        current_agent_position = self._sim.get_agent_state().position
        current_agent_rotation = self._sim.get_agent_state().rotation
        if action['action'] == HabitatSimActions.find or action['action'] == HabitatSimActions.leave:
            if self.target_object_in_block is False:
                if action['action'] == HabitatSimActions.leave:
                    dense_reward['leave_success_reward'] = 1
                    self.current_episode.info['leave_success'] = 1
                else:
                    dense_reward['leave_success_reward'] = -1
                    self.current_episode.info['leave_success'] = 0
            else:
                dis_between_agent_and_target = np.linalg.norm(np.array(current_agent_position) - np.array(self.navigable_target_position))
                visible = self.object_name_to_idx[self.target_object_name] in observations["semantic"]
                if action['action'] == HabitatSimActions.find and visible and dis_between_agent_and_target < self.task._config.measurements.ddnplus_basic_success['ddnplus_success_distance']:
                    dense_reward['find_success_reward'] = 5
                    self.current_episode.info['find_success'] = 1
                else:
                    self.current_episode.info['find_success'] = 0
                self.current_episode.info['dis_between_agent_and_target'] = dis_between_agent_and_target
            done = True
            # if action['action'] == HabitatSimActions.find:
            #     print("Find")
            # if action['action'] == HabitatSimActions.leave:
            #     print("Leave")

            return dense_reward, done
        if self.target_object_in_block is False:
            dense_reward['delta_distance_reward'] = 0
        else:
            current_position = self._sim.get_agent_state().position
            current_distance = self._sim.geodesic_distance(current_position, self.target_object_position)
            dense_reward['delta_distance_reward'] = max((self.distance - current_distance) * 10, -0.5)
            self.distance = current_distance

        exploration_x = round(current_agent_position[0], 2)
        exploration_y = round(current_agent_position[2], 2)
        w = round(current_agent_rotation.w, 2)
        x = round(current_agent_rotation.x, 2)
        y = round(current_agent_rotation.y, 2)
        z = round(current_agent_rotation.z, 2)
        if (exploration_x, exploration_y, w, x, y, z) not in self.exploration_map:
            self.exploration_map.add((exploration_x, exploration_y, w, x, y, z))
            dense_reward['exploration_reward'] = 0.00
        else:
            dense_reward['exploration_reward'] = -0.01

        dis_to_start_position = np.linalg.norm(np.array(current_agent_position) - np.array(self.current_episode.start_position))
        if dis_to_start_position > self.block_size * 1.3:
            dense_reward['block_penalty'] = -0.5

        return dense_reward, done

    @staticmethod
    @numba.njit
    def _seed_numba(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self._seed_numba(seed)
        self._sim.seed(seed)
        self._task.seed(seed)

    def reconfigure(self, config: "DictConfig") -> None:
        self._config = self._task.overwrite_sim_config(config, self.current_episode)

        self._sim.reconfigure(self._config.simulator, self.current_episode)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DDNPlusEnv:
    r"""Fundamental environment class for :ref:`habitat`.

    :data observation_space: ``SpaceDict`` object corresponding to sensor in
        sim and task.
    :data action_space: ``gym.space`` object corresponding to valid actions.

    All the information  needed for working on embodied task with simulator
    is abstracted inside :ref:`Env`. Acts as a base for other derived
    environment classes. :ref:`Env` consists of three major components:
    ``dataset`` (`episodes`), ``simulator`` (:ref:`sim`) and :ref:`task` and
    connects all the three components together.
    """

    observation_space: spaces.Dict
    action_space: spaces.Dict
    _config: "DictConfig"
    _dataset: Optional[Dataset[Episode]]
    number_of_episodes: Optional[int]
    _current_episode: Optional[Episode]
    _episode_iterator: Optional[Iterator[Episode]]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool
    _episode_from_iter_on_reset: bool
    _episode_force_changed: bool

    def __init__(self, config: "DictConfig", block_size=2, object_name_to_idx=None, args=None) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """

        if "habitat" in config:
            config = config.habitat
        self.args = args
        self._config = config
        self._dataset = None
        if self._dataset is None and config.dataset.type:
            self._dataset = make_dataset(id_dataset=config.dataset.type, config=config.dataset)
        self.object_name_to_idx = object_name_to_idx
        self._current_episode = None
        self._episode_iterator = None
        self._episode_from_iter_on_reset = True
        self._episode_force_changed = False

        # load the first scene if dataset is present
        if self._dataset:
            assert (len(self._dataset.episodes) > 0), "dataset should have non-empty episodes list"
            # self._setup_episode_iterator()
            # self.current_episode = next(self.episode_iterator)
            self._current_episode = random.choice(self.episodes)
            while len(self._current_episode.scene_ids[self.args.scene_mode]) == 0:
                self._current_episode = random.choice(self.episodes)
            self._current_episode.scene_id = random.choice(self._current_episode.scene_ids[self.args.scene_mode])
            with read_write(self._config):
                self._config.simulator.scene_dataset = (self.current_episode.scene_dataset_config)
                self._config.simulator.scene = self.current_episode.scene_id

            self.number_of_episodes = len(self.episodes)
        else:
            self.number_of_episodes = None
        print("num_of_episodes: ", self.number_of_episodes)
        self._sim = make_sim(id_sim=self._config.simulator.type, config=self._config.simulator)

        self._task = make_task(
            self._config.task.type,
            config=self._config.task,
            sim=self._sim,
            dataset=self._dataset,
        )
        self._task.object_name_to_idx = {k.split(".n")[0]: v for k, v in self.object_name_to_idx.items()}
        self.observation_space = spaces.Dict({
            **self._sim.sensor_suite.observation_spaces.spaces,
            **self._task.sensor_suite.observation_spaces.spaces,
        })
        self.action_space = self._task.action_space
        self._max_episode_seconds = (self._config.environment.max_episode_seconds)
        self._max_episode_steps = self._config.environment.max_episode_steps
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False

    def _setup_episode_iterator(self):
        assert self._dataset is not None
        iter_option_dict = {k.lower(): v for k, v in self._config.environment.iterator_options.items()}
        iter_option_dict["seed"] = self._config.seed
        self._episode_iterator = self._dataset.get_episode_iterator(**iter_option_dict)

    @property
    def current_episode(self) -> Episode:
        assert self._current_episode is not None
        return self._current_episode

    @current_episode.setter
    def current_episode(self, episode: Episode) -> None:
        self._current_episode = episode
        # This allows the current episode to be set here
        # and then reset be called without the episode changing
        self._episode_from_iter_on_reset = False
        self._episode_force_changed = True

    @property
    def episode_iterator(self) -> Iterator[Episode]:
        return self._episode_iterator

    @episode_iterator.setter
    def episode_iterator(self, new_iter: Iterator[Episode]) -> None:
        self._episode_iterator = new_iter
        self._episode_force_changed = True
        self._episode_from_iter_on_reset = True

    @property
    def episodes(self) -> List[Episode]:
        return (self._dataset.episodes if self._dataset else cast(List[Episode], []))

    @episodes.setter
    def episodes(self, episodes: List[Episode]) -> None:
        assert (len(episodes) > 0), "Environment doesn't accept empty episodes list."
        assert (self._dataset is not None), "Environment must have a dataset to set episodes"
        self._dataset.episodes = episodes
        self._setup_episode_iterator()
        self._current_episode = None
        self._episode_force_changed = True
        self._episode_from_iter_on_reset = True

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def task(self) -> EmbodiedTask:
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (self._episode_start_time), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        return (self._max_episode_steps != 0 and self._max_episode_steps <= self._elapsed_steps) or (self._max_episode_seconds != 0 and self._max_episode_seconds <= self._elapsed_seconds)

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False

    def reset(self, scene_name=None) -> Observations:
        r"""Resets the environments and returns the initial observations.

        :return: initial observations from the environment.
        """
        while True:
            try:
                self._reset_stats()

                # Delete the shortest path cache of the current episode
                # Caching it for the next time we see this episode isn't really worth
                # it
                if self._current_episode is not None:
                    self._current_episode._shortest_path_cache = None

                # if (self._episode_iterator is not None
                #         and self._episode_from_iter_on_reset):
                # self._current_episode = next(self._episode_iterator)
                self._current_episode = random.choice(self.episodes)
                while len(self._current_episode.scene_ids[self.args.scene_mode]) == 0:
                    self._current_episode = random.choice(self.episodes)
                if scene_name is not None:
                    while scene_name not in self._current_episode.scene_ids[self.args.scene_mode]:
                        self._current_episode = random.choice(self.episodes)
                if scene_name is not None:
                    self._current_episode.scene_id = scene_name
                else:
                    self._current_episode.scene_id = random.choice(self._current_episode.scene_ids[self.args.scene_mode])

                # print("reset scene_id: ", self._current_episode.scene_id)

                # This is always set to true after a reset that way
                # on the next reset an new episode is taken (if possible)
                self._episode_from_iter_on_reset = True
                self._episode_force_changed = False

                assert self._current_episode is not None, "Reset requires an episode"

                # reset scene_id
                self.current_episode.info = {}
                self.reconfigure(self._config)
                while True:
                    num_try_to_sample_navigable_point = 0
                    max_try_to_sample_navigable_point = 10
                    while num_try_to_sample_navigable_point < max_try_to_sample_navigable_point:
                        self.reset_agent()

                        observations = self.task.reset(episode=self.current_episode)
                        self._task.measurements.reset_measures(
                            episode=self.current_episode,
                            task=self.task,
                            observations=observations,
                        )
                        if self._task.measurements.measures['ddnplus_distance_to_goal']._metric < 99999:
                            return observations
                        num_try_to_sample_navigable_point += 1

                    self._current_episode = random.choice(self.episodes)
                    while len(self._current_episode.scene_ids[self.args.scene_mode]) == 0:
                        self._current_episode = random.choice(self.episodes)
                    self._current_episode.scene_id = random.choice(self._current_episode.scene_ids[self.args.scene_mode])
                    if scene_name is not None:
                        while scene_name not in self._current_episode.scene_ids[self.args.scene_mode]:
                            self._current_episode = random.choice(self.episodes)
                        self._current_episode.scene_id = scene_name
                    print("reset scene_id: ", self._current_episode.scene_id)
                    self.reconfigure(self._config)
                    self._current_episode.info = {}
            except Exception as e:
                print(e)
                continue

    def reset_agent(self):
        self.current_episode.start_position = self._sim.sample_navigable_point()
        while self.current_episode.start_position[1] > 0.5:
            self.current_episode.start_position = self._sim.sample_navigable_point()
        self.current_episode.start_rotation = random.choice(self._task.rotation)
        self.reconfigure(self._config)

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active
        if self._past_limit():
            self._episode_over = True

        if self.episode_iterator is not None and isinstance(self.episode_iterator, EpisodeIterator):
            self.episode_iterator.step_taken()

    def step(self, action: Union[int, str, Dict[str, Any]], **kwargs) -> Observations:
        r"""Perform an action in the environment and return observations.

        :param action: action (belonging to :ref:`action_space`) to be
            performed inside the environment. Action is a name or index of
            allowed task's action and action arguments (belonging to action's
            :ref:`action_space`) to support parametrized and continuous
            actions.
        :return: observations after taking action in environment.
        """

        assert (self._episode_start_time is not None), "Cannot call step before calling reset"
        assert (self._episode_over is False), "Episode over, call reset before calling step"
        assert (not self._episode_force_changed), "Episode was changed either by setting current_episode or changing the episodes list. Call reset before stepping the environment again."

        if "action_args" not in action.keys():
            action = action["action"]
        # Support simpler interface as well
        if isinstance(action, (str, int, np.integer)):
            action = {"action": action}

        observations = self.task.step(action=action, episode=self.current_episode)

        self._task.measurements.update_measures(
            episode=self.current_episode,
            action=action,
            task=self.task,
            observations=observations,
        )

        # self._update_step_stats()

        return observations, 0, self.episode_over, self.current_episode.info

    @staticmethod
    @numba.njit
    def _seed_numba(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self._seed_numba(seed)
        self._sim.seed(seed)
        self._task.seed(seed)

    def reconfigure(self, config: "DictConfig") -> None:
        self._config = self._task.overwrite_sim_config(config, self.current_episode)

        self._sim.reconfigure(self._config.simulator, self.current_episode)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Env:
    r"""Fundamental environment class for :ref:`habitat`.

    :data observation_space: ``SpaceDict`` object corresponding to sensor in
        sim and task.
    :data action_space: ``gym.space`` object corresponding to valid actions.

    All the information  needed for working on embodied task with simulator
    is abstracted inside :ref:`Env`. Acts as a base for other derived
    environment classes. :ref:`Env` consists of three major components:
    ``dataset`` (`episodes`), ``simulator`` (:ref:`sim`) and :ref:`task` and
    connects all the three components together.
    """

    observation_space: spaces.Dict
    action_space: spaces.Dict
    _config: "DictConfig"
    _dataset: Optional[Dataset[Episode]]
    number_of_episodes: Optional[int]
    _current_episode: Optional[Episode]
    _episode_iterator: Optional[Iterator[Episode]]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool
    _episode_from_iter_on_reset: bool
    _episode_force_changed: bool

    def __init__(self, config: "DictConfig", dataset: Optional[Dataset[Episode]] = None) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """

        if "habitat" in config:
            config = config.habitat
        self._config = config
        self._dataset = dataset
        if self._dataset is None and config.dataset.type:
            self._dataset = make_dataset(id_dataset=config.dataset.type, config=config.dataset)

        self._current_episode = None
        self._episode_iterator = None
        self._episode_from_iter_on_reset = True
        self._episode_force_changed = False

        # load the first scene if dataset is present
        if self._dataset:
            assert (len(self._dataset.episodes) > 0), "dataset should have non-empty episodes list"
            self._setup_episode_iterator()
            self.current_episode = next(self.episode_iterator)
            with read_write(self._config):
                self._config.simulator.scene_dataset = (self.current_episode.scene_dataset_config)
                self._config.simulator.scene = self.current_episode.scene_id

            self.number_of_episodes = len(self.episodes)
        else:
            self.number_of_episodes = None

        self._sim = make_sim(id_sim=self._config.simulator.type, config=self._config.simulator)

        self._task = make_task(
            self._config.task.type,
            config=self._config.task,
            sim=self._sim,
            dataset=self._dataset,
        )
        self.observation_space = spaces.Dict({
            **self._sim.sensor_suite.observation_spaces.spaces,
            **self._task.sensor_suite.observation_spaces.spaces,
        })
        self.action_space = self._task.action_space
        self._max_episode_seconds = (self._config.environment.max_episode_seconds)
        self._max_episode_steps = self._config.environment.max_episode_steps
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False

    def _setup_episode_iterator(self):
        assert self._dataset is not None
        iter_option_dict = {k.lower(): v for k, v in self._config.environment.iterator_options.items()}
        iter_option_dict["seed"] = self._config.seed

        self._episode_iterator = self._dataset.get_episode_iterator(**iter_option_dict)

    @property
    def current_episode(self) -> Episode:
        assert self._current_episode is not None
        return self._current_episode

    @current_episode.setter
    def current_episode(self, episode: Episode) -> None:
        self._current_episode = episode
        # This allows the current episode to be set here
        # and then reset be called without the episode changing
        self._episode_from_iter_on_reset = False
        self._episode_force_changed = True

    @property
    def episode_iterator(self) -> Iterator[Episode]:
        return self._episode_iterator

    @episode_iterator.setter
    def episode_iterator(self, new_iter: Iterator[Episode]) -> None:
        self._episode_iterator = new_iter
        self._episode_force_changed = True
        self._episode_from_iter_on_reset = True

    @property
    def episodes(self) -> List[Episode]:
        return (self._dataset.episodes if self._dataset else cast(List[Episode], []))

    @episodes.setter
    def episodes(self, episodes: List[Episode]) -> None:
        assert (len(episodes) > 0), "Environment doesn't accept empty episodes list."
        assert (self._dataset is not None), "Environment must have a dataset to set episodes"
        self._dataset.episodes = episodes
        self._setup_episode_iterator()
        self._current_episode = None
        self._episode_force_changed = True
        self._episode_from_iter_on_reset = True

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def task(self) -> EmbodiedTask:
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (self._episode_start_time), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        return (self._max_episode_steps != 0 and self._max_episode_steps <= self._elapsed_steps) or (self._max_episode_seconds != 0 and self._max_episode_seconds <= self._elapsed_seconds)

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False

    def reset(self) -> Observations:
        r"""Resets the environments and returns the initial observations.

        :return: initial observations from the environment.
        """
        self._reset_stats()

        # Delete the shortest path cache of the current episode
        # Caching it for the next time we see this episode isn't really worth
        # it
        if self._current_episode is not None:
            self._current_episode._shortest_path_cache = None

        if (self._episode_iterator is not None and self._episode_from_iter_on_reset):
            self._current_episode = next(self._episode_iterator)

        # This is always set to true after a reset that way
        # on the next reset an new episode is taken (if possible)
        self._episode_from_iter_on_reset = True
        self._episode_force_changed = False

        assert self._current_episode is not None, "Reset requires an episode"
        self.reconfigure(self._config)

        observations = self.task.reset(episode=self.current_episode)
        self._task.measurements.reset_measures(
            episode=self.current_episode,
            task=self.task,
            observations=observations,
        )

        return observations

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active
        if self._past_limit():
            self._episode_over = True

        if self.episode_iterator is not None and isinstance(self.episode_iterator, EpisodeIterator):
            self.episode_iterator.step_taken()

    def step(self, action: Union[int, str, Dict[str, Any]], **kwargs) -> Observations:
        r"""Perform an action in the environment and return observations.

        :param action: action (belonging to :ref:`action_space`) to be
            performed inside the environment. Action is a name or index of
            allowed task's action and action arguments (belonging to action's
            :ref:`action_space`) to support parametrized and continuous
            actions.
        :return: observations after taking action in environment.
        """

        assert (self._episode_start_time is not None), "Cannot call step before calling reset"
        assert (self._episode_over is False), "Episode over, call reset before calling step"
        assert (not self._episode_force_changed), "Episode was changed either by setting current_episode or changing the episodes list. Call reset before stepping the environment again."

        # Support simpler interface as well
        if isinstance(action, (str, int, np.integer)):
            action = {"action": action}

        observations = self.task.step(action=action, episode=self.current_episode)

        self._task.measurements.update_measures(
            episode=self.current_episode,
            action=action,
            task=self.task,
            observations=observations,
        )

        self._update_step_stats()

        return observations

    @staticmethod
    @numba.njit
    def _seed_numba(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self._seed_numba(seed)
        self._sim.seed(seed)
        self._task.seed(seed)

    def reconfigure(self, config: "DictConfig") -> None:
        self._config = self._task.overwrite_sim_config(config, self.current_episode)

        self._sim.reconfigure(self._config.simulator, self.current_episode)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RLEnv(gym.Env):
    r"""Reinforcement Learning (RL) environment class which subclasses ``gym.Env``.

    This is a wrapper over :ref:`Env` for RL users. To create custom RL
    environments users should subclass `RLEnv` and define the following
    methods: :ref:`get_reward_range()`, :ref:`get_reward()`,
    :ref:`get_done()`, :ref:`get_info()`.

    As this is a subclass of ``gym.Env``, it implements `reset()` and
    `step()`.
    """

    _env: Env

    def __init__(self, config: "DictConfig", dataset: Optional[Dataset] = None) -> None:
        """Constructor

        :param config: config to construct :ref:`Env`
        :param dataset: dataset to construct :ref:`Env`.
        """
        if "habitat" in config:
            config = config.habitat
        self._core_env_config = config
        self._env = Env(config, dataset)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.number_of_episodes = self._env.number_of_episodes
        self.reward_range = self.get_reward_range()

    @property
    def config(self) -> "DictConfig":
        return self._core_env_config

    @property
    def habitat_env(self) -> Env:
        return self._env

    @property
    def episodes(self) -> List[Episode]:
        return self._env.episodes

    @episodes.setter
    def episodes(self, episodes: List[Episode]) -> None:
        self._env.episodes = episodes

    def current_episode(self, all_info: bool = False) -> BaseEpisode:
        r"""Returns the current episode of the environment.

        :param all_info: If true, all the information in the episode
                         will be provided. Otherwise, only episode_id
                         and scene_id will be included.
        :return: The BaseEpisode object for the current episode.
        """
        if all_info:
            return self._env.current_episode
        else:
            return BaseEpisode(
                episode_id=self._env.current_episode.episode_id,
                scene_id=self._env.current_episode.scene_id,
            )

    @profiling_wrapper.RangeContext("RLEnv.reset")
    def reset(self, *, return_info: bool = False, **kwargs) -> Union[Observations, Tuple[Observations, Dict]]:
        observations = self._env.reset()
        if return_info:
            return observations, self.get_info(observations)
        else:
            return observations

    def get_reward_range(self):
        r"""Get min, max range of reward.

        :return: :py:`[min, max]` range of reward.
        """
        raise NotImplementedError

    def get_reward(self, observations: Observations) -> Any:
        r"""Returns reward after action has been performed.

        :param observations: observations from simulator and task.
        :return: reward after performing the last action.

        This method is called inside the :ref:`step()` method.
        """
        raise NotImplementedError

    def get_done(self, observations: Observations) -> bool:
        r"""Returns boolean indicating whether episode is done after performing
        the last action.

        :param observations: observations from simulator and task.
        :return: done boolean after performing the last action.

        This method is called inside the step method.
        """
        raise NotImplementedError

    def get_info(self, observations) -> Dict[Any, Any]:
        r"""..

        :param observations: observations from simulator and task.
        :return: info after performing the last action.
        """
        raise NotImplementedError

    @profiling_wrapper.RangeContext("RLEnv.step")
    def step(self, *args, **kwargs) -> Tuple[Observations, Any, bool, dict]:
        r"""Perform an action in the environment.

        :return: :py:`(observations, reward, done, info)`
        """

        observations = self._env.step(*args, **kwargs)
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

    def render(self, mode: str = "rgb") -> np.ndarray:
        return self._env.render(mode)

    def close(self) -> None:
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
