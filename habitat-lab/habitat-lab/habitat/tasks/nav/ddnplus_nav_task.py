# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union, Dict
import attr
import numpy as np
from gym import spaces
import quaternion
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.core.dataset import BaseEpisode, Dataset, Episode
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes, Simulator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.utils import not_none_validator
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.utils.common import compute_instance_bounding_boxes, bbox_iou, find_shortest_path, sample_object_navigable_point
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
    HeadingSensor,
)
from omegaconf import DictConfig

try:
    from habitat.datasets.ddnplus_nav.ddnplus_nav_dataset import (
        DDNPlusNavDatasetV1, )
except ImportError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig
import random
import json


@registry.register_measure
class DDNPlusBasicSuccess(Measure):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "ddnplus_basic_success"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._success_distance = self._config.ddnplus_success_distance
        self.find_times = self._config.ddnplus_find_times

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        # task.measurements.check_measure_dependencies(
        #     self.uuid, [DDNPlusDistanceToGoal.cls_uuid]
        # )
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        # distance_to_target = task.measurements.measures[
        #     DDNPlusDistanceToGoal.cls_uuid
        # ].get_metric()
        if self.find_times < task.find_called_times:
            return 0.0
        if (hasattr(task, "is_stop_called") and task.is_stop_called  # type: ignore
                # and distance_to_target < self._success_distance
            ):

            for solution in episode.basic_solution:
                min_len = 0
                solution_idx = [task.object_name_to_idx[solution_name.split(".n")[0]] for solution_name in solution]
                for element in list(set(task.find_obj_list)):
                    if element in solution_idx:
                        min_len += 1

                self._metric = max(min_len / len(solution), self._metric)
            t = 1
        else:

            self._metric = 0.0


@registry.register_measure
class DDNPlusPreferenceSuccess(Measure):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "ddnplus_preference_success"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._success_distance = self._config.ddnplus_success_distance
        self.find_times = self._config.ddnplus_find_times

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        # task.measurements.check_measure_dependencies(
        #     self.uuid, [DDNPlusDistanceToGoal.cls_uuid]
        # )
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        # distance_to_target = task.measurements.measures[
        #     DDNPlusDistanceToGoal.cls_uuid
        # ].get_metric()
        if self.find_times < task.find_called_times:
            return 0.0
        if (hasattr(task, "is_stop_called") and task.is_stop_called  # type: ignore
                # and distance_to_target < self._success_distance
            ):

            for solution in episode.preferred_solution:
                min_len = 0
                solution_idx = [task.object_name_to_idx[solution_name.split(".n")[0]] for solution_name in solution]
                for element in list(set(task.find_obj_list)):
                    if element in solution_idx:
                        min_len += 1

                self._metric = max(min_len / len(solution), self._metric)
            t = 1
        else:

            self._metric = 0.0


@registry.register_measure
class DDNPlusDistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "ddnplus_distance_to_goal"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._task = kwargs['task']
        current_position = self._sim.get_agent_state().position
        self.scene_id = self._sim.curr_scene_name.split(".scene_instance")[0]
        scene_name = self._sim.curr_scene_name.split(".scene_instance")[0]
        if os.path.exists("habitat-lab/data/datasets/ddnplus/hssd-hab_v0.2.5/scene_object_navigable_point/{}.json".format(scene_name)):
            with open("habitat-lab/data/datasets/ddnplus/hssd-hab_v0.2.5/scene_object_navigable_point/{}.json".format(scene_name), 'r') as f:
                self._object_to_positions = json.load(f)
        else:
            self._object_to_positions = sample_object_navigable_point(self._sim, self._task, self._sim.get_agent_state().position)
            self._object_to_positions_list = {}
            for k, v in self._object_to_positions.items():
                self._object_to_positions_list[k] = [list(i) for i in v]
            with open("habitat-lab/data/datasets/ddnplus/hssd-hab_v0.2.5/scene_object_navigable_point/{}.json".format(scene_name), 'w') as f:
                json.dump(self._object_to_positions_list, f, indent=4)

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):

        self._previous_position = None
        if self.scene_id != self._sim.curr_scene_name.split(".scene_instance")[0] or self.scene_id is None:
            scene_name = self._sim.curr_scene_name.split(".scene_instance")[0]
            if os.path.exists("habitat-lab/data/datasets/ddnplus/hssd-hab_v0.2.5/scene_object_navigable_point/{}.json".format(scene_name)):
                with open("habitat-lab/data/datasets/ddnplus/hssd-hab_v0.2.5/scene_object_navigable_point/{}.json".format(scene_name), 'r') as f:
                    self._object_to_positions = json.load(f)
            else:
                self._object_to_positions = sample_object_navigable_point(self._sim, self._task, self._sim.get_agent_state().position)
                self._object_to_positions_list = {}
                for k, v in self._object_to_positions.items():
                    self._object_to_positions_list[k] = [list(i) for i in v]
                with open("habitat-lab/data/datasets/ddnplus/hssd-hab_v0.2.5/scene_object_navigable_point/{}.json".format(scene_name), 'w') as f:
                    json.dump(self._object_to_positions_list, f, indent=4)
        self.scene_id = episode.scene_id
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(self, episode: NavigationEpisode, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        self._metric = 99999
        self.target_cateogry = None
        # for solution in episode.basic_solution:
        #     if any(isinstance(element, list) for element in solution):
        #         raise ValueError("solution contains a list, which is not allowed.")
        #     if any(element not in self._object_to_positions.keys() for element in solution):
        #         continue
        #     target_cateogry = {k: self._object_to_positions[k] for k in solution}
        #     path, min_path_distance = find_shortest_path(current_position, target_cateogry, self._sim)
        #     if min_path_distance < self._metric:
        #         self._metric = min_path_distance
        #         self.min_path = path
        #         self.target_cateogry = target_cateogry
        for solution in episode.preferred_solution:
            if any(isinstance(element, list) for element in solution):
                raise ValueError("solution contains a list, which is not allowed.")
            if any(element not in self._object_to_positions.keys() for element in solution):
                continue
            target_cateogry = {k: self._object_to_positions[k] for k in solution}
            path, min_path_distance = find_shortest_path(current_position, target_cateogry, self._sim)
            if min_path_distance < self._metric:
                self._metric = min_path_distance
                self.min_path = path
                self.target_cateogry = target_cateogry
        t = 1


@registry.register_measure
class DDNPlusDistanceToOneGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "ddnplus_distance_to_one_goal"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._task = kwargs['task']
        current_position = self._sim.get_agent_state().position
        self.scene_id = self._sim.curr_scene_name.split(".scene_instance")[0]
        scene_name = self._sim.curr_scene_name.split(".scene_instance")[0]
        if os.path.exists("habitat-lab/data/datasets/ddnplus/hssd-hab_v0.2.5/scene_object_navigable_point/{}.json".format(scene_name)):
            with open("habitat-lab/data/datasets/ddnplus/hssd-hab_v0.2.5/scene_object_navigable_point/{}.json".format(scene_name), 'r') as f:
                self._object_to_positions = json.load(f)
        else:
            self._object_to_positions = sample_object_navigable_point(self._sim, self._task, self._sim.get_agent_state().position)
            self._object_to_positions_list = {}
            for k, v in self._object_to_positions.items():
                self._object_to_positions_list[k] = [list(i) for i in v]
            with open("habitat-lab/data/datasets/ddnplus/hssd-hab_v0.2.5/scene_object_navigable_point/{}.json".format(scene_name), 'w') as f:
                json.dump(self._object_to_positions_list, f, indent=4)
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = None
        if self.scene_id != episode.scene_id.split(".scene_instance")[0] or self.scene_id is None:
            scene_name = episode.scene_id.split('/')[-1].split(".scene_instance")[0]
            if os.path.exists("habitat-lab/data/datasets/ddnplus/hssd-hab_v0.2.5/scene_object_navigable_point/{}.json".format(scene_name)):
                with open("habitat-lab/data/datasets/ddnplus/hssd-hab_v0.2.5/scene_object_navigable_point/{}.json".format(scene_name), 'r') as f:
                    self._object_to_positions = json.load(f)
            else:
                self._object_to_positions = sample_object_navigable_point(self._sim, self._task, self._sim.get_agent_state().position)
                self._object_to_positions_list = {}
                for k, v in self._object_to_positions.items():
                    self._object_to_positions_list[k] = [list(i) for i in v]
                with open("habitat-lab/data/datasets/ddnplus/hssd-hab_v0.2.5/scene_object_navigable_point/{}.json".format(scene_name), 'w') as f:
                    json.dump(self._object_to_positions_list, f, indent=4)
        current_position = self._sim.get_agent_state().position
        self._metric = 99999
        self.target_cateogry = None
        self.min_obj_position = None
        # for solution in episode.basic_solution:
        #     if any(element not in self._object_to_positions.keys() for element in solution):
        #         continue
        #     target_cateogry = {k: self._object_to_positions[k] for k in solution}
        #     for k, v in target_cateogry.items():
        #         for position in v:
        #             dis = self._sim.geodesic_distance(current_position, position[0])
        #             if self._metric > dis:
        #                 self._metric = dis
        #                 self.min_obj_position = position
        #                 self.min_obj_name = k
        for solution in episode.preferred_solution:
            if any(element not in self._object_to_positions.keys() for element in solution):
                continue
            target_cateogry = {k: self._object_to_positions[k] for k in solution}
            for k, v in target_cateogry.items():
                for position in v:
                    dis = self._sim.geodesic_distance(current_position, position[0])
                    if self._metric > dis:
                        self._metric = dis
                        self.min_obj_position = position
                        self.min_obj_name = k

    def update_metric(self, episode: NavigationEpisode, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        self._metric = self._sim.geodesic_distance(current_position, self.min_obj_position[0])


@registry.register_measure
class DDNPlusDistanceToOneGoalReward(Measure):
    """
    The measure calculates a reward based on the distance towards the goal.
    The reward is `- (new_distance - previous_distance)` i.e. the
    decrease of distance to the goal.
    """

    cls_uuid: str = "ddnplus_distance_to_one_goal_reward"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._previous_distance: Optional[float] = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [DDNPlusDistanceToOneGoal.cls_uuid])
        self._previous_distance = task.measurements.measures[DDNPlusDistanceToOneGoal.cls_uuid].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        distance_to_target = task.measurements.measures[DDNPlusDistanceToOneGoal.cls_uuid].get_metric()
        self._metric = -(distance_to_target - self._previous_distance)
        self._previous_distance = distance_to_target


@registry.register_measure
class DDNPlusBasicSPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    The measure depends on Distance to Goal measure and Success measure
    to improve computational
    performance for sophisticated goal areas.
    """

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any):
        self._previous_position: Union[None, np.ndarray, List[float]] = None
        self._start_end_episode_distance: Optional[float] = None
        self._agent_episode_distance: Optional[float] = None
        self._episode_view_points: Optional[List[Tuple[float, float, float]]] = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "ddnplus_basic_spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [DDNPlusDistanceToGoal.cls_uuid, DDNPlusBasicSuccess.cls_uuid])

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[DDNPlusDistanceToGoal.cls_uuid].get_metric()
        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        ep_success = task.measurements.measures[DDNPlusBasicSuccess.cls_uuid].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(current_position, self._previous_position)

        self._previous_position = current_position

        self._metric = ep_success * (self._start_end_episode_distance / max(self._start_end_episode_distance, self._agent_episode_distance))


@registry.register_measure
class DDNPlusPreferenceSPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    The measure depends on Distance to Goal measure and Success measure
    to improve computational
    performance for sophisticated goal areas.
    """

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any):
        self._previous_position: Union[None, np.ndarray, List[float]] = None
        self._start_end_episode_distance: Optional[float] = None
        self._agent_episode_distance: Optional[float] = None
        self._episode_view_points: Optional[List[Tuple[float, float, float]]] = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "ddnplus_preference_spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [DDNPlusDistanceToGoal.cls_uuid, DDNPlusBasicSuccess.cls_uuid])

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[DDNPlusDistanceToGoal.cls_uuid].get_metric()
        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        ep_success = task.measurements.measures[DDNPlusPreferenceSuccess.cls_uuid].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(current_position, self._previous_position)

        self._previous_position = current_position

        self._metric = ep_success * (self._start_end_episode_distance / max(self._start_end_episode_distance, self._agent_episode_distance))


@registry.register_measure
class DDNPlusSoftSPL(DDNPlusBasicSPL):
    r"""Soft SPL

    Similar to spl with a relaxed soft-success criteria. Instead of a boolean
    success is now calculated as 1 - (ratio of distance covered to target).
    """

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "ddnplus_soft_spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [DDNPlusDistanceToGoal.cls_uuid])

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[DDNPlusDistanceToGoal.cls_uuid].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        distance_to_target = task.measurements.measures[DDNPlusDistanceToGoal.cls_uuid].get_metric()

        ep_soft_success = max(0, (1 - distance_to_target / self._start_end_episode_distance))

        self._agent_episode_distance += self._euclidean_distance(current_position, self._previous_position)

        self._previous_position = current_position

        self._metric = ep_soft_success * (self._start_end_episode_distance / max(self._start_end_episode_distance, self._agent_episode_distance))


@registry.register_measure
class DDNPlusDistanceToGoalReward(Measure):
    """
    The measure calculates a reward based on the distance towards the goal.
    The reward is `- (new_distance - previous_distance)` i.e. the
    decrease of distance to the goal.
    """

    cls_uuid: str = "ddnplus_distance_to_goal_reward"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._previous_distance: Optional[float] = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [DDNPlusDistanceToGoal.cls_uuid])
        self._previous_distance = task.measurements.measures[DDNPlusDistanceToGoal.cls_uuid].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        distance_to_target = task.measurements.measures[DDNPlusDistanceToGoal.cls_uuid].get_metric()
        self._metric = -(distance_to_target - self._previous_distance)
        self._previous_distance = distance_to_target


@attr.s(auto_attribs=True, kw_only=True)
class DDNPlusGoalNavEpisode():
    r"""DDNPlusGoal Navigation Episode

    :param object_category: Category of the obect
    """
    task_instruction: Optional[str] = None
    basic_demand_instruction: Optional[str] = None
    preferred_demand_instruction: Optional[str] = None

    basic_solution: Optional[List[List]] = None
    preferred_solution: Optional[List[List]] = None

    basic_attribute: Optional[str] = None
    preferred_attribute: Optional[str] = None
    scene_dataset_config: Optional[str] = None

    # found_basic_objects: Optional[List[str]] = None
    # found_preference_objects: Optional[List[str]] = None

    start_rotation: Optional[List[float]] = None
    start_position: Optional[List[float]] = None

    scene_ids: Optional[Dict[str, List[str]]] = None
    info = {}

    # @property
    # def goals_key(self) -> str:
    #     r"""The key to retrieve the goals"""
    #     return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@attr.s(auto_attribs=True)
class DDNPlusViewLocation:
    r"""ObjectViewLocation provides information about a position around an object goal
    usually that is navigable and the object is visible with specific agent
    configuration that episode's dataset was created.
     that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        agent_state: navigable AgentState with a position and a rotation where
        the object is visible.
        iou: an intersection of a union of the object and a rectangle in the
        center of view. This metric is used to evaluate how good is the object
        view form current position. Higher iou means better view, iou equals
        1.0 if whole object is inside of the rectangle and no pixel inside
        the rectangle belongs to anything except the object.
    """
    agent_state: AgentState
    iou: Optional[float]


@attr.s(auto_attribs=True, kw_only=True)
class DDNPlusGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    # object_id: str = attr.ib(default=None, validator=not_none_validator)
    # object_name: Optional[str] = None
    # object_name_id: Optional[int] = None
    # object_category: Optional[str] = None
    # room_id: Optional[str] = None
    # room_name: Optional[str] = None
    # view_points: Optional[List[ObjectViewLocation]] = None
    task_instruction: Optional[str] = None
    basic_demand: Optional[str] = None
    preference: Optional[str] = None

    basic_object_list: Optional[List[str]] = None
    basic_object_list_id: Optional[List[int]] = None

    preference_object_list: Optional[List[str]] = None
    preference_object_list_id: Optional[List[int]] = None


class StringSpace(spaces.Space):

    def __init__(self, strings):
        super().__init__((), np.str_)
        self.strings = strings

    def sample(self):
        return self.strings

    def contains(self, x):
        return self.strings

    def __repr__(self):
        return "StringSpace({})".format(self.strings)


@registry.register_sensor
class DDNPlusGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            goal_spec that specifies which id use for goal specification,
            goal_spec_max_val the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "ddnplusgoal"
    observation_space: str

    def __init__(
        self,
        sim,
        config: "DictConfig",
        dataset: "DDNPlusNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):

        return StringSpace("DummySpace")

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: DDNPlusGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        if len(episode.task_instruction) == 0:
            logger.error(f"No goal specified for episode {episode.episode_id}.")
            return None
        else:
            return (episode.task_instruction, episode.basic_attribute, episode.preferred_attribute)


@registry.register_sensor(name="DDNPlusCompassSensor")
class DDNPlusEpisodicCompassSensor(HeadingSensor):
    r"""The agents heading in the coordinate frame defined by the episode,
    theta=0 is defined by the agents state at t=0
    """
    cls_uuid: str = "ddnpluscompass"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(self, observations, episode, *args: Any, **kwargs: Any):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        if isinstance(rotation_world_agent, quaternion.quaternion):
            return self._quat_to_xy_heading(rotation_world_agent.inverse() * rotation_world_start)
        else:
            raise ValueError("Agent's rotation was not a quaternion")


@registry.register_task_action
class FindAction(SimulatorTaskAction):
    name: str = "find"

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.find_called_times = 0  # type: ignore
        task.find_obj_list = []

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.find_called_times += 1  # type: ignore
        # bbox = kwargs['bbox']
        semantic_map = kwargs['semantic_map'][:, :, 0]
        depth_map = kwargs['depth_map'][:, :, 0]
        ddnplus_success_distance = task.measurements.measures['ddnplus_basic_success']._config["ddnplus_success_distance"]
        gt_bboxes = compute_instance_bounding_boxes(semantic_map, depth_map, depth_threshold=ddnplus_success_distance)
        # IoU = bbox_iou(bbox, gt_bboxes)
        # max_IoU = -1
        # max_IoU_idx = -1
        # for object_id in IoU.keys():
        #     if max(IoU[object_id]) > max_IoU:
        #         max_IoU = max(IoU[object_id])
        #         max_IoU_idx = object_id
        # if max_IoU in task._dataset.id_to_task_category_id:
        #     task.find_obj_list.append(task._dataset.id_to_task_category_id[max_IoU_idx])
        # for object_id in IoU.keys():
        #     task.find_obj_list.append(task._dataset.id_to_task_category_id[object_id])
        task.find_obj_list.extend(list(gt_bboxes.keys()))
        pass


@registry.register_task_action
class LeaveAction(SimulatorTaskAction):
    name: str = "leave"

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        pass

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        pass


@registry.register_task(name="DDNPlusNav-v1")
class DDNPlusNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """

    def __init__(self, config: DictConfig, sim: Simulator, dataset: Dataset = None) -> None:
        super().__init__(config, sim, dataset)
        with open("habitat-lab/data/datasets/ddnplus/hssd-hab_v0.2.5/rotation.json", 'r') as f:
            self.rotation = json.load(f)

    def overwrite_sim_config(self, config: Any, episode: Episode) -> Any:
        with read_write(config):
            config.simulator.scene = episode.scene_id
            if episode.start_position is None:

                agent_config = get_agent_config(config.simulator)
                agent_config.is_set_start_state = False
            else:
                agent_config = get_agent_config(config.simulator)
                agent_config.start_position = episode.start_position
                agent_config.start_rotation = [float(k) for k in episode.start_rotation]
                agent_config.is_set_start_state = True
        return config
