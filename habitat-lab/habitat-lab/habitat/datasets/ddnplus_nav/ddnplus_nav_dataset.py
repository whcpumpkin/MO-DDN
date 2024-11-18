#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.ddnplus_nav_task import (
    DDNPlusGoal,
    DDNPlusGoalNavEpisode,
    DDNPlusViewLocation,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_dataset(name="DDNPlusNav-v1")
class DDNPlusNavDatasetV1(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads DDNPlus Navigation dataset."""
    category_to_task_category_id: Dict[str, int]
    category_to_scene_annotation_category_id: Dict[str, int]
    episodes: List[DDNPlusGoalNavEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json"
    goals_by_category: Dict[str, Sequence[DDNPlusGoal]]

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = {}
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0]["object_category"]
            ep = DDNPlusGoalNavEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category

        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            goals = self.goals_by_category[self.episodes[i].goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            self.episodes[i].goals = goals

        return result

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.goals_by_category = {}
        super().__init__(config)
        self.episodes = list(self.episodes)

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> DDNPlusGoal:
        g = DDNPlusGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = DDNPlusViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    def _load_from_file(self, fname: str, scenes_dir: str) -> None:
        """
        Load the data from a file into `self.episodes`.
        """

        self.from_json(fname, scenes_dir=scenes_dir)

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        with open(json_str) as f:
            deserialized = json.load(f)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized["category_to_task_category_id"]
            self.category_to_scene_annotation_category_id = deserialized["category_to_task_category_id"]
            self.id_to_task_category_id = {v: k for k, v in self.category_to_task_category_id.items()}

        if "category_to_mp3d_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized["category_to_mp3d_category_id"]

        assert set(self.category_to_task_category_id.keys()) == set(self.category_to_scene_annotation_category_id.keys()), "category_to_task and category_to_mp3d must have the same keys"

        if isinstance(deserialized, list) is False:
            return

        for i, episode in enumerate(deserialized):
            # if "episode_id" not in episode:
            #     episode["episode_id"]=i

            if "scene_dataset_config" not in episode:
                episode["scene_dataset_config"] = "data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"
            episode = DDNPlusGoalNavEpisode(**episode)
            episode.episode_id = str(i)

            if scenes_dir is not None:
                # if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                #     episode.scene_id = episode.scene_id[
                #         len(DEFAULT_SCENE_PATH_PREFIX) :
                #     ]
                for key in episode.scene_ids.keys():
                    episode.scene_ids[key] = [os.path.join(scenes_dir, scene_id) for scene_id in episode.scene_ids[key]]

            # episode.goals = self.goals_by_category[episode.goals_key]

            # if episode.shortest_paths is not None:
            #     for path in episode.shortest_paths:
            #         for p_index, point in enumerate(path):
            #             if point is None or isinstance(point, (int, str)):
            #                 point = {
            #                     "action": point,
            #                     "rotation": None,
            #                     "position": None,
            #                 }

            #             path[p_index] = ShortestPathPoint(**point)

            self.episodes.append(episode)  # type: ignore [attr-defined]
