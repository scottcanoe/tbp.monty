# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Union

from tbp.monty.frameworks.environment_utils.transforms import (
    DepthTo3DLocations,
    MissingToMaxDepth,
)
from tbp.monty.simulators.habitat import MultiSensorAgent
from tbp.monty.simulators.habitat.environment import (
    AgentConfig,
    HabitatEnvironment,
    ObjectConfig,
)

# Path to dataset
DATASET_DIR = (
    Path(os.environ.get("MONTY_DATA", "~/tbp/data")).expanduser()
    / "compositional_objects"
)



@dataclass
class MountConfig:
    """Patch and view-finder mount config with 256x256 view-finder resolution."""

    agent_id: Union[str, None] = "agent_id_0"
    sensor_ids: Union[List[str], None] = field(
        default_factory=lambda: ["patch", "view_finder"]
    )
    height: Union[float, None] = 0.0
    position: List[Union[int, float]] = field(default_factory=lambda: [0.0, 1.5, 0.2])
    resolutions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[64, 64], [512, 512]]
    )
    positions: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    rotations: List[List[Union[int, float]]] = field(
        default_factory=lambda: [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    semantics: List[List[Union[int, float]]] = field(
        default_factory=lambda: [False, False]
    )
    zooms: List[float] = field(default_factory=lambda: [10.0, 1.0])


@dataclass
class EnvInitArgs:
    agents: List[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, MountConfig().__dict__)
        ]
    )
    objects: List[ObjectConfig] = field(
        default_factory=lambda: [ObjectConfig("cube", position=(0.0, 1.5, -0.1))]
    )
    scene_id: Union[int, None] = field(default=None)
    seed: int = field(default=42)
    data_path: str = str(DATASET_DIR)


@dataclass
class DatasetArgs:
    env_init_func: Callable = field(default=HabitatEnvironment)
    env_init_args: Dict = field(default_factory=lambda: EnvInitArgs().__dict__)
    transform: Union[Callable, list, None] = None
    rng: Union[Callable, None] = None

    def __post_init__(self):
        agent_args = self.env_init_args["agents"][0].agent_args
        self.transform = [
            MissingToMaxDepth(agent_id=agent_args["agent_id"], max_depth=1),
            DepthTo3DLocations(
                agent_id=agent_args["agent_id"],
                sensor_ids=agent_args["sensor_ids"],
                resolutions=agent_args["resolutions"],
                world_coord=True,
                zooms=agent_args["zooms"],
                get_all_points=True,
                use_semantic_sensor=False,
            ),
        ]

"""
How to use...

# Probably copy a pretraining benchmark config?
pretrain_config = copy.deepcopy(some_pretrain_config)

# Update with dataset args, and specify dataloader args.
pretrain_config["dataset_args"] = DatasetArgs()

"""
