# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import numpy as np
import quaternion as qt

from tbp.monty.cmp import Goal
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import (
    SensorModule,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.models.salience.on_object_observation import (
    on_object_observation,
)
from tbp.monty.frameworks.models.salience.return_inhibitor import ReturnInhibitor
from tbp.monty.frameworks.models.salience.segmentation import SlicMerge
from tbp.monty.frameworks.models.salience.strategies import (
    SalienceStrategy,
    Uniform,
)
from tbp.monty.frameworks.models.sensor_modules import SnapshotTelemetry
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.memento import Memento

__all__ = ["SalienceSM"]


class SalienceSM(SensorModule):
    def __init__(
        self,
        sensor_module_id: str,
        save_raw_obs: bool = False,
        salience_strategy: SalienceStrategy | None = None,
        return_inhibitor: ReturnInhibitor | None = None,
        snapshot_telemetry: SnapshotTelemetry | None = None,
        voxel_size: float = 0.01,
        voxel_lifetime: int = 3,
    ) -> None:
        self._sensor_module_id = sensor_module_id
        self._save_raw_obs = save_raw_obs
        self._salience_strategy = (
            Uniform() if salience_strategy is None else salience_strategy
        )
        self._return_inhibitor = (
            ReturnInhibitor() if return_inhibitor is None else return_inhibitor
        )
        self._snapshot_telemetry = (
            SnapshotTelemetry() if snapshot_telemetry is None else snapshot_telemetry
        )
        self._voxel_size = voxel_size
        self._voxel_lifetime = voxel_lifetime

        self._goals: list[Goal] = []
        # TODO: Goes away once experiment code is extracted
        self.is_exploring = False

        self._segmenter = SlicMerge()
        self._voxel_grid: set[tuple[int, int, int]] = set()

    @property
    def sensor_module_id(self) -> str:
        return self._sensor_module_id

    def state_dict(self) -> Memento:
        return self._snapshot_telemetry.state_dict()

    def update_state(self, agent: AgentState) -> None:
        """Update information about the sensor's location and rotation."""
        sensor = agent.sensors[SensorID(self.sensor_module_id)]
        self.state = SensorState(
            position=agent.position
            + qt.rotate_vectors(agent.rotation, sensor.position),  # type: ignore
            rotation=agent.rotation * sensor.rotation,
        )

    def step(
        self,
        ctx: RuntimeContext,
        observation: SensorObservation,
        motor_only_step: bool = False,
    ) -> None:
        """Generate goal for the current step.

        If `motor_only_step` is True, this method will return without using the
        salience strategy, stepping the return inhibitor, or modifying `self._goals`
        in any way.

        Args:
            ctx: The runtime context.
            observation: Sensor observation.
            motor_only_step: Whether the current step is a motor-only step.

        """

        if motor_only_step:
            return

        salience_map = self._salience_strategy(
            ctx=ctx,
            rgba=observation["rgba"],
            depth=observation["depth"],  # type: ignore
        )
        rgb = observation["rgba"][:, :, :3]
        segmentation_mask = self._segmenter.segment(rgb)  # type: ignore

        # Build voxel grid from segmentation mask
        seg_voxels = self._build_voxel_grid(observation, segmentation_mask)

        # TODO: growing/shrinking voxel grid over time.
        self._voxel_grid = seg_voxels

        on_object = on_object_observation(observation, salience_map)
        ior_weights = self._return_inhibitor(
            on_object.center_location, on_object.locations
        )
        salience = self._weight_salience(ctx, on_object.salience, ior_weights)

        # Filter goals to only include those within segmented voxels
        in_voxel_grid = self._in_voxel_grid(on_object.locations)
        self._goals = [
            Goal(
                location=on_object.locations[i],
                morphological_features=None,
                non_morphological_features=None,
                confidence=salience[i],
                use_state=True,
                sender_id=self._sensor_module_id,
                sender_type="SM",
                goal_tolerances=None,
            )
            for i in np.flatnonzero(in_voxel_grid)
        ]

        if self._save_raw_obs and not self.is_exploring:
            info = {
                "salience_map": salience_map,
                "segmentation_mask": segmentation_mask,
                "goals": self._goals,
            }
            self._snapshot_telemetry.raw_observation(
                observation,
                self.state.rotation,
                self.state.position,  # type: ignore
                info=info,
            )

    def _weight_salience(
        self,
        ctx: RuntimeContext,
        salience: np.ndarray,
        ior_weights: np.ndarray,
    ) -> np.ndarray:
        weighted_salience = self._decay_salience(salience, ior_weights)

        weighted_salience = self._randomize_salience(ctx, weighted_salience)

        return self._normalize_salience(weighted_salience)

    def _decay_salience(
        self, salience: np.ndarray, ior_weights: np.ndarray
    ) -> np.ndarray:
        decay_factor = 0.75
        return salience - decay_factor * ior_weights

    def _randomize_salience(
        self, ctx: RuntimeContext, weighted_salience: np.ndarray
    ) -> np.ndarray:
        randomness_factor = 0.05
        weighted_salience += ctx.rng.normal(
            loc=0, scale=randomness_factor, size=weighted_salience.shape[0]
        )
        return weighted_salience

    def _normalize_salience(self, weighted_salience: np.ndarray) -> np.ndarray:
        if weighted_salience.size == 0:
            return weighted_salience

        min_ = weighted_salience.min()
        max_ = weighted_salience.max()
        scale = max_ - min_
        if np.isclose(scale, 0):
            return np.clip(weighted_salience, 0, 1)

        return (weighted_salience - min_) / scale

    def _build_voxel_grid(
        self, observation: SensorObservation, segmentation_mask: np.ndarray
    ) -> set[tuple[int, int, int]]:
        """Build a set of occupied voxels from segmentation mask.

        Args:
            observation: Sensor observation containing semantic_3d data.
            segmentation_mask: Binary mask indicating segmented pixels.

        Returns:
            Set of voxel indices as (x, y, z) tuples.

        """
        grid_shape = observation["rgba"].shape[:2]
        semantic_3d = observation["semantic_3d"]
        all_locations = semantic_3d[:, 0:3].reshape(grid_shape + (3,))
        seg_rows, seg_cols = np.where(segmentation_mask)
        seg_locations = all_locations[seg_rows, seg_cols]

        if len(seg_locations) == 0:
            return set()

        voxel_indices = (seg_locations / self._voxel_size).astype(int)
        return set(map(tuple, voxel_indices))

    def _in_voxel_grid(self, locations: np.ndarray) -> np.ndarray:
        """Check which locations fall within an occupied voxel.

        Args:
            locations: 3D coordinates as array of shape (num_locations, 3).

        Returns:
            Boolean array of shape (num_locations,) that is True wherever the
            corresponding location's voxel is in the grid.

        """
        if len(self._voxel_grid) == 0:
            return np.zeros(len(locations), dtype=bool)

        voxel_indices = (locations / self._voxel_size).astype(int)
        return np.fromiter(
            (tuple(idx) in self._voxel_grid for idx in voxel_indices),
            dtype=bool,
            count=len(voxel_indices),
        )

    def reset(self) -> None:
        self._goals.clear()
        self._return_inhibitor.reset()
        self._snapshot_telemetry.reset()
        self.is_exploring = False
        self._voxel_grid.clear()

    def propose_goals(self) -> list[Goal]:
        return self._goals
