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
from tbp.monty.frameworks.models.salience.return_inhibitor import ReturnInhibitor
from tbp.monty.frameworks.models.salience.segmentation.protocol import (
    SegmentationStrategy,
)
from tbp.monty.frameworks.models.salience.segmentation.region_tracker import (
    RegionTracker,
)
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
        segmentation_strategy: SegmentationStrategy | None = None,
        region_tracker: RegionTracker | None = None,
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
        # Accumulates observed points into an estimate of the object's region in
        # space; owns all voxel/region representation and merging details.
        self._region_tracker = (
            RegionTracker() if region_tracker is None else region_tracker
        )

        self._goals: list[Goal] = []
        # TODO: Goes away once experiment code is extracted
        self.is_exploring = False

        self._segmentation_strategy = segmentation_strategy

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
            + qt.rotate_vectors(agent.rotation, sensor.position),  # type: ignore[arg-type]
            rotation=agent.rotation * sensor.rotation,
        )

    def reset(self) -> None:
        self._goals.clear()
        self._return_inhibitor.reset()
        self._snapshot_telemetry.reset()
        self.is_exploring = False
        self._region_tracker.reset()

    def propose_goals(self) -> list[Goal]:
        return self._goals

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

        rgba = observation["rgba"]
        depth = observation["depth"]
        semantic_3d = observation["semantic_3d"]
        image_shape = depth.shape
        locations_map = semantic_3d[:, 0:3].reshape(image_shape + (3,))
        on_object_mask = semantic_3d[:, 3].reshape(image_shape) > 0

        # Compute salience map and do weighting.
        salience_map = self._salience_strategy(ctx=ctx, rgba=rgba, depth=depth)
        center_row, center_col = image_shape[0] // 2, image_shape[1] // 2
        if on_object_mask[center_row, center_col]:
            center_location = locations_map[center_row, center_col]
        else:
            center_location = None

        ior_weights = self._return_inhibitor(
            center_location,
            locations_map[on_object_mask],
        )
        weighted_salience = self._weight_salience(
            ctx,
            salience_map[on_object_mask],
            ior_weights,
        )
        weighted_salience_map = np.zeros_like(salience_map)
        weighted_salience_map[on_object_mask] = weighted_salience

        goal_locations = locations_map[on_object_mask]
        goal_salience = weighted_salience

        info = {"salience_map": salience_map}  # telemetry

        if self._segmentation_strategy is not None:
            segmentation_map = self._segmentation_strategy(
                ctx=ctx, rgba=rgba, depth=depth
            )

            # Update region tracker with points that are both on-object and
            # within the segmented region.
            surface_map = segmentation_map * on_object_mask.astype(float)
            surface_rows, surface_cols = np.where(surface_map > 0.0)
            surface_locations = locations_map[surface_rows, surface_cols]
            surface_salience = weighted_salience_map[surface_rows, surface_cols]

            # First step the region tracker. Then filter out points using it.
            self._region_tracker.step(
                surface_locations, features={"confidence": surface_salience}
            )
            on_surface = self._region_tracker.contains_points(surface_locations)
            goal_locations = surface_locations[on_surface]
            goal_salience = surface_salience[on_surface]

            info["segmentation"] = {
                "segmentation_map": segmentation_map,
                "surface_map": surface_map,
                "surface_locations": surface_locations,
                "surface_salience": surface_salience,
            }
            info["region"] = {
                "voxel_size": self._region_tracker.voxel_size,
                "voxel_grid": self._region_tracker.grid.copy(),
            }

        # Finally, build goals.
        self._goals = [
            Goal(
                location=goal_locations[i],
                morphological_features=None,
                non_morphological_features=None,
                confidence=goal_salience[i],
                use_state=True,
                sender_id=self._sensor_module_id,
                sender_type="SM",
                goal_tolerances=None,
            )
            for i in range(len(goal_locations))
        ]
        info["goals"] = self._goals
        if self._save_raw_obs and not self.is_exploring:
            self._snapshot_telemetry.raw_observation(
                observation,
                self.state.rotation,
                self.state.position,  # type: ignore[arg-type]
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
