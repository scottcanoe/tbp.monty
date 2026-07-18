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
import numpy.typing as npt
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
from tbp.monty.frameworks.models.salience.region import (
    RegionTracker,
    VoxelGrid,
)
from tbp.monty.frameworks.models.salience.return_inhibitor import ReturnInhibitor
from tbp.monty.frameworks.models.salience.segmentation.protocol import (
    SegmentationStrategy,
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
        self._region_tracker = VoxelGrid() if region_tracker is None else region_tracker

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
            + qt.rotate_vectors(agent.rotation, sensor.position),  # type: ignore
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

        rgba, depth = observation["rgba"], observation["depth"]

        salience_map = self._salience_strategy(ctx=ctx, rgba=rgba, depth=depth)
        on_object = on_object_observation(observation, salience_map)
        ior_weights = self._return_inhibitor(
            on_object.center_location, on_object.locations
        )
        salience = self._weight_salience(
            ctx,
            on_object.salience,
            ior_weights,
        )

        segmentation_mask = None
        if self._segmentation_strategy is not None:
            segmentation_mask = self._segmentation_strategy(
                ctx=ctx, rgba=rgba, depth=depth
            )
            observed_points, observed_features = self._segmented_points(
                observation, segmentation_mask, salience_map
            )
            self._region_tracker.observe(observed_points, observed_features)
            # Only build goals for locations within the tracked region.
            goal_indices = np.flatnonzero(
                self._region_tracker.contains(on_object.locations)
            )
        else:
            goal_indices = range(len(on_object.locations))

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
            for i in goal_indices
        ]

        if self._save_raw_obs and not self.is_exploring:
            info: dict = {
                "salience_map": salience_map,
                "goals": self._goals,
            }
            if segmentation_mask is not None:
                info["segmentation_mask"] = segmentation_mask
                info["region"] = self._region_tracker.state_dict()
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

    def _segmented_points(
        self,
        observation: SensorObservation,
        segmentation_mask: np.ndarray,
        salience_map: np.ndarray,
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]:
        """Extract on-object segmented points and the features aligned with them.

        Only on-object pixels contribute. The segmentation mask can bleed into the
        background, and those pixels carry max-depth locations that would otherwise
        pollute the region estimate as the sensor moves. Every returned feature is
        sampled at the same pixels as the points, so it stays row-aligned with them.

        Args:
            observation: Sensor observation containing rgba, depth, and semantic_3d.
            segmentation_mask: Binary mask indicating segmented pixels.
            salience_map: Per-pixel salience with the same grid shape as the frame.

        Returns:
            A ``(points, features)`` pair. ``points`` is an ``(num_points, 3)``
            array of world coordinates; ``features`` maps ``salience``/``depth``/
            ``rgba`` to arrays whose leading axis matches ``points``.

        """
        rgba, depth = observation["rgba"], observation["depth"]
        grid_shape = rgba.shape[:2]
        semantic_3d = observation["semantic_3d"]
        all_locations = semantic_3d[:, 0:3].reshape(grid_shape + (3,))
        on_object = semantic_3d[:, 3].reshape(grid_shape).astype(int) > 0
        seg_rows, seg_cols = np.where(np.logical_and(segmentation_mask, on_object))

        points = all_locations[seg_rows, seg_cols]
        features = {
            "salience": salience_map[seg_rows, seg_cols],
            "depth": depth[seg_rows, seg_cols],
            "rgba": rgba[seg_rows, seg_cols],
        }
        return points, features
