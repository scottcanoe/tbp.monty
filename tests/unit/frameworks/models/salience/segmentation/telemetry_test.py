# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""The segmentation telemetry a SalienceSM records, end to end."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
import quaternion as qt

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.motor_system_state import SensorState
from tbp.monty.frameworks.models.salience.segmentation.region_tracker import (
    RegionTracker,
)
from tbp.monty.frameworks.models.salience.sensor_module import SalienceSM

PATCH = 8
VOXEL_SIZE = 0.02


def segment_everything(ctx, rgba, depth=None, locations=None) -> np.ndarray:  # noqa: ARG001
    """Segment the whole frame.

    Returns:
        An all-ones mask covering the frame.

    """
    return np.ones(rgba.shape[:2], dtype=np.uint8)


class SegmentationTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        grid = np.indices((PATCH, PATCH)).reshape(2, -1).T * VOXEL_SIZE
        self.locations = np.concatenate([grid, np.zeros((PATCH * PATCH, 1))], axis=1)
        semantic_3d = np.concatenate(
            [self.locations, np.ones((PATCH * PATCH, 1))], axis=1
        )
        self.observation = {
            "rgba": np.full((PATCH, PATCH, 4), 200, dtype=np.uint8),
            "depth": np.full((PATCH, PATCH), 0.3),
            "semantic_3d": semantic_3d,
        }
        self.tracker = RegionTracker(voxel_size=VOXEL_SIZE)
        self.sensor_module = SalienceSM(
            sensor_module_id="test",
            save_raw_obs=True,
            salience_strategy=MagicMock(return_value=np.full((PATCH, PATCH), 0.5)),
            return_inhibitor=MagicMock(return_value=np.zeros(PATCH * PATCH)),
            snapshot_telemetry=MagicMock(),
            segmentation_strategy=segment_everything,
            region_tracker=self.tracker,
        )
        self.sensor_module.state = SensorState(
            position=np.zeros(3), rotation=qt.quaternion(1, 0, 0, 0)
        )

    def step(self) -> dict:
        """Run one step and return the telemetry info it recorded.

        Returns:
            The ``info`` mapping passed to the snapshot telemetry.

        """
        self.sensor_module.step(
            RuntimeContext(rng=np.random.RandomState(0)), self.observation
        )
        raw_observation = self.sensor_module._snapshot_telemetry.raw_observation
        return raw_observation.call_args.kwargs["info"]

    def test_records_the_salience_map(self) -> None:
        info = self.step()
        self.assertEqual(info["salience_map"].shape, (PATCH, PATCH))

    def test_records_the_segmentation_map(self) -> None:
        info = self.step()
        segmentation = info["segmentation"]
        self.assertEqual(segmentation["segmentation_map"].shape, (PATCH, PATCH))

    def test_records_the_surface_the_segmentation_and_object_agree_on(self) -> None:
        info = self.step()
        self.assertEqual(info["segmentation"]["surface_map"].shape, (PATCH, PATCH))

    def test_records_the_points_handed_to_the_tracker(self) -> None:
        info = self.step()
        segmentation = info["segmentation"]
        self.assertEqual(
            segmentation["surface_locations"].shape, (PATCH * PATCH, 3)
        )
        self.assertEqual(len(segmentation["surface_salience"]), PATCH * PATCH)

    def test_records_the_voxel_size_so_voxels_can_be_placed_in_the_world(self) -> None:
        # Voxel coordinates are indices; without the size they are not locations.
        info = self.step()
        self.assertEqual(info["region"]["voxel_size"], VOXEL_SIZE)

    def test_records_the_voxel_grid(self) -> None:
        info = self.step()
        grid = info["region"]["voxel_grid"]
        self.assertGreater(len(grid), 0)
        self.assertEqual(grid.voxels.shape, (len(grid), 3))

    def test_voxels_can_be_mapped_back_to_world_coordinates(self) -> None:
        info = self.step()
        region = info["region"]
        corners = region["voxel_grid"].voxels * region["voxel_size"]
        self.assertTrue(np.isfinite(corners).all())
        np.testing.assert_array_less(
            np.abs(corners).max(), PATCH * VOXEL_SIZE + VOXEL_SIZE
        )

    def test_the_logged_grid_is_a_snapshot_not_a_live_reference(self) -> None:
        info = self.step()
        logged = info["region"]["voxel_grid"]
        before = len(logged)
        # A later step rebuilds the tracker's grid; the logged one must not move.
        self.sensor_module._region_tracker.step(np.array([[9.0, 9.0, 9.0]]))
        self.assertEqual(len(logged), before)

    def test_no_segmentation_strategy_records_no_segmentation_telemetry(self) -> None:
        self.sensor_module._segmentation_strategy = None
        info = self.step()
        for key in ("segmentation", "region"):
            self.assertNotIn(key, info)
