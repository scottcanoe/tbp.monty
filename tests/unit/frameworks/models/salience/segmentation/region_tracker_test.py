# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest

import numpy as np

from tbp.monty.frameworks.models.salience.segmentation.region_tracker import (
    RegionTracker,
)
from tbp.monty.frameworks.models.salience.segmentation.voxels import NumericFeature

# Two points inside one voxel, and a third far enough away to occupy its own.
NEAR_POINTS = np.array([[0.0, 0, 0], [0.005, 0, 0]])
FAR_POINT = np.array([[0.5, 0, 0]])
ALL_POINTS = np.vstack([NEAR_POINTS, FAR_POINT])
ALL_RGB = np.array([[200.0, 0, 0], [100.0, 0, 0], [0.0, 0, 200]])
ALL_SALIENCE = np.array([0.1, 0.3, 0.9])


class RegionTrackerStepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = RegionTracker(voxel_size=0.01)

    def test_points_sharing_a_voxel_collapse_to_one_row(self) -> None:
        self.tracker.step(ALL_POINTS, {"rgb": ALL_RGB})
        self.assertEqual(len(self.tracker.grid.data), 2)

    def test_a_voxels_feature_is_the_reduction_of_its_points(self) -> None:
        self.tracker.step(ALL_POINTS, {"rgb": ALL_RGB})
        rgb = self.tracker.grid.data["rgb"].to_numpy()
        np.testing.assert_allclose(rgb[0], [150.0, 0.0, 0.0])

    def test_a_lone_point_is_carried_through_unchanged(self) -> None:
        self.tracker.step(FAR_POINT, {"rgb": np.array([[0.0, 0, 200]])})
        np.testing.assert_allclose(
            self.tracker.grid.data["rgb"].to_numpy()[0], [0.0, 0.0, 200.0]
        )

    def test_no_points_yields_an_empty_grid(self) -> None:
        self.tracker.step(np.empty((0, 3)))
        self.assertEqual(len(self.tracker.grid.data), 0)

    def test_a_single_flat_point_is_accepted(self) -> None:
        self.tracker.step(np.array([0.0, 0, 0]), {"s": np.array([1.0])})
        self.assertEqual(len(self.tracker.grid.data), 1)

    def test_each_step_replaces_the_previous_grid(self) -> None:
        # Multi-step accumulation is not implemented; the newest observation wins.
        self.tracker.step(NEAR_POINTS, {"rgb": ALL_RGB[:2]})
        self.tracker.step(FAR_POINT, {"rgb": ALL_RGB[2:]})
        np.testing.assert_array_equal(self.tracker.grid.voxels, [[50, 0, 0]])

    def test_reset_discards_the_grid(self) -> None:
        self.tracker.step(ALL_POINTS, {"rgb": ALL_RGB})
        self.tracker.reset()
        self.assertEqual(len(self.tracker.grid.data), 0)


class RegionTrackerFeatureSelectionTest(unittest.TestCase):
    def test_undeclared_features_are_inferred_from_the_supplied_values(self) -> None:
        tracker = RegionTracker(voxel_size=0.01)
        tracker.step(ALL_POINTS, {"rgb": ALL_RGB, "salience": ALL_SALIENCE})
        self.assertEqual(tracker.grid.feature_names, ("rgb", "salience"))
        self.assertEqual(tracker.grid.data["rgb"].to_numpy().shape, (2, 3))
        self.assertEqual(tracker.grid.data["salience"].to_numpy().shape, (2, 1))

    def test_declared_features_are_the_only_ones_kept(self) -> None:
        tracker = RegionTracker(
            voxel_size=0.01, features={"rgb": NumericFeature(3, np.float32)}
        )
        tracker.step(ALL_POINTS, {"rgb": ALL_RGB, "salience": ALL_SALIENCE})
        self.assertEqual(tracker.grid.feature_names, ("rgb",))

    def test_declared_features_are_stored_in_their_declared_dtype(self) -> None:
        tracker = RegionTracker(
            voxel_size=0.01, features={"rgb": NumericFeature(3, np.float32)}
        )
        tracker.step(ALL_POINTS, {"rgb": ALL_RGB})
        self.assertEqual(tracker.grid.data["rgb"].to_numpy().dtype, np.float32)

    def test_a_declared_feature_with_no_values_is_an_error(self) -> None:
        tracker = RegionTracker(
            voxel_size=0.01, features={"rgb": NumericFeature(3, np.float32)}
        )
        with self.assertRaises(ValueError):
            tracker.step(ALL_POINTS, {"salience": ALL_SALIENCE})


class RegionTrackerContainsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = RegionTracker(voxel_size=0.01)
        self.tracker.step(ALL_POINTS, {"rgb": ALL_RGB})

    def test_many_locations_yield_an_array(self) -> None:
        result = self.tracker.contains(np.array([[0.0, 0, 0], [9.0, 9, 9]]))
        np.testing.assert_array_equal(result, [True, False])

    def test_a_single_location_yields_a_builtin_bool(self) -> None:
        # Not np.bool_: the result is annotated as a bool and may be serialized.
        inside = self.tracker.contains(np.array([0.0, 0, 0]))
        self.assertIsInstance(inside, bool)
        self.assertTrue(inside)
        self.assertFalse(self.tracker.contains(np.array([9.0, 9, 9])))

    def test_any_location_in_an_occupied_voxel_is_contained(self) -> None:
        # A different point in the same voxel as an observed one.
        self.assertTrue(self.tracker.contains(np.array([0.009, 0, 0])))

    def test_an_empty_grid_contains_nothing(self) -> None:
        empty = RegionTracker(voxel_size=0.01)
        self.assertFalse(empty.contains(np.array([0.0, 0, 0])))
        np.testing.assert_array_equal(empty.contains(np.array([[0.0, 0, 0]])), [False])


class RegionTrackerRegionsTest(unittest.TestCase):
    def test_separated_voxels_form_distinct_regions(self) -> None:
        tracker = RegionTracker(voxel_size=0.01)
        tracker.step(ALL_POINTS, {"rgb": ALL_RGB})
        self.assertEqual(len(tracker.regions()), 2)

    def test_adjacent_voxels_form_one_region(self) -> None:
        tracker = RegionTracker(voxel_size=0.01)
        adjacent = np.array([[0.0, 0, 0], [0.011, 0, 0]])
        tracker.step(adjacent, {"rgb": np.zeros((2, 3))})
        self.assertEqual(len(tracker.regions()), 1)

    def test_a_region_carries_its_own_voxels_and_features(self) -> None:
        tracker = RegionTracker(voxel_size=0.01)
        tracker.step(ALL_POINTS, {"rgb": ALL_RGB})
        regions = sorted(tracker.regions(), key=lambda r: r.voxels[0, 0])
        np.testing.assert_array_equal(regions[0].voxels, [[0, 0, 0]])
        np.testing.assert_allclose(
            regions[0].data["rgb"].to_numpy()[0], [150.0, 0.0, 0.0]
        )

    def test_an_empty_grid_has_no_regions(self) -> None:
        self.assertEqual(RegionTracker().regions(), [])

    def test_regions_can_be_compared_by_a_features_distance(self) -> None:
        # The point of carrying features: telling regions apart.
        rgb = NumericFeature(3, np.float32)
        tracker = RegionTracker(voxel_size=0.01, features={"rgb": rgb})
        tracker.step(ALL_POINTS, {"rgb": ALL_RGB})
        regions = sorted(tracker.regions(), key=lambda r: r.voxels[0, 0])
        summaries = [rgb.reduce(r.data["rgb"].to_numpy()) for r in regions]
        self.assertGreater(float(rgb.distance(summaries[0], summaries[1])), 100.0)


class RegionTrackerStateDictTest(unittest.TestCase):
    def test_exports_voxels_and_every_feature_row_aligned(self) -> None:
        tracker = RegionTracker(voxel_size=0.01)
        tracker.step(ALL_POINTS, {"rgb": ALL_RGB, "salience": ALL_SALIENCE})
        state = tracker.state_dict()
        self.assertEqual(state["voxels"].shape, (2, 3))
        self.assertEqual(state["rgb"].shape, (2, 3))
        self.assertEqual(state["salience"].shape, (2, 1))

    def test_an_empty_grid_exports_empty_voxels(self) -> None:
        self.assertEqual(RegionTracker().state_dict()["voxels"].shape, (0, 3))
