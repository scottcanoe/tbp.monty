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
    AGE,
    COUNT,
    TRACKED_FEATURES,
    RegionTracker,
)
from tbp.monty.frameworks.models.salience.segmentation.voxels import NumericFeature

# Two points inside one voxel, and a third far enough away to occupy its own.
NEAR_POINTS = np.array([[0.0, 0, 0], [0.005, 0, 0]])
FAR_POINT = np.array([[0.5, 0, 0]])
# The voxels those points fall in, at voxel_size=0.01.
NEAR_VOXEL = (0, 0, 0)
FAR_VOXEL = (50, 0, 0)
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

    def test_a_step_adds_to_the_grid_rather_than_replacing_it(self) -> None:
        self.tracker.step(NEAR_POINTS, {"rgb": ALL_RGB[:2]})
        self.tracker.step(FAR_POINT, {"rgb": ALL_RGB[2:]})
        self.assertEqual(
            sorted(tuple(int(c) for c in v) for v in self.tracker.grid.voxels),
            [NEAR_VOXEL, FAR_VOXEL],
        )

    def test_reset_discards_the_grid(self) -> None:
        self.tracker.step(ALL_POINTS, {"rgb": ALL_RGB})
        self.tracker.reset()
        self.assertEqual(len(self.tracker.grid.data), 0)


class RegionTrackerAgeTest(unittest.TestCase):
    """Age is the tracker's own feature, not one reduced from the points."""

    def setUp(self) -> None:
        self.tracker = RegionTracker(voxel_size=0.01, lifetime=4)
        self.tracker.step(ALL_POINTS, {"rgb": ALL_RGB})

    def test_every_voxel_carries_an_age(self) -> None:
        self.assertIn(AGE, self.tracker.grid.feature_names)

    def test_age_starts_at_the_full_lifetime(self) -> None:
        ages = self.tracker.grid.data[AGE].to_numpy()
        np.testing.assert_array_equal(ages.ravel(), [4, 4])

    def test_age_is_stored_as_an_integer(self) -> None:
        self.assertEqual(self.tracker.grid.data[AGE].to_numpy().dtype, np.int32)

    def test_age_is_added_even_with_no_point_features(self) -> None:
        tracker = RegionTracker(voxel_size=0.01, lifetime=2)
        tracker.step(ALL_POINTS)
        self.assertEqual(point_features(tracker), ())
        self.assertIn(AGE, tracker.grid.feature_names)
        np.testing.assert_array_equal(
            tracker.grid.data[AGE].to_numpy().ravel(), [2, 2]
        )

    def test_lifetime_is_exposed(self) -> None:
        self.assertEqual(RegionTracker(lifetime=9).lifetime, 9)

    def test_lifetime_must_be_positive(self) -> None:
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                RegionTracker(lifetime=bad)

    def test_a_point_feature_cannot_shadow_the_tracker_managed_age(self) -> None:
        tracker = RegionTracker(voxel_size=0.01)
        with self.assertRaises(ValueError):
            tracker.step(ALL_POINTS, {AGE: ALL_SALIENCE})

    def test_a_re_observed_voxel_returns_to_a_full_age(self) -> None:
        self.tracker.step(NEAR_POINTS, {"rgb": ALL_RGB[:2]})
        ages = ages_by_voxel(self.tracker)
        self.assertEqual(ages[NEAR_VOXEL], 4)


def point_features(tracker: RegionTracker) -> tuple[str, ...]:
    """Return the grid's features that came from the observed points.

    Excludes the ones the tracker writes itself, so tests of feature selection do
    not have to be updated whenever another tracked feature is added.

    Returns:
        The names of the point-derived features, in grid order.

    """
    return tuple(
        name
        for name in tracker.grid.feature_names
        if name not in TRACKED_FEATURES
    )


def feature_by_voxel(
    tracker: RegionTracker, feature: str
) -> dict[tuple[int, int, int], int]:
    """Map each occupied voxel to one of its scalar feature values.

    Returns:
        Voxel coordinate to value, for every voxel the tracker holds.

    """
    data = tracker.grid.data
    voxels = [tuple(int(c) for c in index) for index in data.index]
    return dict(zip(voxels, data[feature].to_numpy().ravel().tolist()))


def ages_by_voxel(tracker: RegionTracker) -> dict[tuple[int, int, int], int]:
    """Map each occupied voxel to its remaining age.

    Returns:
        Voxel coordinate to age, for every voxel the tracker holds.

    """
    return feature_by_voxel(tracker, AGE)


def counts_by_voxel(tracker: RegionTracker) -> dict[tuple[int, int, int], int]:
    """Map each occupied voxel to how many times it has been observed.

    Returns:
        Voxel coordinate to count, for every voxel the tracker holds.

    """
    return feature_by_voxel(tracker, COUNT)


class RegionTrackerMemoryTest(unittest.TestCase):
    """Voxels persist across steps, ageing until they are re-observed or expire."""

    def setUp(self) -> None:
        self.tracker = RegionTracker(voxel_size=0.01, lifetime=3)

    def observe_near(self, colour: float = 200.0) -> None:
        """Observe the near voxel with the given red intensity."""
        self.tracker.step(NEAR_POINTS[:1], {"rgb": np.array([[colour, 0.0, 0.0]])})

    def observe_far(self) -> None:
        """Observe the far voxel."""
        self.tracker.step(FAR_POINT, {"rgb": np.array([[0.0, 0.0, 200.0]])})

    def test_an_unobserved_voxel_is_remembered(self) -> None:
        self.observe_near()
        self.observe_far()
        self.assertEqual(set(ages_by_voxel(self.tracker)), {NEAR_VOXEL, FAR_VOXEL})

    def test_an_unobserved_voxel_ages_by_one_step(self) -> None:
        self.observe_near()
        self.observe_far()
        self.assertEqual(ages_by_voxel(self.tracker)[NEAR_VOXEL], 2)

    def test_a_voxel_expires_after_its_lifetime_of_steps(self) -> None:
        self.observe_near()
        for _ in range(3):
            self.observe_far()
        self.assertEqual(set(ages_by_voxel(self.tracker)), {FAR_VOXEL})

    def test_observing_nothing_still_ages_the_grid(self) -> None:
        self.observe_near()
        self.tracker.step(np.empty((0, 3)))
        self.assertEqual(ages_by_voxel(self.tracker)[NEAR_VOXEL], 2)

    def test_the_grid_empties_once_everything_expires(self) -> None:
        self.observe_near()
        for _ in range(3):
            self.tracker.step(np.empty((0, 3)))
        self.assertEqual(len(self.tracker.grid), 0)

    def test_re_observing_refreshes_a_voxels_features(self) -> None:
        self.observe_near(colour=200.0)
        self.observe_far()
        self.observe_near(colour=50.0)
        rgb = self.tracker.grid.data.loc[NEAR_VOXEL, "rgb"].to_numpy()
        np.testing.assert_allclose(rgb, [50.0, 0.0, 0.0])

    def test_age_stays_an_integer_across_steps(self) -> None:
        self.observe_near()
        self.observe_far()
        self.assertEqual(self.tracker.grid.data[AGE].to_numpy().dtype, np.int32)

    def test_reset_forgets_everything(self) -> None:
        self.observe_near()
        self.tracker.reset()
        self.assertEqual(len(self.tracker.grid), 0)

    def test_step_reports_both_what_is_remembered_and_what_was_seen(self) -> None:
        self.observe_near()
        result = self.tracker.step(
            FAR_POINT, {"rgb": np.array([[0.0, 0.0, 200.0]])}
        )
        self.assertEqual(len(result["grid"]), 2)
        self.assertEqual(len(result["observed"]), 1)

    def test_contains_points_covers_remembered_voxels(self) -> None:
        # Memory is what lets a goal fall inside a voxel seen on an earlier step.
        self.observe_near()
        self.observe_far()
        np.testing.assert_array_equal(
            self.tracker.contains_points(np.vstack([NEAR_POINTS[:1], FAR_POINT])),
            [True, True],
        )


class RegionTrackerCountTest(unittest.TestCase):
    """Count tallies how many times a voxel has been observed."""

    def setUp(self) -> None:
        self.tracker = RegionTracker(voxel_size=0.01, lifetime=3)

    def observe_near(self) -> None:
        """Observe the near voxel."""
        self.tracker.step(NEAR_POINTS[:1], {"rgb": np.array([[200.0, 0.0, 0.0]])})

    def observe_far(self) -> None:
        """Observe the far voxel."""
        self.tracker.step(FAR_POINT, {"rgb": np.array([[0.0, 0.0, 200.0]])})

    def test_every_voxel_carries_a_count(self) -> None:
        self.observe_near()
        self.assertIn(COUNT, self.tracker.grid.feature_names)

    def test_a_newly_seen_voxel_starts_at_one(self) -> None:
        self.observe_near()
        self.assertEqual(counts_by_voxel(self.tracker)[NEAR_VOXEL], 1)

    def test_re_observing_adds_to_the_count(self) -> None:
        for expected in (1, 2, 3):
            self.observe_near()
            self.assertEqual(counts_by_voxel(self.tracker)[NEAR_VOXEL], expected)

    def test_an_unobserved_voxel_keeps_its_count(self) -> None:
        self.observe_near()
        self.observe_near()
        self.observe_far()
        counts = counts_by_voxel(self.tracker)
        self.assertEqual(counts[NEAR_VOXEL], 2)
        self.assertEqual(counts[FAR_VOXEL], 1)

    def test_counts_are_tallied_independently_per_voxel(self) -> None:
        self.observe_near()
        self.observe_far()
        self.observe_far()
        counts = counts_by_voxel(self.tracker)
        self.assertEqual(counts[NEAR_VOXEL], 1)
        self.assertEqual(counts[FAR_VOXEL], 2)

    def test_count_restarts_after_a_voxel_expires(self) -> None:
        # An expired voxel is forgotten, so its tally starts over.
        self.observe_near()
        self.observe_near()
        for _ in range(3):
            self.tracker.step(np.empty((0, 3)))
        self.assertEqual(len(self.tracker.grid), 0)
        self.observe_near()
        self.assertEqual(counts_by_voxel(self.tracker)[NEAR_VOXEL], 1)

    def test_count_stays_an_integer_across_steps(self) -> None:
        self.observe_near()
        self.observe_near()
        self.assertEqual(self.tracker.grid.data[COUNT].to_numpy().dtype, np.int32)

    def test_count_is_added_even_with_no_point_features(self) -> None:
        tracker = RegionTracker(voxel_size=0.01)
        tracker.step(NEAR_POINTS[:1])
        self.assertEqual(point_features(tracker), ())
        self.assertIn(COUNT, tracker.grid.feature_names)

    def test_a_point_feature_cannot_shadow_the_tracker_managed_count(self) -> None:
        with self.assertRaises(ValueError):
            self.tracker.step(NEAR_POINTS[:1], {COUNT: np.array([1.0])})


class RegionTrackerFeatureSelectionTest(unittest.TestCase):
    def test_undeclared_features_are_inferred_from_the_supplied_values(self) -> None:
        tracker = RegionTracker(voxel_size=0.01)
        tracker.step(ALL_POINTS, {"rgb": ALL_RGB, "salience": ALL_SALIENCE})
        self.assertEqual(point_features(tracker), ("rgb", "salience"))
        # The tracker's own features accompany whatever it keeps.
        for tracked in TRACKED_FEATURES:
            self.assertIn(tracked, tracker.grid.feature_names)
        self.assertEqual(tracker.grid.data["rgb"].to_numpy().shape, (2, 3))
        self.assertEqual(tracker.grid.data["salience"].to_numpy().shape, (2, 1))

    def test_declared_features_are_the_only_ones_kept(self) -> None:
        tracker = RegionTracker(
            voxel_size=0.01, features={"rgb": NumericFeature(3, np.float32)}
        )
        tracker.step(ALL_POINTS, {"rgb": ALL_RGB, "salience": ALL_SALIENCE})
        self.assertEqual(point_features(tracker), ("rgb",))

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
        result = self.tracker.contains_points(np.array([[0.0, 0, 0], [9.0, 9, 9]]))
        np.testing.assert_array_equal(result, [True, False])

    def test_a_single_flat_point_is_accepted(self) -> None:
        # Normalized to (1, 3), so the result is still an array.
        np.testing.assert_array_equal(
            self.tracker.contains_points(np.array([0.0, 0, 0])), [True]
        )

    def test_any_location_in_an_occupied_voxel_is_contained(self) -> None:
        # A different point in the same voxel as an observed one.
        np.testing.assert_array_equal(
            self.tracker.contains_points(np.array([[0.009, 0, 0]])), [True]
        )

    def test_an_empty_grid_contains_nothing(self) -> None:
        empty = RegionTracker(voxel_size=0.01)
        np.testing.assert_array_equal(
            empty.contains_points(np.array([[0.0, 0, 0]])), [False]
        )


class RegionTrackerStateDictTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = RegionTracker(voxel_size=0.01)
        self.tracker.step(ALL_POINTS, {"rgb": ALL_RGB, "salience": ALL_SALIENCE})

    def test_exports_the_current_grid(self) -> None:
        grid = self.tracker.state_dict()["grid"]
        self.assertEqual(len(grid), 2)
        self.assertEqual(
            tuple(n for n in grid.feature_names if n not in TRACKED_FEATURES),
            ("rgb", "salience"),
        )

    def test_an_empty_grid_is_exported_as_an_empty_grid(self) -> None:
        self.assertEqual(len(RegionTracker().state_dict()["grid"]), 0)


class VoxelGridSnapshotTest(unittest.TestCase):
    """A logged grid must not change when the tracker moves on."""

    def test_len_counts_occupied_voxels(self) -> None:
        tracker = RegionTracker(voxel_size=0.01)
        tracker.step(ALL_POINTS, {"rgb": ALL_RGB})
        self.assertEqual(len(tracker.grid), 2)
        self.assertEqual(len(RegionTracker().grid), 0)

    def test_a_copy_is_unaffected_by_later_steps(self) -> None:
        tracker = RegionTracker(voxel_size=0.01)
        tracker.step(NEAR_POINTS, {"rgb": ALL_RGB[:2]})
        snapshot = tracker.grid.copy()
        # The grid remembers, so a later step grows it past the snapshot.
        tracker.step(FAR_POINT, {"rgb": ALL_RGB[2:]})
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(len(tracker.grid), 2)

    def test_a_copy_holds_its_own_data(self) -> None:
        tracker = RegionTracker(voxel_size=0.01)
        tracker.step(ALL_POINTS, {"rgb": ALL_RGB})
        snapshot = tracker.grid.copy()
        self.assertIsNot(snapshot.data, tracker.grid.data)
        np.testing.assert_array_equal(snapshot.voxels, tracker.grid.voxels)
