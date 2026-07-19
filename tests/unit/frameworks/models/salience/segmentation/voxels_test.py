# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from tbp.monty.frameworks.models.salience.segmentation.voxels import (
    FEATURE_LEVELS,
    VOXEL_LEVELS,
    NumericFeature,
    VoxelGrid,
    as_array_voxels,
    as_tuple_voxels,
    empty_voxel_frame,
    voxelize_and_bin_points,
)


class AsVoxelsTest(unittest.TestCase):
    def test_as_array_voxels_reshapes_a_single_flat_voxel(self) -> None:
        np.testing.assert_array_equal(as_array_voxels([1, 2, 3]), [[1, 2, 3]])

    def test_as_array_voxels_rejects_non_triples(self) -> None:
        with self.assertRaises(AssertionError):
            as_array_voxels([[1, 2], [3, 4]])

    def test_as_tuple_voxels_coerces_to_builtin_ints(self) -> None:
        voxels = as_tuple_voxels(np.array([[1, 2, 3]]))
        self.assertEqual(voxels, ((1, 2, 3),))
        for coord in voxels[0]:
            self.assertIsInstance(coord, int)

    def test_as_tuple_voxels_returns_json_serializable_coordinates(self) -> None:
        # These tuples become index labels and reach telemetry, so numpy scalars
        # would break serialization.
        voxels = as_tuple_voxels(np.array([[1, 2, 3]]))
        self.assertEqual(json.dumps(list(voxels[0])), "[1, 2, 3]")


class VoxelizeAndBinPointsTest(unittest.TestCase):
    def test_bins_points_sharing_a_voxel_together(self) -> None:
        points = np.array([[0.0, 0, 0], [0.005, 0, 0], [0.5, 0, 0]])
        binned = voxelize_and_bin_points(points, voxel_size=0.01)
        self.assertEqual(binned, {(0, 0, 0): [0, 1], (50, 0, 0): [2]})

    def test_keys_are_occupied_voxels_in_first_seen_order(self) -> None:
        points = np.array([[0.5, 0, 0], [0.0, 0, 0], [0.5, 0, 0]])
        binned = voxelize_and_bin_points(points, voxel_size=0.01)
        self.assertEqual(list(binned), [(50, 0, 0), (0, 0, 0)])

    def test_floors_negative_coordinates(self) -> None:
        binned = voxelize_and_bin_points(np.array([[-0.005, 0, 0]]), voxel_size=0.01)
        self.assertEqual(list(binned), [(-1, 0, 0)])

    def test_keys_are_builtin_int_tuples(self) -> None:
        binned = voxelize_and_bin_points(np.array([[0.0, 0, 0]]), voxel_size=0.01)
        for coord in next(iter(binned)):
            self.assertIsInstance(coord, int)


class NumericFeatureShapeTest(unittest.TestCase):
    def test_normalizes_a_bare_int_to_a_one_tuple(self) -> None:
        self.assertEqual(NumericFeature(3).shape, (3,))

    def test_normalizes_a_numpy_int_to_builtin_ints(self) -> None:
        shape = NumericFeature(np.int64(4)).shape
        self.assertEqual(shape, (4,))
        self.assertIsInstance(shape[0], int)

    def test_preserves_a_sequence_shape(self) -> None:
        self.assertEqual(NumericFeature((3,)).shape, (3,))
        self.assertEqual(NumericFeature([2, 2]).shape, (2, 2))

    def test_defaults_to_a_scalar_of_one_component(self) -> None:
        self.assertEqual(NumericFeature().shape, (1,))

    def test_rejects_empty_and_non_positive_shapes(self) -> None:
        for bad in [(), 0, (3, 0)]:
            with self.assertRaises(AssertionError):
                NumericFeature(bad)


class NumericFeatureReduceTest(unittest.TestCase):
    def test_reduce_averages_observations(self) -> None:
        feature = NumericFeature(3, np.float32)
        reduced = feature.reduce(np.array([[200.0, 0, 0], [100.0, 0, 0]]))
        np.testing.assert_allclose(reduced, [150.0, 0.0, 0.0])

    def test_reduce_returns_the_declared_shape_and_dtype(self) -> None:
        feature = NumericFeature(3, np.float32)
        reduced = feature.reduce(np.array([[1.0, 2, 3]]))
        self.assertEqual(reduced.shape, (3,))
        self.assertEqual(reduced.dtype, np.float32)

    def test_reduce_of_a_scalar_feature_keeps_one_component(self) -> None:
        reduced = NumericFeature(1, np.float32).reduce(np.array([0.1, 0.5]))
        self.assertEqual(reduced.shape, (1,))
        np.testing.assert_allclose(reduced, [0.3])


class NumericFeatureDistanceTest(unittest.TestCase):
    def test_distance_between_two_values_is_a_scalar(self) -> None:
        feature = NumericFeature(3, np.float32)
        distance = feature.distance(np.array([3.0, 4, 0]), np.array([0.0, 0, 0]))
        self.assertEqual(np.ndim(distance), 0)
        self.assertAlmostEqual(float(distance), 5.0, places=5)

    def test_distance_broadcasts_one_value_against_many(self) -> None:
        feature = NumericFeature(3, np.float32)
        distances = feature.distance(
            np.array([0.0, 0, 0]), np.array([[3.0, 4, 0], [0.0, 0, 0]])
        )
        np.testing.assert_allclose(distances, [5.0, 0.0], atol=1e-5)

    def test_distance_of_a_scalar_feature_is_the_absolute_difference(self) -> None:
        feature = NumericFeature(1, np.float32)
        distance = feature.distance(np.array([0.2]), np.array([0.9]))
        self.assertAlmostEqual(float(distance), 0.7, places=5)


class EmptyVoxelFrameTest(unittest.TestCase):
    def test_has_both_index_levels_named(self) -> None:
        frame = empty_voxel_frame()
        self.assertEqual(tuple(frame.index.names), VOXEL_LEVELS)
        self.assertEqual(tuple(frame.columns.names), FEATURE_LEVELS)

    def test_is_empty(self) -> None:
        self.assertEqual(len(empty_voxel_frame()), 0)


class VoxelGridTest(unittest.TestCase):
    def test_default_grid_is_empty_but_well_formed(self) -> None:
        grid = VoxelGrid()
        self.assertEqual(len(grid.data), 0)
        self.assertEqual(grid.voxels.shape, (0, 3))
        self.assertEqual(grid.feature_names, ())

    def test_rejects_a_frame_without_the_expected_index_levels(self) -> None:
        with self.assertRaises(AssertionError):
            VoxelGrid(pd.DataFrame({"a": [1]}))

    def test_accepts_its_own_frame_round_trip(self) -> None:
        grid = VoxelGrid.from_voxels_and_features([[0, 0, 0]], {"s": [1.0]})
        self.assertEqual(len(VoxelGrid(grid.data).data), 1)


class VoxelGridFromVoxelsAndFeaturesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = VoxelGrid.from_voxels_and_features(
            [[0, 0, 0], [1, 0, 0]],
            {
                "salience": np.array([0.1, 0.9], np.float32),
                "rgb": np.array([[1, 2, 3], [4, 5, 6]], np.float32),
                "age": np.array([6, 3], np.int32),
            },
        )

    def test_indexes_rows_by_voxel_coordinate(self) -> None:
        self.assertEqual(tuple(self.grid.data.index.names), VOXEL_LEVELS)
        np.testing.assert_array_equal(self.grid.voxels, [[0, 0, 0], [1, 0, 0]])

    def test_indexes_columns_by_feature_and_component(self) -> None:
        self.assertEqual(tuple(self.grid.data.columns.names), FEATURE_LEVELS)
        self.assertEqual(self.grid.feature_names, ("salience", "rgb", "age"))

    def test_a_multi_component_feature_spans_several_columns(self) -> None:
        self.assertEqual(self.grid.data["rgb"].to_numpy().shape, (2, 3))

    def test_a_scalar_feature_occupies_one_column(self) -> None:
        self.assertEqual(self.grid.data["salience"].to_numpy().shape, (2, 1))

    def test_each_feature_keeps_its_own_dtype(self) -> None:
        # A shared numeric block would upcast the int feature to float.
        self.assertEqual(self.grid.data["age"].to_numpy().dtype, np.int32)
        self.assertEqual(self.grid.data["rgb"].to_numpy().dtype, np.float32)

    def test_no_voxels_yields_an_empty_grid(self) -> None:
        self.assertEqual(len(VoxelGrid.from_voxels_and_features([]).data), 0)

    def test_voxels_without_features_have_no_columns(self) -> None:
        grid = VoxelGrid.from_voxels_and_features([[0, 0, 0]])
        self.assertEqual(grid.data.shape, (1, 0))

    def test_rejects_a_feature_whose_rows_do_not_match_the_voxels(self) -> None:
        with self.assertRaises(AssertionError):
            VoxelGrid.from_voxels_and_features([[0, 0, 0]], {"rgb": np.zeros((5, 3))})
