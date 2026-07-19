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

from tbp.monty.frameworks.models.salience.segmentation.voxel_geometry import (
    connected_components,
)


def component_sizes(components: list[np.ndarray]) -> list[int]:
    """Return component sizes, largest first, for order-independent assertions.

    Returns:
        The number of voxels in each component, descending.

    """
    return sorted((len(c) for c in components), reverse=True)


class ConnectedComponentsTest(unittest.TestCase):
    def test_no_voxels_yields_no_components(self) -> None:
        self.assertEqual(connected_components(np.empty((0, 3), dtype=int)), [])

    def test_a_single_voxel_is_its_own_component(self) -> None:
        components = connected_components(np.array([[0, 0, 0]]))
        self.assertEqual(component_sizes(components), [1])

    def test_separated_voxels_are_distinct_components(self) -> None:
        voxels = np.array([[0, 0, 0], [50, 0, 0]])
        self.assertEqual(component_sizes(connected_components(voxels)), [1, 1])

    def test_components_index_into_the_input(self) -> None:
        voxels = np.array([[0, 0, 0], [50, 0, 0]])
        components = connected_components(voxels)
        indices = sorted(int(i) for c in components for i in c)
        self.assertEqual(indices, [0, 1])

    def test_a_chain_of_touching_voxels_is_one_component(self) -> None:
        voxels = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        self.assertEqual(component_sizes(connected_components(voxels)), [4])

    def test_face_neighbours_join_under_every_connectivity(self) -> None:
        voxels = np.array([[0, 0, 0], [1, 0, 0]])
        for connectivity in (6, 18, 26):
            components = connected_components(voxels, connectivity)
            self.assertEqual(component_sizes(components), [2])

    def test_edge_neighbours_join_only_from_18_connectivity(self) -> None:
        voxels = np.array([[0, 0, 0], [1, 1, 0]])
        self.assertEqual(component_sizes(connected_components(voxels, 6)), [1, 1])
        self.assertEqual(component_sizes(connected_components(voxels, 18)), [2])
        self.assertEqual(component_sizes(connected_components(voxels, 26)), [2])

    def test_corner_neighbours_join_only_under_26_connectivity(self) -> None:
        voxels = np.array([[0, 0, 0], [1, 1, 1]])
        self.assertEqual(component_sizes(connected_components(voxels, 6)), [1, 1])
        self.assertEqual(component_sizes(connected_components(voxels, 18)), [1, 1])
        self.assertEqual(component_sizes(connected_components(voxels, 26)), [2])

    def test_rejects_an_unsupported_connectivity(self) -> None:
        with self.assertRaises(ValueError):
            connected_components(np.array([[0, 0, 0]]), connectivity=7)
