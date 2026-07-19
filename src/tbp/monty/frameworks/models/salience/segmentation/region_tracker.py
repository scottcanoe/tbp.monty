# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Tracking of the regions of space an object occupies.

A region tracker is handed the points observed on a step and decides how to turn
them into a :class:`VoxelGrid` -- including which features each voxel carries. It
owns whatever state persists across steps, so that the grid itself stays an
immutable snapshot.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from tbp.monty.frameworks.models.salience.segmentation.voxel_geometry import (
    connected_components,
)
from tbp.monty.frameworks.models.salience.segmentation.voxels import (
    VOXEL_LEVELS,
    Feature,
    NumericFeature,
    Voxel,
    VoxelGrid,
    voxelize_and_bin_points,
)
from tbp.monty.memento import Memento


class RegionTracker:
    """Voxelizes each step's observed points into a :class:`VoxelGrid`.

    Points are quantized to voxels, and the points landing in a voxel are reduced
    to that voxel's feature values by each :class:`Feature`'s own ``reduce`` -- the
    reduction policy belongs to the feature, not to this tracker. Which features a
    voxel carries is this tracker's choice: either the features it was declared
    with, or one inferred for each feature supplied with the observation.

    Multi-step accumulation is not implemented yet -- each observation replaces the
    previous grid rather than integrating with it.
    """

    def __init__(
        self,
        voxel_size: float = 0.005,
        features: Mapping[str, Feature] | None = None,
    ) -> None:
        self._voxel_size = voxel_size
        self._declared = dict(features) if features is not None else None
        self._grid = VoxelGrid()

    @property
    def voxel_size(self) -> float:
        """Edge length of a voxel, in world units."""
        return self._voxel_size

    @property
    def grid(self) -> VoxelGrid:
        """The grid built from the most recent observation."""
        return self._grid

    def step(
        self,
        points: npt.NDArray[np.floating],
        features: Mapping[str, npt.NDArray] | None = None,
    ) -> dict:
        """Build this step's grid from the observed points.

        Args:
            points: Observed world-coordinate points as an ``(num_points, 3)``
                array. May be empty, yielding an empty grid.
            features: Point features aligned with ``points``, keyed by name. The
                tracker chooses which of these become voxel features.

        Returns:
            Dictionary of stuff.

        """
        features = dict(features) if features is not None else {}
        self._grid = self._build(points, features)
        result = {}
        result["grid"] = self._grid
        return result

    def contains_points(
        self, points: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.bool_]:
        """Test which locations fall within an occupied voxel.

        Args:
            points: a (N, 3) array of points.

        Returns:
            A boolean array with shape (N,).

        """
        occupied = self._grid.data.index
        # Normalize to (N, 3) so one path serves both shapes; squeeze at the end.
        points = np.atleast_2d(points)
        if len(occupied) == 0:
            return np.zeros(len(points), dtype=bool)

        indices = np.floor(points / self._voxel_size).astype(int)
        query = pd.MultiIndex.from_arrays(indices.T, names=VOXEL_LEVELS)
        return query.isin(occupied)

    def connected_components(self, connectivity: int = 26) -> list[VoxelGrid]:
        """Group occupied voxels into distinct connected regions.

        Args:
            connectivity: Voxel adjacency, one of 6, 18, or 26.

        Returns:
            One :class:`VoxelGrid` per connected component, each holding that
            region's voxels and the corresponding rows of every feature.

        """
        if len(self._grid) == 0:
            return []

        return [
            VoxelGrid(self._grid.data.iloc[members])
            for members in connected_components(self._grid.voxels, connectivity)
        ]

    def reset(self) -> None:
        """Discard the current grid."""
        self._grid = VoxelGrid()

    def state_dict(self) -> Memento:
        """Export the current state for telemetry."""  # noqa: DOC201
        return {"grid": self._grid}

    def _region_labels(self, connectivity: int) -> npt.NDArray[np.intp]:
        """Label each occupied voxel with the region it belongs to.

        Args:
            connectivity: Voxel adjacency, one of 6, 18, or 26.

        Returns:
            An ``(num_voxels,)`` array of region indices, row-aligned with the
            grid's voxels.

        """
        voxels = self._grid.voxels
        labels = np.zeros(len(voxels), dtype=np.intp)
        if len(voxels) == 0:
            return labels
        for label, members in enumerate(connected_components(voxels, connectivity)):
            labels[members] = label
        return labels

    def _build(
        self,
        points: npt.NDArray[np.floating],
        features: dict[str, npt.NDArray],
    ) -> VoxelGrid:
        """Voxelize ``points`` and reduce their features onto each voxel.

        Returns:
            A grid of the observed voxels and their feature values.

        """
        points = np.atleast_2d(points)
        if points.size == 0:
            return VoxelGrid()
        covoxel_points = voxelize_and_bin_points(points, self._voxel_size)
        voxels: list[Voxel] = list(covoxel_points.keys())
        all_covoxel_groups: list[list[int]] = list(covoxel_points.values())

        # Drive off the resolved features rather than the supplied ones: the
        # tracker chooses what a voxel carries, and inferred features live here.
        combined_features: dict[str, npt.NDArray] = {}
        for feat_name, feat_def in self._features_to_add(features).items():
            feat_vals = np.asarray(features[feat_name])
            feat_vals_voxel = np.zeros(
                (len(voxels), *feat_def.shape),
                dtype=feat_def.dtype,
            )
            for voxel_idx, voxel_points in enumerate(all_covoxel_groups):
                # A lone point is already in reduced form; nothing to combine.
                if len(voxel_points) == 1:
                    feat_vals_voxel[voxel_idx] = feat_vals[voxel_points[0]]
                    continue

                # Otherwise let the feature decide how to combine them.
                feat_vals_voxel[voxel_idx] = feat_def.reduce(feat_vals[voxel_points])

            combined_features[feat_name] = feat_vals_voxel

        return VoxelGrid.from_voxels_and_features(voxels, combined_features)

    def _features_to_add(
        self, values: Mapping[str, npt.NDArray]
    ) -> Mapping[str, Feature]:
        """Decide which features this tracker puts on a voxel.

        Returns:
            The declared features, or one inferred per supplied point feature.

        Raises:
            ValueError: If a declared feature has no values in ``values``.

        """
        if self._declared is None:
            return {
                name: NumericFeature(
                    np.shape(np.asarray(value)[0]) or 1,
                    np.asarray(value).dtype,
                )
                for name, value in values.items()
            }
        missing = [name for name in self._declared if name not in values]
        if missing:
            raise ValueError(f"no values supplied for feature(s): {', '.join(missing)}")
        return self._declared
