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

from tbp.monty.frameworks.models.salience.segmentation.voxels import (
    VOXEL_LEVELS,
    Feature,
    NumericFeature,
    Voxel,
    VoxelGrid,
    voxelize_and_bin_points,
)
from tbp.monty.memento import Memento

AGE = "age"
"""Name of the feature holding how many steps a voxel has left to live."""

COUNT = "count"
"""Name of the feature holding how many times a voxel has been observed."""

TRACKED_FEATURES = (AGE, COUNT)
"""Features the tracker writes itself, rather than reducing from the points."""


class RegionTracker:
    """Voxelizes each step's observed points into a :class:`VoxelGrid`.

    Points are quantized to voxels, and the points landing in a voxel are reduced
    to that voxel's feature values by each :class:`Feature`'s own ``reduce`` -- the
    reduction policy belongs to the feature, not to this tracker. Which features a
    voxel carries is this tracker's choice: either the features it was declared
    with, or one inferred for each feature supplied with the observation.

    On top of those, every voxel carries two features the tracker writes itself,
    because they describe the voxel's history rather than what was observed:
    ``age``, how many steps it has left before it expires, and ``count``, how many
    times it has been observed.

    The grid is a memory rather than a snapshot of the latest look. Each step
    merges what was just observed into what was already there: a re-observed voxel
    is refreshed to a full age with its newest features and its count carried
    forward, one that was not seen ages by a step, and one whose age runs out is
    dropped. So the grid grows as the sensor uncovers more of a surface and
    shrinks where it has moved on, while ``count`` accumulates the evidence behind
    each voxel.
    """

    def __init__(
        self,
        voxel_size: float = 0.005,
        features: Mapping[str, Feature] | None = None,
        lifetime: int = 6,
    ) -> None:
        if lifetime < 1:
            raise ValueError(f"lifetime must be >= 1, got {lifetime}")
        self._voxel_size = voxel_size
        self._declared = dict(features) if features is not None else None
        self._lifetime = lifetime
        self._age_feature = NumericFeature(1, np.int32)
        self._count_feature = NumericFeature(1, np.int32)
        self._grid = VoxelGrid()

    @property
    def voxel_size(self) -> float:
        """Edge length of a voxel, in world units."""
        return self._voxel_size

    @property
    def lifetime(self) -> int:
        """How many steps a voxel survives without being re-observed."""
        return self._lifetime

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
            A mapping with the ``grid`` the tracker now holds, and the ``observed``
            grid built from this step's points alone.

        """
        features = dict(features) if features is not None else {}
        observed = self._build(points, features)
        # Age what is already held before folding in what was just seen, so that
        # a re-observed voxel's fresh row lands on top of the tick rather than
        # after it.
        aged = self._age(self._grid.data)
        merged = self._merge(aged, observed)
        self._grid = VoxelGrid(self._expire(merged))
        return {"grid": self._grid, "observed": observed}

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

    def reset(self) -> None:
        """Discard the current grid."""
        self._grid = VoxelGrid()

    def state_dict(self) -> Memento:
        """Export the current state for telemetry."""  # noqa: DOC201
        return {"grid": self._grid}

    def _age(self, remembered: pd.DataFrame) -> pd.DataFrame:
        """Tick every held voxel one step closer to expiring.

        Args:
            remembered: The voxels held going into this step.

        Returns:
            The frame with every age decremented by one.

        """
        if len(remembered) == 0:
            return remembered

        aged = remembered.copy()
        # Subtracting through the frame would widen the dtype, so write back the
        # declared one: age is meant to stay an integer count of steps.
        aged[AGE] = (aged[AGE] - 1).astype(self._age_feature.dtype)
        return aged

    def _merge(
        self, remembered: pd.DataFrame, observed: VoxelGrid
    ) -> pd.DataFrame:
        """Fold this step's observations into the voxels already held.

        A re-observed voxel is replaced wholesale by its fresh row, so its age
        returns to a full lifetime and its features reflect the latest look. Its
        ``count`` is the exception: sightings accumulate, so the fresh tally is
        added to the one already held. A voxel that was not seen this step is
        carried through untouched, keeping whatever age it arrived with.

        Args:
            remembered: The voxels held from earlier steps, already aged.
            observed: The grid built from this step's points alone.

        Returns:
            The merged frame, before expired voxels are dropped.

        """
        if len(remembered) == 0:
            return observed.data
        if len(observed) == 0:
            return remembered

        fresh = observed.data.copy()
        seen_before = fresh.index.intersection(remembered.index)
        if len(seen_before):
            fresh.loc[seen_before, COUNT] = (
                fresh.loc[seen_before, COUNT].to_numpy()
                + remembered.loc[seen_before, COUNT].to_numpy()
            ).astype(self._count_feature.dtype)

        # The fresh row wins outright, so drop the stale one it replaces.
        carried = remembered.drop(index=fresh.index, errors="ignore")
        if len(carried) == 0:
            return fresh
        return pd.concat([fresh, carried])

    @staticmethod
    def _expire(data: pd.DataFrame) -> pd.DataFrame:
        """Drop voxels whose age has run out.

        Args:
            data: A merged frame, possibly holding voxels aged past their end.

        Returns:
            The frame with expired rows removed.

        """
        if len(data) == 0:
            return data
        return data[data[AGE].to_numpy().ravel() > 0]

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

        # Age and count are the tracker's own, not reduced from the points. Every
        # voxel here was just observed, so it starts with a full lifetime and a
        # single sighting. Ageing and tallying happen in _merge, once this grid
        # meets the one already held.
        combined_features[AGE] = np.full(
            (len(voxels), *self._age_feature.shape),
            self._lifetime,
            dtype=self._age_feature.dtype,
        )
        combined_features[COUNT] = np.ones(
            (len(voxels), *self._count_feature.shape),
            dtype=self._count_feature.dtype,
        )

        return VoxelGrid.from_voxels_and_features(voxels, combined_features)

    def _features_to_add(
        self, values: Mapping[str, npt.NDArray]
    ) -> Mapping[str, Feature]:
        """Decide which features this tracker puts on a voxel.

        Returns:
            The declared features, or one inferred per supplied point feature.

        Raises:
            ValueError: If a declared feature has no values in ``values``, or if a
                point feature would collide with one the tracker manages.

        """
        if self._declared is None:
            resolved: Mapping[str, Feature] = {
                name: NumericFeature(
                    np.shape(np.asarray(value)[0]) or 1,
                    np.asarray(value).dtype,
                )
                for name, value in values.items()
            }
        else:
            missing = [name for name in self._declared if name not in values]
            if missing:
                raise ValueError(
                    f"no values supplied for feature(s): {', '.join(missing)}"
                )
            resolved = self._declared

        # The tracker writes these itself, so a point feature of the same name
        # would be silently overwritten. Say so instead.
        clashing = [name for name in TRACKED_FEATURES if name in resolved]
        if clashing:
            raise ValueError(
                f"{', '.join(repr(n) for n in clashing)} managed by the tracker "
                f"and cannot be point feature(s)"
            )
        return resolved
