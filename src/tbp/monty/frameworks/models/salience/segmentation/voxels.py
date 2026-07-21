# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Voxelized estimate of the regions of space an object occupies.

Observed points are quantized into voxels, each of which carries a set of named
features (see :class:`Feature`). A :class:`VoxelGrid` is an immutable snapshot of
the occupied voxels and their features, indexed by voxel coordinate on the rows
and by ``(feature, component)`` on the columns.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Protocol, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from tbp.monty.frameworks.models.buffer import BufferEncoder

Voxel = Tuple[int, int, int]  # hashable voxel coordinates.


def as_array_voxels(voxels: Iterable) -> npt.NDArray[np.int_]:
    array = np.asarray(voxels, dtype=int)
    array = array.reshape(-1, 3) if array.ndim == 1 else array
    assert array.ndim == 2 and array.shape[1] == 3
    return array


def as_tuple_voxels(voxels: Iterable) -> tuple[Voxel, ...]:
    # Coerce to builtin ints: numpy scalars are not JSON-serializable, and these
    # tuples end up in telemetry.
    array_voxels = as_array_voxels(voxels)  # This does validation for us.
    return tuple(tuple(int(c) for c in row) for row in array_voxels)


def voxelize_and_bin_points(
    points: npt.NDArray[np.floating],
    voxel_size: float,
) -> dict[Voxel, list[int]]:
    """Quantize/bin 3D locations into voxels.

    Args:
        points: An array containing (x, y, z) coordinates.
        voxel_size: Edge length of a voxel.

    Returns:
        covoxel_points: A dictionary from voxel (x, y, z) to the the indices of the
            points inside it. (indices index into ``points``.) The keys of this
            dictionary are all the occupied voxels.

    """
    # as_tuple_voxels validates the shape and coerces to builtin ints, which these
    # keys need since they become index labels and end up in telemetry.
    voxels = as_tuple_voxels(np.floor(points / voxel_size))
    covoxel_points: dict[Voxel, list[int]] = defaultdict(list)
    for point_ind, voxel in enumerate(voxels):
        covoxel_points[voxel].append(point_ind)

    return covoxel_points


class Feature(Protocol):
    """What a per-voxel value is, and what can be done with values of its kind.

    A feature is unnamed: the name under which it is stored and supplied is the
    key it is registered against, not a property of the feature itself. What makes
    a feature meaningful is the pair of operations it provides -- :meth:`reduce`,
    to summarize the many observations that fall in one voxel, and
    :meth:`distance`, to compare two values so that voxels and regions can be
    compared.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of a single voxel's value, e.g. ``(3,)`` for RGB."""
        ...

    @property
    def dtype(self) -> np.dtype:
        """The dtype values of this feature are stored as."""
        ...

    def reduce(self, values: npt.NDArray) -> npt.NDArray:
        """Summarize many observations of this feature into a single value.

        Args:
            values: Observed values as a ``(num_observations, *shape)`` array.

        Returns:
            The summary value, of shape ``shape``.

        """
        ...

    def distance(self, a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
        """Measure how far apart two values of this feature are.

        Args:
            a: Values broadcastable to ``(..., *shape)``.
            b: Values broadcastable to ``(..., *shape)``.

        Returns:
            Distances with the feature's own axes reduced away, so comparing two
            single values yields a scalar and comparing two stacks yields ``(N,)``.

        """
        ...


class NumericFeature(Feature):
    """A numeric feature summarized by its mean and compared by Euclidean distance.

    Suits continuous quantities -- salience, depth, colour. Features needing other
    semantics (angles, labels) implement :class:`Feature` with their own
    :meth:`reduce` and :meth:`distance`.
    """

    def __init__(
        self,
        shape: int | Sequence[int] = 1,
        dtype: npt.DTypeLike = np.float32,
    ) -> None:
        if isinstance(shape, (int, np.integer)):
            shape = (int(shape),)
        else:
            shape = tuple(int(s) for s in shape)
        assert len(shape) > 0 and all(s > 0 for s in shape), (
            f"shape must be non-empty and positive, got {shape}"
        )
        self._shape = shape
        self._dtype = np.dtype(dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def distance(self, a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
        """Euclidean distance between two values.

        Returns:
            The distance, with this feature's own axes reduced away.

        """
        delta = a - b
        axes = tuple(range(-len(self._shape), 0))
        return np.sqrt(np.sum(delta**2, axis=axes)).astype(self._dtype)

    def reduce(self, values: npt.NDArray) -> float | npt.NDArray:
        """Average the observed values.

        Returns:
            The mean value, of shape ``shape`` and this feature's dtype.

        """
        stacked = values.reshape(len(values), -1)
        return stacked.mean(axis=0).reshape(self._shape).astype(self._dtype)



VOXEL_LEVELS = ("x", "y", "z")
"""Names of the row index levels: a voxel's integer grid coordinate."""

FEATURE_LEVELS = ("feature", "component")
"""Names of the column index levels: a feature and which of its components."""


def empty_voxel_frame() -> pd.DataFrame:
    """Build the empty frame a grid holds when no voxels are occupied.

    Returns:
        A frame with no rows or columns, but with both index levels in place.

    """
    return pd.DataFrame(
        index=pd.MultiIndex.from_tuples([], names=VOXEL_LEVELS),
        columns=pd.MultiIndex.from_tuples([], names=FEATURE_LEVELS),
    )


class VoxelGrid:
    """A set of voxels and their features/metadata."""

    def __init__(self, data: pd.DataFrame | None = None) -> None:
        if data is None:
            data = empty_voxel_frame()
        else:
            assert isinstance(data, pd.DataFrame)
            assert isinstance(data.index, pd.MultiIndex)
            assert tuple(data.index.names) == VOXEL_LEVELS
            assert isinstance(data.columns, pd.MultiIndex)
            assert tuple(data.columns.names) == FEATURE_LEVELS
        self._data = data

    def __len__(self) -> int:
        """Return the number of occupied voxels.

        Returns:
            How many voxels the grid holds.

        """
        return len(self._data)

    def copy(self) -> VoxelGrid:
        """Return an independent copy of this grid.

        A grid is a snapshot, so callers that hold on to one -- telemetry, most
        of all -- need it detached from whatever the tracker does next.

        Returns:
            A grid over a copy of this one's data.

        """
        return VoxelGrid(self._data.copy())

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def voxels(self) -> npt.NDArray[np.int_]:
        """Occupied voxel coordinates as an ``(num_voxels, 3)`` array."""
        if len(self._data) == 0:
            return np.empty((0, 3), dtype=int)
        return as_array_voxels(self._data.index.tolist())

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The names of the features every voxel carries."""
        return tuple(self._data.columns.get_level_values("feature").unique())

    @classmethod
    def from_voxels_and_features(
        cls,
        voxels: Sequence[Sequence[int]],
        features: Mapping[str, Sequence] | None = None,
    ) -> VoxelGrid:
        """Build a grid from voxel coordinates and their row-aligned features.

        Each feature is flattened to ``(num_voxels, num_components)``, so a scalar
        feature occupies one column and an RGBA feature occupies four.

        Args:
            voxels: Voxel coordinates, one row per voxel.
            features: Feature values keyed by name, each with one row per voxel.

        Returns:
            The grid holding those voxels and features.

        """
        if len(voxels) == 0:
            return cls()
        features = {} if features is None else dict(features)
        voxels = as_tuple_voxels(voxels)
        n_voxels = len(voxels)
        columns = {}
        for feat_name, feat_data in features.items():
            feat_data = np.asarray(feat_data)  # noqa: PLW2901
            assert len(feat_data) == n_voxels, (
                f"feature '{feat_name}' has {len(feat_data)} rows, "
                f"but voxels has {n_voxels}"
            )
            # Build one column per component so each keeps its own dtype.
            flat = feat_data.reshape(n_voxels, -1)
            for component in range(flat.shape[1]):
                columns[feat_name, component] = flat[:, component]

        index = pd.MultiIndex.from_tuples(voxels, names=VOXEL_LEVELS)
        if not columns:
            return cls(empty_voxel_frame().reindex(index))
        data = pd.DataFrame(columns, index=index)
        data.columns = data.columns.set_names(FEATURE_LEVELS)
        return cls(data)


def encode_voxel_grid(grid: VoxelGrid) -> dict:
    """Encode a grid for JSON serialization.

    Args:
        grid: The grid to encode.

    Returns:
        A dictionary with voxel coordinates and encoded features.

    """
    df = grid.data
    if len(df) == 0:
        return {"voxels": np.empty((0, 3), dtype=int), "features": {}}

    # Arrays are handed back as-is: BufferEncoder already knows how to encode an
    # ndarray, and will recurse into this mapping to do so.
    voxels = np.array(df.index.to_numpy().tolist())
    features: dict[str, npt.NDArray] = {}
    for feat_name in df.columns.get_level_values("feature").unique():
        # Flatten to (num_voxels, num_components) so each component is a column.
        features[str(feat_name)] = df[feat_name].to_numpy().reshape(len(df), -1)

    return {"voxels": voxels, "features": features}


BufferEncoder.register(VoxelGrid, encode_voxel_grid)
