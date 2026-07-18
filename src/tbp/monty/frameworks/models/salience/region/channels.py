# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Per-voxel metadata channels.

A :class:`VoxelChannel` maintains a single column of per-voxel metadata as voxels
are observed over time. Age is just one channel (a decaying lifetime); features
associated with observed points (salience, depth, color, ...) are others. A grid
composes a named set of channels, so extending the metadata carried by a voxel is
a matter of adding a channel rather than changing the grid.

Each channel folds observations along two axes:

* spatially, over the points that land in a voxel on a given step, and
* temporally, as a voxel is re-observed across steps.

It also knows how to reduce itself over a set of voxels into a single region-level
value, which is what lets distinct regions be compared.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, Tuple

import numpy as np
import numpy.typing as npt

Voxel = Tuple[int, int, int]

PointGroups = Sequence[npt.NDArray[np.intp]]
"""For each observed voxel, the indices of the points that fell within it."""


class VoxelChannel(Protocol):
    """One column of per-voxel metadata, accumulated over time."""

    def update(
        self,
        voxels: Sequence[Voxel],
        groups: PointGroups,
        values: npt.NDArray | None,
    ) -> None:
        """Fold this step's observations into the channel.

        Args:
            voxels: The unique voxels observed this step.
            groups: For each voxel, the indices of the points that fell within it.
            values: Feature values aligned with the points as an ``(num_points,
                ...)`` array, or ``None`` for channels that consume no feature
                (e.g. age, count).

        """
        ...

    def evict(self, survivors: set[Voxel]) -> None:
        """Drop per-voxel state for any voxel not in ``survivors``."""
        ...

    def values(self, voxels: Sequence[Voxel]) -> npt.NDArray:
        """Return this channel's value for each voxel in ``voxels``.

        Args:
            voxels: The voxels to read, in the desired output order.

        Returns:
            An array whose leading axis matches ``voxels``.

        """
        ...



class Decay(VoxelChannel):
    """Lifetime that resets to ``lifetime`` on observation and decays otherwise.

    Voxels not re-observed on a step have their lifetime decremented and expire
    once it reaches zero, giving a short-horizon estimate of recent occupancy.
    """

    def __init__(self, lifetime: int = 6) -> None:
        if lifetime < 1:
            raise ValueError(f"lifetime must be >= 1, got {lifetime}")
        self._lifetime = lifetime
        self._age: dict[Voxel, int] = {}

    def update(
        self,
        voxels: Sequence[Voxel],
        groups: PointGroups,  # noqa: ARG002
        values: npt.NDArray | None,  # noqa: ARG002
    ) -> None:
        self._age = {voxel: age - 1 for voxel, age in self._age.items()}
        for voxel in voxels:
            self._age[voxel] = self._lifetime

    def survivors(self) -> set[Voxel]:
        return {voxel for voxel, age in self._age.items() if age > 0}

    def evict(self, survivors: set[Voxel]) -> None:
        self._age = {
            voxel: age for voxel, age in self._age.items() if voxel in survivors
        }

    def values(self, voxels: Sequence[Voxel]) -> npt.NDArray:
        """Return the remaining lifetime of each voxel.

        Returns:
            Integer lifetimes with shape ``(len(voxels),)``.

        """
        return np.array([self._age[voxel] for voxel in voxels], dtype=int)



class Count(VoxelChannel):
    """Total number of points ever accumulated into each voxel."""

    def __init__(self) -> None:
        self._count: dict[Voxel, int] = {}

    def update(
        self,
        voxels: Sequence[Voxel],
        groups: PointGroups,
        values: npt.NDArray | None,  # noqa: ARG002
    ) -> None:
        for voxel, group in zip(voxels, groups):
            self._count[voxel] = self._count.get(voxel, 0) + len(group)

    def evict(self, survivors: set[Voxel]) -> None:
        self._count = {
            voxel: count for voxel, count in self._count.items() if voxel in survivors
        }

    def values(self, voxels: Sequence[Voxel]) -> npt.NDArray:
        """Return the accumulated point count of each voxel.

        Returns:
            Integer counts with shape ``(len(voxels),)``.

        """
        return np.array([self._count[voxel] for voxel in voxels], dtype=int)



class Mean(VoxelChannel):
    """Running mean of a point feature, exact via accumulated sums and counts.

    Stores per-voxel feature sums and counts, so both the per-voxel mean and a
    count-weighted region mean are exact. Works for scalar features (salience,
    depth) and vector features (e.g. RGBA color).
    """

    def __init__(self) -> None:
        self._sum: dict[Voxel, npt.NDArray[np.float64]] = {}
        self._n: dict[Voxel, int] = {}

    def update(
        self,
        voxels: Sequence[Voxel],
        groups: PointGroups,
        values: npt.NDArray | None,
    ) -> None:
        if values is None:
            raise ValueError("Mean channel requires point feature values, got None")
        for voxel, group in zip(voxels, groups):
            group_sum = np.asarray(values[group], dtype=float).reshape(len(group), -1)
            total = group_sum.sum(axis=0)
            if voxel in self._sum:
                self._sum[voxel] = self._sum[voxel] + total
                self._n[voxel] += len(group)
            else:
                self._sum[voxel] = total
                self._n[voxel] = len(group)

    def evict(self, survivors: set[Voxel]) -> None:
        self._sum = {v: s for v, s in self._sum.items() if v in survivors}
        self._n = {v: n for v, n in self._n.items() if v in survivors}

    def values(self, voxels: Sequence[Voxel]) -> npt.NDArray:
        """Return the mean feature value of each voxel.

        Returns:
            A ``(len(voxels), feature_dim)`` array of per-voxel means, or an empty
            array when ``voxels`` is empty.

        """
        if not voxels:
            return np.empty((0,))
        return np.stack([self._sum[v] / self._n[v] for v in voxels])

