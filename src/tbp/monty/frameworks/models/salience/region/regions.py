# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Grouping occupied voxels into distinct spatial regions.

A region is a connected component of occupied voxels. Regions are recomputed from
the current occupancy each time they are requested (they carry no identity across
steps yet); each region exposes the per-channel statistics of the voxels it
contains, which is what allows regions to be compared.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import numpy.typing as npt

Voxel = Tuple[int, int, int]

_CONNECTIVITIES = (6, 18, 26)


@dataclass
class Region:
    """A distinct region of space as a set of connected voxels.

    Attributes:
        voxels: The region's voxel indices as an ``(num_voxels, 3)`` array.
        stats: Per-channel region-level statistics, keyed by channel name (e.g.
            a mean color, a total point count).

    """

    voxels: npt.NDArray[np.int_]
    stats: dict[str, Any]


def _neighbor_offsets(connectivity: int) -> list[Voxel]:
    """Return the neighbor offsets for a given voxel connectivity.

    Returns:
        The list of ``(dx, dy, dz)`` offsets, excluding the origin.

    Raises:
        ValueError: If ``connectivity`` is not one of 6, 18, or 26.

    """
    if connectivity not in _CONNECTIVITIES:
        raise ValueError(
            f"connectivity must be one of {_CONNECTIVITIES}, got {connectivity}"
        )
    offsets = []
    for offset in itertools.product((-1, 0, 1), repeat=3):
        manhattan = sum(abs(o) for o in offset)
        if manhattan == 0:
            continue
        if connectivity == 6 and manhattan > 1:
            continue
        if connectivity == 18 and manhattan > 2:
            continue
        offsets.append(offset)
    return offsets


def connected_components(
    voxels: npt.NDArray[np.int_],
    connectivity: int = 26,
) -> list[npt.NDArray[np.intp]]:
    """Partition voxels into connected components.

    Args:
        voxels: Voxel indices as an ``(num_voxels, 3)`` integer array.
        connectivity: Voxel adjacency, one of 6, 18, or 26.

    Returns:
        A list of index arrays, one per component, indexing into ``voxels``.

    """
    n = len(voxels)
    if n == 0:
        return []

    index = {tuple(int(c) for c in voxel): i for i, voxel in enumerate(voxels)}
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    offsets = _neighbor_offsets(connectivity)
    for i, voxel in enumerate(voxels):
        for off in offsets:
            neighbor = (
                int(voxel[0]) + off[0],
                int(voxel[1]) + off[1],
                int(voxel[2]) + off[2],
            )
            j = index.get(neighbor)
            if j is not None:
                union(i, j)

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        components[find(i)].append(i)
    return [np.asarray(members, dtype=np.intp) for members in components.values()]
