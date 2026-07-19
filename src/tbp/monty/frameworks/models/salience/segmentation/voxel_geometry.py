# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Spatial relationships between voxels.

Groups occupied voxels into distinct connected components by adjacency, which is
what lets regions observed across saccades be told apart and compared.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Iterable, Literal

import numpy as np
import numpy.typing as npt

from tbp.monty.frameworks.models.salience.segmentation.voxels import (
    Voxel,
)


def connected_components(
    voxels: Iterable[Voxel] | npt.NDArray[np.intp],
    connectivity: Literal[6, 18, 26] = 26,
) -> list[npt.NDArray[np.intp]]:
    """Partition voxels into connected components.

    Args:
        voxels: Voxel indices as an ``(num_voxels, 3)`` integer array.
        connectivity: Voxel adjacency, one of 6 (faces only), 18
            faces and edges), or 26 (faces, edges, and corners).

    Returns:
        A list of index arrays, one per component, indexing into ``voxels``.

    """
    voxel_array = np.asarray(voxels, dtype=np.intp)
    voxels = tuple(tuple(row) for row in voxel_array)
    n_voxels = len(voxels)
    if n_voxels == 0:
        return []

    voxel_to_idx: dict[Voxel, int] = {voxel: idx for idx, voxel in enumerate(voxels)}
    parent = list(range(n_voxels))

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
                voxel[0] + off[0],
                voxel[1] + off[1],
                voxel[2] + off[2],
            )
            j = voxel_to_idx.get(neighbor)
            if j is not None:
                union(i, j)

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n_voxels):
        components[find(i)].append(i)
    return [np.asarray(members, dtype=np.intp) for members in components.values()]


def _neighbor_offsets(connectivity: int) -> tuple[Voxel]:
    """Return the neighbor offsets for a given voxel connectivity.

    Returns:
        The list of ``(dx, dy, dz)`` offsets, excluding the origin.

    Raises:
        ValueError: If ``connectivity`` is not one of 6, 18, or 26.

    """
    if connectivity not in (6, 18, 26):
        raise ValueError(
            f"connectivity must be one of 6, 18, or 26, got {connectivity}"
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
