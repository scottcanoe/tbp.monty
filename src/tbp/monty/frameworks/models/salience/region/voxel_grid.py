# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt

from tbp.monty.memento import Memento

Voxel = Tuple[int, int, int]


def voxel_coords(
    xyz_array: npt.NDArray[np.float64],
    voxel_size: float,
) -> npt.NDArray[np.int_]:
    """Quantize/bin 3D locations into integer-valued voxel indices.

    Args:
        xyz_array: World-coordinate points as an ``(..., 3)`` array.
        voxel_size: Edge length of a voxel, in world units.

    Returns:
        Integer voxel indices, same shape as ``xyz_array``.

    """
    return np.floor(xyz_array / voxel_size).astype(int)


class DecayingVoxelGrid:
    """Region estimate as a set of occupied voxels with decaying lifetimes.

    Each observed voxel has its lifetime (re)set to ``voxel_lifetime``. On every
    step, voxels that are not re-observed have their lifetime decremented and are
    dropped once it expires. The result is a short-horizon estimate of where the
    object has recently been observed.

    Implements the ``RegionTracker`` protocol.
    """

    def __init__(self, voxel_size: float = 0.001, voxel_lifetime: int = 6) -> None:
        if voxel_lifetime < 1:
            raise ValueError(f"voxel_lifetime must be >= 1, got {voxel_lifetime}")
        self._voxel_size = voxel_size
        self._voxel_lifetime = voxel_lifetime
        # Maps an occupied voxel to the number of steps it survives without being
        # re-observed.
        self._grid: dict[Voxel, int] = {}

    def observe(self, points: npt.NDArray[np.float64]) -> None:
        """Refresh observed voxels and decay the rest by one step.

        Args:
            points: Observed world-coordinate points as an ``(num_points, 3)``
                array. May be empty, in which case the grid only decays.

        """
        observed = self._voxels(points)
        aged = {
            voxel: lifetime - 1
            for voxel, lifetime in self._grid.items()
            if lifetime > 1
        }
        aged.update(dict.fromkeys(observed, self._voxel_lifetime))
        self._grid = aged

    def contains(self, locations: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Test which locations fall within an occupied voxel.

        Args:
            locations: World-coordinate points as an ``(num_locations, 3)`` array.

        Returns:
            Boolean array of shape ``(num_locations,)`` that is ``True`` wherever
            the corresponding location's voxel is currently in the grid.

        """
        if len(self._grid) == 0:
            return np.zeros(len(locations), dtype=bool)

        indices = voxel_coords(locations, self._voxel_size)
        return np.fromiter(
            (tuple(idx) in self._grid for idx in indices),
            dtype=bool,
            count=len(indices),
        )

    def reset(self) -> None:
        """Clear all occupied voxels."""
        self._grid.clear()

    def state_dict(self) -> Memento:
        """Flatten the voxel grid into arrays suitable for telemetry.

        Returns:
            A mapping with ``voxel_grid`` of shape ``(num_voxels, 3)`` holding
            integer voxel indices, and ``voxel_lifetimes`` of shape
            ``(num_voxels,)`` holding each voxel's remaining lifetime. Rows
            correspond elementwise.

        """
        voxels = np.array(list(self._grid.keys()), dtype=int).reshape(-1, 3)
        lifetimes = np.array(list(self._grid.values()), dtype=int)
        return {"voxel_grid": voxels, "voxel_lifetimes": lifetimes}

    def _voxels(self, points: npt.NDArray[np.float64]) -> set[Voxel]:
        """Quantize points to the set of occupied voxel indices.

        Returns:
            The set of voxel indices covered by ``points``, as (x, y, z) tuples.

        """
        if len(points) == 0:
            return set()
        return set(map(tuple, voxel_coords(points, self._voxel_size)))
