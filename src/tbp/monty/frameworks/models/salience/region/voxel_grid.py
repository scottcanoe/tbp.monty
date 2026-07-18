# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Tuple

import numpy as np
import numpy.typing as npt

from tbp.monty.frameworks.models.salience.region.channels import (
    Count,
    Decay,
    Mean,
    VoxelChannel,
)
from tbp.monty.frameworks.models.salience.region.regions import (
    Region,
    connected_components,
)
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


class VoxelGrid:
    """Region estimate as occupied voxels carrying named metadata channels.

    Observed points are quantized to voxels, and each voxel accumulates a set of
    named channels (see :mod:`.channels`) -- age, point count, and any per-point
    features such as salience, depth, or color. One channel is designated the
    lifecycle channel; it decides which voxels remain alive, and all other channels
    are evicted in lockstep with it.

    Implements the ``RegionTracker`` protocol.
    """

    def __init__(
        self,
        voxel_size: float = 0.001,
        channels: Mapping[str, VoxelChannel] | None = None,
        lifecycle: str = "age",
    ) -> None:
        self._voxel_size = voxel_size
        self._channels: dict[str, VoxelChannel] = dict(
            channels if channels is not None else default_channels()
        )
        if lifecycle not in self._channels:
            raise ValueError(
                f"lifecycle channel {lifecycle!r} not in channels "
                f"{sorted(self._channels)}"
            )
        self._lifecycle = lifecycle
        self._live: set[Voxel] = set()

    def observe(
        self,
        points: npt.NDArray[np.float64],
        features: Mapping[str, npt.NDArray] | None = None,
    ) -> None:
        """Fold this step's points and their features into the grid.

        Args:
            points: Observed world-coordinate points as an ``(num_points, 3)``
                array. May be empty, in which case the grid only decays.
            features: Point features aligned with ``points``, keyed by channel
                name (e.g. ``{"salience": (N,), "rgba": (N, 4)}``).

        """
        features = features or {}
        voxels, groups = self._group(points)

        for name, channel in self._channels.items():
            channel.update(voxels, groups, features.get(name))

        survivors = self._channels[self._lifecycle].survivors()
        for channel in self._channels.values():
            channel.evict(survivors)
        self._live = survivors

    def contains(self, locations: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Test which locations fall within an occupied voxel.

        Args:
            locations: World-coordinate points as an ``(num_locations, 3)`` array.

        Returns:
            Boolean array of shape ``(num_locations,)``, ``True`` where the
            location's voxel is currently occupied.

        """
        if not self._live:
            return np.zeros(len(locations), dtype=bool)
        indices = voxel_coords(locations, self._voxel_size)
        return np.fromiter(
            (tuple(idx) in self._live for idx in indices),
            dtype=bool,
            count=len(indices),
        )

    def regions(self, connectivity: int = 26) -> list[Region]:
        """Group occupied voxels into distinct connected regions.

        Args:
            connectivity: Voxel adjacency, one of 6, 18, or 26.

        Returns:
            One :class:`Region` per connected component, each carrying the
            aggregated statistics of every channel over its voxels.

        """
        voxels = self._sorted_voxels()
        if len(voxels) == 0:
            return []
        components = connected_components(voxels, connectivity)
        regions = []
        for members in components:
            member_voxels = [tuple(int(c) for c in voxels[i]) for i in members]
            stats = {
                name: channel.aggregate(member_voxels)
                for name, channel in self._channels.items()
            }
            regions.append(Region(voxels=voxels[members], stats=stats))
        return regions

    def reset(self) -> None:
        """Clear all occupied voxels and channel state."""
        self._live = set()
        # Channels evict everything once nothing survives.
        for channel in self._channels.values():
            channel.evict(set())

    def state_dict(self) -> Memento:
        """Export occupied voxels and every channel value for telemetry.

        Returns:
            A mapping with ``voxels`` of shape ``(num_voxels, 3)`` and one entry
            per channel giving that channel's per-voxel values (row-aligned with
            ``voxels``).

        """
        voxels = self._sorted_voxels()
        keys = [tuple(int(c) for c in voxel) for voxel in voxels]
        out: dict = {"voxels": voxels}
        for name, channel in self._channels.items():
            out[name] = channel.values(keys)
        return out

    def _group(
        self, points: npt.NDArray[np.float64]
    ) -> tuple[list[Voxel], list[npt.NDArray[np.intp]]]:
        """Group points by the voxel they fall in.

        Returns:
            A ``(voxels, groups)`` pair: the unique observed voxels, and for each
            the indices of the points that landed in it.

        """
        if len(points) == 0:
            return [], []
        indices = voxel_coords(points, self._voxel_size)
        buckets: dict[Voxel, list[int]] = defaultdict(list)
        for i, idx in enumerate(indices):
            buckets[tuple(int(c) for c in idx)].append(i)
        voxels = list(buckets)
        groups = [np.asarray(members, dtype=np.intp) for members in buckets.values()]
        return voxels, groups

    def _sorted_voxels(self) -> npt.NDArray[np.int_]:
        """Return live voxels as a deterministically ordered ``(V, 3)`` array.

        Returns:
            Sorted voxel indices; shape ``(0, 3)`` when empty.

        """
        return np.array(sorted(self._live), dtype=int).reshape(-1, 3)


def default_channels() -> dict[str, VoxelChannel]:
    """Return the default channel set: age, count, and salience/depth/color means.

    Returns:
        A fresh mapping of channel name to channel, matching the features the
        :class:`~...sensor_module.SalienceSM` extracts.

    """
    return {
        "age": Decay(),
        "count": Count(),
        "salience": Mean(),
        "depth": Mean(),
        "rgba": Mean(),
    }
