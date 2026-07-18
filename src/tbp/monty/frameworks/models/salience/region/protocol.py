# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from tbp.monty.memento import Memento


class RegionTracker(Protocol):
    """Builds and maintains an estimate of a region of 3D space over time.

    A ``RegionTracker`` accumulates observations of world-coordinate points across
    steps into a stable estimate of one or more spatial regions. It owns every
    implementation detail of how points are represented (e.g. voxels), how new
    observations are merged into the existing estimate, and what statistics are
    tracked about the region(s).

    Consumers only ever feed it observed points and query whether locations fall
    within the current estimate, which keeps the tracker decoupled from sensor and
    observation formats and lets different representations/strategies be swapped in.
    """

    def observe(self, points: npt.NDArray[np.float64]) -> None:
        """Incorporate newly observed points and advance the estimate by one step.

        How the estimate advances (decay, accumulation, merging, ...) is entirely
        up to the implementation.

        Args:
            points: World-coordinate points as an ``(num_points, 3)`` array. May be
                empty, in which case the estimate is advanced with no new evidence.

        """
        ...

    def contains(self, locations: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Test which query locations fall within the current region estimate.

        Args:
            locations: World-coordinate points as an ``(num_locations, 3)`` array.

        Returns:
            Boolean array of shape ``(num_locations,)``, ``True`` where the
            corresponding location falls within the region estimate.

        """
        ...

    def reset(self) -> None:
        """Clear all accumulated region state."""
        ...

    def state_dict(self) -> Memento:
        """Export the current region state (and any statistics) for telemetry.

        Returns:
            A mapping of region state suitable for snapshotting/logging.

        """
        ...
