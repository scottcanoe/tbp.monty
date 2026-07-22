# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Seeded random-walker segmentation (Grady 2006).

Seeds the fixation as foreground and a ring *around* the fixation as background,
then solves the combinatorial Dirichlet problem (a single sparse linear system on
the image graph): every pixel gets the probability that a random walker released
there reaches the foreground seed before the surrounding ring.

Two design choices make it follow boundaries rather than produce a diffuse blob:

* **Ring background, not the frame border.** With the competing seed on a ring
  around the fixation, the nearest enclosing contour sits *between* the two seeds
  and becomes a real barrier the walker must cross. This yields the local region
  bounded by the fixation's own edges -- the part, not the whole object -- and the
  ring radius is a continuous scale dial.
* **Std-normalized edge sensitivity.** skimage scales its edge weights by the
  image's global standard deviation, so a raw ``beta`` means something different
  on every image. Dividing that std back out makes ``beta`` image-independent.

The underlying problem is linear with a unique solution depending continuously on
the edge weights, so the field shifts smoothly as the sensor moves rather than
flipping discontinuously the way a thresholded or greedy method does.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np
import numpy.typing as npt
from skimage.segmentation import random_walker

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.salience.segmentation.protocol import (
    SegmentationStrategy,
)


class RandomWalker(SegmentationStrategy):
    """Random-walker segmentation of the local region at the sensor's fixation.

    The sensor patch is centred on what it fixates, so the foreground seed is
    placed at the centre of the frame and the background ring around it.

    Radii are fractions of the frame's shorter side rather than pixel counts:
    sensor resolution is configurable, and the ring must stay inside the frame for
    the ring-versus-fixation mechanism to work at all.
    """

    def __init__(
        self,
        beta: float = 350.0,
        region_radius: float = 0.3,
        fg_radius: float = 0.015,
        smooth_sigma: float = 1.0,
        max_dim: int = 384,
        threshold: float = 0.5,
    ) -> None:
        """Initialize the random-walker segmentation strategy.

        Args:
            beta: How strongly colour edges block the walker, image-independent.
                Higher snaps the field to boundaries; lower diffuses more smoothly.
            region_radius: Radius of the background ring, as a fraction of the
                frame's shorter side. The scale dial: small selects just the part
                under the fixation, large the whole surface it sits on. Clamped to
                keep the ring inside the frame.
            fg_radius: Radius of the foreground seed disk at the fixation, as a
                fraction of the frame's shorter side.
            smooth_sigma: Gaussian blur applied before solving, to stop the field
                chasing texture. 0 disables it.
            max_dim: Solve at up to this long-side resolution, upscaling the field
                afterwards. The linear solve cost grows with pixel count.
            threshold: Foreground probability at or above which a pixel is
                segmented. The solve produces a soft membership field; this is
                where it is committed to a mask.

        """
        self._beta = beta
        self._region_radius = region_radius
        self._fg_radius = fg_radius
        self._smooth_sigma = smooth_sigma
        self._max_dim = max_dim
        self._threshold = threshold

    def __call__(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        rgba: npt.NDArray[np.uint8],
        depth: npt.NDArray[np.float64] | None = None,  # noqa: ARG002
        locations: npt.NDArray[np.float64] | None = None,  # noqa: ARG002
    ) -> npt.NDArray[np.uint8]:
        """Segment the region containing the centre of the frame.

        Args:
            ctx: The runtime context.
            rgba: The observed frame.
            depth: Unused.
            locations: Unused.

        Returns:
            A binary mask of the same height and width as ``rgba``, non-zero where
            the walker reaches the fixation seed with at least ``threshold``
            probability.

        """
        image = rgba[..., :3]
        height, width = image.shape[:2]
        # The sensor patch is centred on what it fixates.
        y, x = height // 2, width // 2

        work, solve_width, solve_height, seed_x, seed_y, scale = self._downscale(
            image, x, y
        )

        work = work.astype(np.float32) / 255.0
        if self._smooth_sigma > 0:
            work = cv2.GaussianBlur(work, (0, 0), float(self._smooth_sigma))

        # Divide skimage's internal std-normalization back out, so `beta` is
        # image-independent.
        beta = self._beta * 10.0 * float(work.std()) * np.sqrt(work.shape[-1])

        markers = self._markers(solve_height, solve_width, seed_x, seed_y)

        with warnings.catch_warnings():
            # The CG solve can nudge probabilities a hair outside [0, 1]; we clip,
            # so silence that specific tolerance warning.
            warnings.filterwarnings("ignore", message=".*probability range.*")
            probabilities = random_walker(
                work,
                markers,
                beta=beta,
                mode="cg_j",
                return_full_prob=True,
                channel_axis=-1,
            )

        # Labels are sorted ascending, so index 0 is label 1 (the foreground).
        foreground = np.clip(probabilities[0], 0.0, 1.0)
        if scale < 1.0:
            foreground = cv2.resize(
                foreground, (width, height), interpolation=cv2.INTER_LINEAR
            )
        return (foreground >= self._threshold).astype(np.uint8)

    def _downscale(
        self, image: npt.NDArray[np.uint8], x: int, y: int
    ) -> tuple[npt.NDArray, int, int, int, int, float]:
        """Cap the solve resolution, carrying the fixation along with it.

        Returns:
            The image to solve on, its width and height, the fixation within it,
            and the scale that was applied.

        """
        height, width = image.shape[:2]
        scale = min(1.0, float(self._max_dim) / max(height, width))
        if scale >= 1.0:
            return image, width, height, x, y, 1.0

        solve_width = max(1, round(width * scale))
        solve_height = max(1, round(height * scale))
        work = cv2.resize(
            image, (solve_width, solve_height), interpolation=cv2.INTER_AREA
        )
        seed_x = int(np.clip(round(x * scale), 0, solve_width - 1))
        seed_y = int(np.clip(round(y * scale), 0, solve_height - 1))
        return work, solve_width, solve_height, seed_x, seed_y, scale

    def _markers(
        self, height: int, width: int, seed_x: int, seed_y: int
    ) -> npt.NDArray[np.int32]:
        """Seed the fixation disk as foreground and a ring around it as background.

        Returns:
            An ``(height, width)`` label image: 0 unlabeled, 1 foreground,
            2 background.

        """
        # Radii are relative to the shorter side, and the ring is clamped to stay
        # inside the frame: a ring that falls off-image would leave the walker
        # competing against the corners instead of the surrounding contour.
        short_side = min(height, width)
        fg_radius = max(1, round(self._fg_radius * short_side))
        ring_radius = max(fg_radius + 2, round(self._region_radius * short_side))
        ring_radius = min(ring_radius, short_side // 2 - 1)

        markers = np.zeros((height, width), dtype=np.int32)
        cv2.circle(markers, (seed_x, seed_y), ring_radius, 2, 3)
        cv2.circle(markers, (seed_x, seed_y), fg_radius, 1, -1)

        # If the ring did not land (a frame too small to hold it), fall back to the
        # corners so a background label still exists.
        if not (markers == 2).any():
            markers[0, 0] = markers[0, -1] = markers[-1, 0] = markers[-1, -1] = 2
        if not (markers == 1).any():
            markers[seed_y, seed_x] = 1
        return markers
