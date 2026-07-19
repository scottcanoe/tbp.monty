# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest

import numpy as np

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.salience.segmentation.strategies.random_walker import (
    RandomWalker,
)

PATCH = 64


def centred_square_patch(half: int = 12) -> np.ndarray:
    """Build a patch with a bright square centred on the fixation.

    Returns:
        An ``(PATCH, PATCH, 4)`` uint8 frame, dark except for a bright square.

    """
    rgba = np.zeros((PATCH, PATCH, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    centre = PATCH // 2
    rgba[centre - half : centre + half, centre - half : centre + half, :3] = 255
    return rgba


class RandomWalkerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = RuntimeContext(rng=np.random.RandomState(0))
        self.strategy = RandomWalker()

    def test_returns_a_uint8_mask_matching_the_frame(self) -> None:
        mask = self.strategy(ctx=self.ctx, rgba=centred_square_patch())
        self.assertEqual(mask.shape, (PATCH, PATCH))
        self.assertEqual(mask.dtype, np.uint8)

    def test_mask_is_binary(self) -> None:
        mask = self.strategy(ctx=self.ctx, rgba=centred_square_patch())
        self.assertEqual(set(np.unique(mask)).issubset({0, 1}), True)

    def test_segments_the_object_under_the_fixation(self) -> None:
        mask = self.strategy(ctx=self.ctx, rgba=centred_square_patch())
        centre = PATCH // 2
        self.assertEqual(mask[centre, centre], 1)

    def test_does_not_segment_the_far_surroundings(self) -> None:
        mask = self.strategy(ctx=self.ctx, rgba=centred_square_patch())
        self.assertEqual(mask[0, 0], 0)

    def test_the_segmented_region_follows_the_object_boundary(self) -> None:
        # The bright square spans 24px; the mask should cover much of it without
        # bleeding across the whole 64px frame.
        mask = self.strategy(ctx=self.ctx, rgba=centred_square_patch(half=12))
        covered = int(mask.sum())
        self.assertGreater(covered, 100)
        self.assertLess(covered, PATCH * PATCH // 2)

    def test_accepts_the_optional_protocol_arguments(self) -> None:
        mask = self.strategy(
            ctx=self.ctx,
            rgba=centred_square_patch(),
            depth=np.ones((PATCH, PATCH)),
            locations=np.zeros((PATCH * PATCH, 3)),
        )
        self.assertEqual(mask.shape, (PATCH, PATCH))

    def test_a_uniform_frame_does_not_raise(self) -> None:
        # std is zero, so the edge weighting degenerates; it must still solve.
        uniform = np.full((PATCH, PATCH, 4), 128, dtype=np.uint8)
        mask = self.strategy(ctx=self.ctx, rgba=uniform)
        self.assertEqual(mask.shape, (PATCH, PATCH))

    def test_threshold_controls_how_much_is_committed(self) -> None:
        patch = centred_square_patch()
        permissive = RandomWalker(threshold=0.1)(ctx=self.ctx, rgba=patch)
        strict = RandomWalker(threshold=0.9)(ctx=self.ctx, rgba=patch)
        self.assertGreaterEqual(int(permissive.sum()), int(strict.sum()))

    def test_a_larger_ring_selects_at_least_as_much(self) -> None:
        # region_radius is the scale dial: a wider ring admits a larger region.
        patch = centred_square_patch(half=20)
        small = RandomWalker(region_radius=0.2)(ctx=self.ctx, rgba=patch)
        large = RandomWalker(region_radius=0.45)(ctx=self.ctx, rgba=patch)
        self.assertGreaterEqual(int(large.sum()), int(small.sum()))


class RandomWalkerFrameSizeTest(unittest.TestCase):
    """The ring must stay inside the frame whatever the sensor resolution."""

    def setUp(self) -> None:
        self.ctx = RuntimeContext(rng=np.random.RandomState(0))

    def test_works_across_sensor_resolutions(self) -> None:
        for size in (16, 32, 64, 128):
            rgba = np.zeros((size, size, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            quarter = size // 4
            rgba[quarter : size - quarter, quarter : size - quarter, :3] = 255
            mask = RandomWalker()(ctx=self.ctx, rgba=rgba)
            self.assertEqual(mask.shape, (size, size))
            self.assertEqual(mask[size // 2, size // 2], 1)

    def test_downscales_a_frame_larger_than_the_resolution_cap(self) -> None:
        size = 96
        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        rgba[24:72, 24:72, :3] = 255
        mask = RandomWalker(max_dim=48)(ctx=self.ctx, rgba=rgba)
        self.assertEqual(mask.shape, (size, size))
        self.assertEqual(mask[size // 2, size // 2], 1)

    def test_a_non_square_frame_keeps_its_shape(self) -> None:
        rgba = np.zeros((32, 64, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        rgba[8:24, 24:40, :3] = 255
        mask = RandomWalker()(ctx=self.ctx, rgba=rgba)
        self.assertEqual(mask.shape, (32, 64))
