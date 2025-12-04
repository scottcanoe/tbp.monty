# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import cv2
import numpy as np


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2Lab).astype(np.float32) / 255.0


def lab_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_Lab2RGB)


def rgb_to_opponent(image: np.ndarray) -> np.ndarray:
    R, G, B = cv2.split(image.astype(np.float32))
    L = (R + G + B) / (3 * 255.0)
    a = (R - G + 255.0) / (2 * 255.0)
    b = (B - (G + R) / 2.0 + 255.0) / (2 * 255.0)
    return cv2.merge([L, a, b])


def opponent_to_rgb(image: np.ndarray) -> np.ndarray:
    L, a, b = cv2.split(image)
    R = 255 * L + 255 * a - 170 * b - 42.5
    G = 255 * L - 255 * a - 170 * b + 212.5
    B = 255 * L + 340 * b - 170
    return cv2.merge([R, G, B]).astype(np.uint8)


def rgb_to_opponent_codi(image: np.ndarray) -> np.ndarray:
    R, G, B = cv2.split(image.astype(np.float32))
    L = (R + G + B) / (3 * 255.0)
    a = (R - G) / 255.0
    b = (B - (G + R) / 2.0) / 255.0
    return cv2.merge([L, a, b])


def opponent_codi_to_rgb(image: np.ndarray) -> np.ndarray:
    L, a, b = cv2.split(image)
    R = 255 * L + 127.5 * a - 85 * b
    G = 255 * L - 127.5 * a - 85 * b
    B = 255 * L + 170 * b
    return cv2.merge([R, G, B]).astype(np.uint8)


def downsample(
    image: np.ndarray,
    shape: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Downsample an image.

    Wraps opencv.resize so that the "shape" argument is ordered (height, width),
    and to default to linear interpolation.
    The shape is downsampled by a factor of 2.

    Args:
        image: Image array.
        shape: Target shape (height, width).
        interpolation: Interpolation method. Defaults to cv2.INTER_LINEAR.

    Returns:
        Resized image array.
    """
    return cv2.resize(image, shape[::-1], interpolation=interpolation)


def upsample(
    image: np.ndarray,
    shape: tuple[int, int],
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """Upsample an image.

    Wraps opencv.resize so that the "shape" argument is ordered (height, width),
    and to default to cubic interpolation.

    Args:
        image: Image array.
        shape: Target shape (height, width).
        interpolation: Interpolation method. Defaults to cv2.INTER_LINEAR.

    Returns:
        Resized image array.
    """
    return cv2.resize(image, (shape[1], shape[0]), interpolation=interpolation)


def range_normalize(img: np.ndarray) -> np.ndarray:
    """Standard range normalization."""
    min_val, max_val, _, _ = cv2.minMaxLoc(img)
    if max_val != min_val:
        dst = img / (max_val - min_val) + min_val / (min_val - max_val)
    else:
        dst = img - min_val
    return dst
