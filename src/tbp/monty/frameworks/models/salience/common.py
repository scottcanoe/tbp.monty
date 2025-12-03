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


def resize_to(
    array: np.ndarray,
    shape: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize an image using opencv.

    Wraps opencv.resize so that the "shape" argument is ordered (height, width).

    Args:
        array: Image array.
        shape: Target shape (height, width).
        interpolation: Interpolation method. Defaults to cv2.INTER_LINEAR.

    Returns:
        Resized image array.
    """
    return cv2.resize(array, shape[::-1], interpolation=interpolation)


def range_normalize(img: np.ndarray) -> np.ndarray:
    """Standard range normalization
    
    img must be CV_32F?
    """
    min_val, max_val, _, _ = cv2.minMaxLoc(img)
    if max_val != min_val:
        dst = img / (max_val - min_val) + min_val / (min_val - max_val)
    else:
        dst = img - min_val
    return dst
