# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import timeit
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tbp.monty.frameworks.models.salience.vocus2 import ColorSpace

"""
- image utilities
-------------------------------------------------------------------------------
"""



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


def gaussian_blur(image: np.ndarray, sigma: float, truncate: float = 2.5) -> np.ndarray:
    ksize = int(round(2 * truncate * sigma + 1)) | 1  # Ensure odd
    return cv2.GaussianBlur(
        image, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE
    )


def resize(
    image: np.ndarray,
    shape: tuple[int, int],
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    return cv2.resize(image, (shape[1], shape[0]), interpolation=interpolation)



"""
- Gaussian Pyramid
-------------------------------------------------------------------------------
"""
@dataclass(frozen=True)
class Pyramid:
    
    data: np.ndarray
    
    def __post_init__(self):
        assert isinstance(self.data, np.ndarray)
        assert np.issubdtype(self.data.dtype, np.object_)
        assert self.data.ndim in (1, 2)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def flat(self) -> Iterator[np.ndarray]:
        return self.data.flat

    def apply(self, func: Callable) -> "Pyramid":
        data = np.zeros(self.data.size, dtype=object)
        for i, arr in enumerate(self.data.flat):
            data[i] = func(arr)
        return Pyramid(data.reshape(self.data.shape))

    def __add__(self, other: "Pyramid") -> "Pyramid":
        return Pyramid(self.data + other.data)
    
    def __sub__(self, other: "Pyramid") -> "Pyramid":
        return Pyramid(self.data - other.data)
    
    def __repr__(self) -> str:
        return f"Pyramid(shape={self.shape}, size={self.size}, ndim={self.ndim})"

    def __str__(self) -> str:
        return f"Pyramid(shape={self.shape}, size={self.size}, ndim={self.ndim})"

    def __len__(self) -> int:
        return len(self.data)


def pyramid_level_shapes(
    image_shape: tuple[int, int],
    max_levels: int | None = None,
    min_size: int | None = None,
) -> list[tuple[int, int]]:
    """Compute the shapes of the pyramid levels.

    Args:
        base_shape: The shape of the base level.
        max_levels: The maximum number of levels in the pyramid.
        min_size: The minimum size of the pyramid levels.
    Returns:
        A list of tuples, each containing the shape of a pyramid level.
    """

    max_possible_octaves = int(np.log2(min(image_shape))) + 1
    if max_levels:
        max_levels = min(max_levels, max_possible_octaves)
    else:
        max_levels = max_possible_octaves

    min_size = min_size or 1

    cur_shape = image_shape
    shapes = []
    while len(shapes) < max_levels and min(cur_shape) >= min_size:
        shapes.append(cur_shape)
        cur_shape = (cur_shape[0] // 2, cur_shape[1] // 2)

    return shapes



class PyramidCombine(Protocol):
    def __call__(self, pyramids: Sequence[Pyramid]) -> Pyramid: ...

class PyramidCollapse(Protocol):
    def __call__(self, pyr: Pyramid) -> np.ndarray: ...


def pyramid_combine(
    pyramids: Sequence[Pyramid],
    fn: Callable[[Sequence[np.ndarray]], np.ndarray],
    ) -> np.ndarray:
    n_pyramids = len(pyramids)
    if n_pyramids == 0:
        raise ValueError("No pyramids to combine")
    elif n_pyramids == 1:
        return pyramids[0]

    pyr_arrays = [pyr.data for pyr in pyramids]
    pyr_shape = pyr_arrays[0].shape
    assert all(pyr.shape == pyr_shape for pyr in pyr_arrays[1:])
    pyr_size = pyr_arrays[0].size
    new_data = np.zeros(pyr_size, dtype=object)
    for i, images in enumerate(zip(*[pyr.flat for pyr in pyramids])):
        target_shape = images[0].shape
        resized = [resize(img, target_shape) if img.shape != target_shape else img for img in images]
        new_data[i] = fn(resized)
    new_data = new_data.reshape(pyr_shape)
    return Pyramid(new_data)


def pyramid_combine_max(pyramids: Sequence[Pyramid]) -> Pyramid:
    return pyramid_combine(pyramids, lambda x: np.max(x, axis=0))

def pyramid_combine_mean(pyramids: Sequence[Pyramid]) -> Pyramid:
    return pyramid_combine(pyramids, lambda x: np.mean(x, axis=0))


def pyramid_collapse(
    pyr: Pyramid,
    fn: Callable[[Sequence[np.ndarray]], np.ndarray],
) -> np.ndarray:
    images = list(pyr.data.flat)
    target_shape = images[0].shape
    resized = [resize(img, target_shape, interpolation=cv2.INTER_CUBIC) if img.shape != target_shape else img for img in images]
    return fn(resized)

def pyramid_collapse_max(pyr: Pyramid) -> np.ndarray:
    return pyramid_collapse(pyr, lambda x: np.max(x, axis=0))

def pyramid_collapse_mean(pyr: Pyramid) -> np.ndarray:
    return pyramid_collapse(pyr, lambda x: np.mean(x, axis=0))



"""
- Pyramid Building
------------------------------------------------------------------------------
"""


def gaussian_pyramid(
    image: np.ndarray,
    sigma: float,
    n_scales: int,
    max_levels: int | None = None,
    min_size: int | None = None,
) -> Pyramid:
    """Build multi-scale pyramid following Lowe 2004.

    This creates a 2D pyramid structure:
    - Dimension 1 (octaves): Different resolutions (each half the previous)
    - Dimension 2 (scales): Different smoothing levels within each octave

    Args:
        src: Input image (single channel, float32)
        sigma: Base sigma for Gaussian smoothing

    Returns:
        2D object-type array with shape (n_octaves, n_scales)

    Note: sigmas = [sigma * (2.0 ** (s / n_scales)) for s in range(pyr.size)]
      
    """
    # Calculate maximum number of octaves
    shapes = pyramid_level_shapes(
        image.shape, max_levels=max_levels, min_size=min_size
    )

    # Compute pyramid as in Lowe 2004
    pyr = np.zeros((len(shapes), n_scales + 1), dtype=object)
    for octave in range(len(shapes)):
        # Compute n_scales + 1 (extra scale used as first of next octave)
        for scale in range(n_scales + 1):
            # First scale of first octave: smooth tmp
            if octave == 0 and scale == 0:
                src = image
                dst = gaussian_blur(src, sigma)
                
            # First scale of other octaves: subsample additional scale of previous
            elif octave > 0 and scale == 0:
                src = pyr[octave - 1, n_scales]
                dst = resize(src, shapes[octave])

            # Intermediate scales: smooth previous scale
            else:
                target_sigma = sigma * 2.0 ** (scale / n_scales)
                previous_sigma = sigma * 2.0 ** ((scale - 1)/ n_scales)
                sig_diff = np.sqrt(target_sigma**2 - previous_sigma**2)
                src = pyr[octave, scale - 1]
                dst = gaussian_blur(src, sig_diff)

            pyr[octave, scale] = dst

    # Erase the temporary scale in each octave that was just used to
    # compute the sigmas for the next octave.
    pyr = pyr[:, :-1]
    return Pyramid(pyr)



def center_surround_pyramids(
    image: np.ndarray,
    center_sigma: float,
    surround_sigma: float,
    n_scales: int,
    **kwargs,
    ) -> tuple[Pyramid, Pyramid]:
        
    center = gaussian_pyramid(image, sigma=center_sigma, n_scales=n_scales, **kwargs)
    
    center = center if isinstance(center, Pyramid) else Pyramid(center)
    n_octaves, n_scales = center.data.shape

    # Use adapted surround sigma, a la VOCUS2.
    adapted_sigma = np.sqrt(surround_sigma**2 - center_sigma**2)
    surround = np.zeros((n_octaves, n_scales), dtype=object)
    for level in range(n_octaves):
        for scale in range(n_scales):
            scaled_sigma = adapted_sigma * (2.0 ** (scale / n_scales))
            center_img = center.data[level, scale]
            surround[level, scale] = gaussian_blur(center_img, scaled_sigma)
    
    surround: Pyramid = Pyramid(surround)

    return center, surround


"""
- Laplacian Pyramids
-------------------------------------------------------------------------------
"""


def laplacian_pyramid(
    pyr: Pyramid,
    max_levels: int | None = None,
    min_size: int | None = None,
):
    """Build a multiscale Laplacian pyramid."""

    gauss = pyr.data if isinstance(pyr, Pyramid) else pyr
    n_levels_in = gauss.shape[0]
    n_levels_out = n_levels_in - 1
    if max_levels is not None:
        n_levels_out = min(n_levels_out, max_levels)
    if min_size is not None:
        level_sizes = np.array([min(arrays[0].shape) for arrays in gauss])
        n_levels_big_enough = sum(level_sizes >= min_size)
        n_levels_out = min(n_levels_out, n_levels_big_enough)

    lap = np.zeros([n_levels_out, gauss.shape[1]], dtype=object)
    for scale in range(lap.shape[1]):
        for octave in range(lap.shape[0]):
            center = gauss[octave, scale]
            surround = resize(
                gauss[octave + 1, scale], center.shape, interpolation=cv2.INTER_CUBIC
            )
            lap[octave, scale] = center - surround

    return Pyramid(lap)



"""
- Combining/Collapsing Feature Maps and Image Pyramids
"""



class MapCombine(Protocol):
    def __call__(self, maps: dict[str, np.ndarray]) -> np.ndarray: ...


def map_max(maps: dict[int | str, np.ndarray]) -> np.ndarray:
    np.max(list(maps.values()), axis=0)

def map_sum(maps: dict[int | str, np.ndarray]) -> np.ndarray:
    return np.sum(list(maps.values()), axis=0)

def map_weighted_sum(
    maps: dict[int | str, np.ndarray],
    weights: dict[int | str, float],
) -> np.ndarray:
    return np.sum([weights[key] * img for key, img in maps.items()], axis=0)

def map_mean(maps: dict[int | str, np.ndarray]) -> np.ndarray:
    return np.mean(list(maps.values()), axis=0)

def _map_weighted_mean(
    maps: dict[int | str, np.ndarray],
    weights: dict[int | str, float],
) -> np.ndarray:
    total_weight = sum([abs(weights[key]) for key in maps.keys()])
    normed_weights = {key: weights[key] / total_weight for key in maps.keys()}
    return np.sum([normed_weights[key] * img for key, img in maps.items()], axis=0)


class WeightedMean(MapCombine):
    def __init__(self, weights: dict[int | str, float]):
        self._weights = dict(weights)
    def __call__(self, maps: dict[int | str, np.ndarray]) -> np.ndarray:
        return _map_weighted_mean(maps, self._weights)




@dataclass
class SalienceResult:
    salience_map: np.ndarray | None = None
    components: dict[str, Any] = field(default_factory=dict)


class OnOffSalience:
    def __init__(
        self,
        center_sigma: float,
        surround_sigma: float,
        n_scales: int,
        max_levels: int | None = None,
        min_size: int | None = None,
        combine: PyramidCombine = pyramid_combine_mean,
        collapse: PyramidCollapse = pyramid_collapse_mean,
    ):
    
        self._center_sigma = center_sigma
        self._surround_sigma = surround_sigma
        self._n_scales = n_scales
        self._max_levels = max_levels
        self._min_size = min_size
        self._combine = combine
        self._collapse = collapse

    def _prepare_input(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.float32:
            return image
        if np.issubdtype(image.dtype, np.integer):
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)

    def __call__(self, image: np.ndarray) -> SalienceResult:
        
        # Make sure float32.
        image = self._prepare_input(image)

        # Build center/surround and on/off pyramids.
        center, surround = center_surround_pyramids(
            image,
            center_sigma=self._center_sigma,
            surround_sigma=self._surround_sigma,
            n_scales=self._n_scales,
            max_levels=self._max_levels,
            min_size=self._min_size,
        )

        diff: Pyramid = center - surround
        on: Pyramid = diff.apply(lambda img: np.maximum(img, 0))
        off: Pyramid = diff.apply(lambda img: np.maximum(-img, 0))

        # Combine on/off pyramids to get feature map (a pyramid)
        salience_pyramid = self._combine([on, off])
        salience_map = self._collapse(salience_pyramid)

        return SalienceResult(
            salience_map=salience_map,
            components={
                "center": center,
                "surround": surround,
                "on": on,
                "off": off,
                "salience": salience_pyramid,
            }
        )


class DepthSalience:
    def __init__(
        self,
        center_sigma: float,
        surround_sigma: float,
        n_scales: int,
        max_levels: int | None = None,
        min_size: int | None = None,
        combine: PyramidCombine = pyramid_combine_mean,
        collapse: PyramidCollapse = pyramid_collapse_mean,
    ):

        self._center_sigma = center_sigma
        self._surround_sigma = surround_sigma
        self._n_scales = n_scales
        self._max_levels = max_levels
        self._min_size = min_size
        self._combine = combine
        self._collapse = collapse

    def _prepare_input(self, image: np.ndarray) -> np.ndarray:
        image = -np.log(image)
        if image.dtype == np.float32:
            return image
        if np.issubdtype(image.dtype, np.integer):
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)

    def __call__(self, image: np.ndarray) -> SalienceResult:

        # Make sure float32.
        image = self._prepare_input(image)

        # Build center/surround and on/off pyramids.
        center, surround = center_surround_pyramids(
            image,
            center_sigma=self._center_sigma,
            surround_sigma=self._surround_sigma,
            n_scales=self._n_scales,
            max_levels=self._max_levels,
            min_size=self._min_size,
        )

        diff: Pyramid = center - surround
        on: Pyramid = diff.apply(lambda img: np.maximum(img, 0))
        # off: Pyramid = diff.apply(lambda img: np.maximum(-img, 0))

        # Combine on/off pyramids to get feature map (a pyramid)
        # salience_pyramid = self._combine([on, off])
        salience_pyramid = on
        salience_map = self._collapse(salience_pyramid)

        return SalienceResult(
            salience_map=salience_map,
            components={
                "center": center,
                "surround": surround,
                "on": on,
                # "off": off,
                "salience": salience_pyramid,
            }
        )


class OrientationSalience:
    def __init__(
        self,
        period: float,
        sigma: float | None = None,
        phase: float = np.pi / 2,
        gamma: float = 0.75,
        n_orientations: int = 4,
        combine: MapCombine = map_mean,
        collapse: PyramidCollapse = pyramid_collapse_mean
    ):
        """Builds orientation salience model.

        Args:
            period: wavelength. Good default is center_sigma * 2
            sigma: mask sigma. Good default is 0.3 * period
            phase: phase. Good default is 90 degrees (pi / 2) for edge detection,
                0 for strip detection.
            gamma: Eccentricity. Good default is 0.75
            n_orientations: number of orientations. Good default is 4

        """
        self._period = period
        self._sigma = 0.3 * self._period if sigma is None else sigma
        self._phase = phase
        self._gamma = gamma
        self._n_orientations = n_orientations
        self._combine = combine
        self._collapse = collapse

        self._kernels = self.make_kernels(
            period=self._period,
            sigma=self._sigma,
            phase=self._phase,
            gamma=self._gamma,
            n_orientations=self._n_orientations,
        )

    def __call__(self, pyr: Pyramid) -> SalienceResult:

        pyramids = {}
        feature_maps = {}
        lap = laplacian_pyramid(pyr)
        for ori, kernel in self._kernels.items():
            pyramids[ori] = np.zeros(lap.shape, dtype=object)
            p = np.zeros(lap.shape, dtype=object)
            for level in range(lap.shape[0]):
                for scale in range(lap.shape[1]):
                    amt = cv2.filter2D(
                        lap.data[level, scale], cv2.CV_32F, kernel
                    )
                    p[level, scale] = np.abs(amt)
            pyramids[ori] = Pyramid(p)
            if self._collapse:
                feature_maps[ori] = self._collapse(pyramids[ori])
        if self._combine:
            salience_map = self._combine(feature_maps)
        else:
            salience_map = None
        return SalienceResult(
            salience_map=salience_map,
            components={
                "feature_maps": feature_maps,
                "pyramids": pyramids,
            },
        )

    @staticmethod
    def make_kernels(
        period: float,
        sigma: float,
        phase: float = np.pi / 2,
        gamma: float = 0.75,
        n_orientations: int = 4,
        ) -> dict[str, np.ndarray]:

        kernels = {}
        filter_size = int(7 * sigma + 1) | 1
        for ori in range(n_orientations):
            theta = ori * np.pi / n_orientations
            kernel = cv2.getGaborKernel(
                (filter_size, filter_size),
                sigma=sigma,
                theta=theta,
                lambd=period,
                gamma=gamma,
                psi=phase,
                ktype=cv2.CV_32F,
            )
            # kernel = kernel / np.sum(np.abs(kernel))
            kernel = kernel - np.mean(kernel)  # balance excitation and suppression
            kernels[f"orientation_{ori}"] = kernel

        return kernels


class Sal9000:

    def __init__(
        self,
        color_space: ColorSpace = ColorSpace.OPPONENT,
        color: OnOffSalience | None = None,
        depth: DepthSalience | None = None,
        orientation: OrientationSalience | None = None,
        combine: MapCombine | None = None,
    ):

        self._color_space = color_space
        if color is None:
            self._color = OnOffSalience(
                center_sigma=3.0,
                surround_sigma=5.0,
                n_scales=2,
                max_levels=5,
                min_size=16,
            )
        else:
            self._color = color

        if depth is None:
            # self._depth = DepthSalience(
            # center_sigma=3.0,
            # surround_sigma=5.0,
            # n_scales=2,
            # max_levels=5,
            # min_size=32,
            # )
            self._depth = None
        else:
            self._depth = depth

        if orientation is None:
            pass
            # self._orientation = OrientationSalience(
            #     period=2 * 3.0,
            # )
            self._orientation = None
        else:
            self._orientation = orientation

        if combine is None:
            self._combine = WeightedMean(
                weights={
                    "L": 1,
                    "a": 1,
                    "b": 1,
                    # "depth": 2,
                    # "orientation": 1,
                }
            )
        else:
            self._combine = combine

    def __call__(self, rgba: np.ndarray, depth: np.ndarray | None = None) -> np.ndarray:
        return self.process(rgba, depth).salience_map

    def process(self, rgba: np.ndarray, depth: np.ndarray | None = None) -> SalienceResult:
        Lab, depth = self._prepare_input(rgba, depth)

        components: dict[str, Any] = {}
        feature_maps: dict[str, np.ndarray] = {}

        if self._color:
            L, a, b = cv2.split(Lab)
            for channel, plane in zip(("L", "a", "b"), (L, a, b)):
                channel_result = self._color(plane)
                components[channel] = channel_result
                feature_maps[channel] = channel_result.salience_map

        if self._depth and depth is not None:
            depth_result = self._depth(depth)
            components["depth"] = depth_result
            feature_maps["depth"] = depth_result.salience_map

        if self._orientation:
            orientation_result = self._orientation(components["L"].components["center"])
            components["orientation"] = orientation_result
            feature_maps["orientation"] = orientation_result.salience_map

        components["feature_maps"] = feature_maps

        salience_map = self._combine(feature_maps)

        # Normalize salience_map to [0, 1]
        salience_min = salience_map.min()
        salience_max = salience_map.max()
        scale = salience_max - salience_min
        if np.isclose(scale, 0):
            salience_map = np.clip(salience_map, 0, 1)
        else:
            salience_map = (salience_map - salience_min) / scale
        return SalienceResult(
            salience_map=salience_map,
            components=components,
        )

    def _prepare_input(
        self,
        rgba: np.ndarray,
        depth: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:

        # Get into uint8 0-255 to ensure correct colorspace conversion.
        if rgba.dtype == np.uint8:
            pass
        elif np.issubdtype(rgba.dtype, np.integer):
            rgba = rgba.astype(np.uint8)
        else:
            rgba = (rgba * 255).astype(np.uint8)
        rgb = rgba[:, :, :3]

        # Put into desired color space.
        if self._color_space == ColorSpace.LAB:
            Lab = rgb_to_lab(rgb)
        elif self._color_space == ColorSpace.OPPONENT:
            Lab = rgb_to_opponent(rgb)
        elif self._color_space == ColorSpace.OPPONENT_CODI:
            Lab = rgb_to_opponent_codi(rgb)
        else:
            raise ValueError(f"Unsupported color space: {self._color_space}")

        if depth is not None:
            depth = depth.astype(np.float32)

        return Lab, depth



