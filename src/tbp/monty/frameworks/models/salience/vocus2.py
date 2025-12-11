# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Complete Python implementation of VOCUS2 visual attention model.

This is a faithful translation of the C++ implementation from:
"Traditional Saliency Reloaded: A Good Old Model in New Shape"
by S. Frintrop, T. Werner, and G. M. García, CVPR 2015.

Original C++ implementation by Thomas Werner and Simone Frintrop.
Python translation by Thousand Brains Project, 2025.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import cv2
import numpy as np
from skimage.feature import peak_local_max

from .common import downsample, upsample
from .strategies import SalienceStrategy


class ColorSpace(Enum):
    """Color space options for VOCUS2."""

    LAB = 0  # CIE Lab color space
    OPPONENT_CODI = 1  # Opponent color space (like Klein/Frintrop DAGM 2012)
    OPPONENT = 2  # Opponent color space (shifted and scaled to [0,1])


class FusionOperation(Enum):
    """Fusion operations for combining feature maps."""

    ARITHMETIC_MEAN = 0  # Simple average (good for salient object segmentation)
    MAX = 1  # Maximum value
    UNIQUENESS_WEIGHT = 2  # Weight by uniqueness (Frintrop PhD thesis 2005)


class PyramidStructure(Enum):
    """Pyramid structure options."""

    VOCUS2 = 0  # Surround pyramid derived from center pyramid (CVPR 2015)
    CLASSIC = 1  # Two independent pyramids
    CODI = 2  # Two pyramids derived from a base pyramid


@dataclass
class VOCUS2Result:
    rgba: np.ndarray | None = None
    depth: np.ndarray | None = None
    pyramids: dict = field(default_factory=dict)
    feature_maps: dict = field(default_factory=dict)
    conspicuity_maps: dict = field(default_factory=dict)
    salience_map: np.ndarray | None = None
    info: dict = field(default_factory=dict)


class VOCUS2(SalienceStrategy):
    """Complete VOCUS2 implementation matching the original C++ code.

    This implementation includes all features of the original:
    - Multiple pyramid structures (CLASSIC, NEW, CODI)
    - Multiple color spaces (LAB, OPPONENT_CODI, OPPONENT)
    - Multiple fusion operations (MEAN, MAX, UNIQUENESS_WEIGHT)
    - Orientation features
    - On-off and off-on center-surround contrasts
    - Configurable multi-scale pyramids (octaves + scales)
    """

    def __init__(
        self,
        c_space: ColorSpace = ColorSpace.OPPONENT_CODI,
        fuse_feature: FusionOperation = FusionOperation.ARITHMETIC_MEAN,
        fuse_conspicuity: FusionOperation = FusionOperation.ARITHMETIC_MEAN,
        start_layer: int = 0,
        stop_layer: int = 3,
        center_sigma: float = 3.0,
        surround_sigma: float = 13.0,
        n_scales: int = 2,
        normalize: bool = True,
        pyramid_structure: PyramidStructure = PyramidStructure.VOCUS2,
        orientation: bool = True,
        combined_features: bool = True,
        center_bias: float | None = None,
    ):
        """Initialize VOCUS2.

        Args:
            c_space: Color space to use
            fuse_feature: How to fuse feature maps
            fuse_conspicuity: How to fuse conspicuity maps
            start_layer: First pyramid layer (0 = original resolution)
            stop_layer: Last pyramid layer (each layer is half previous size)
            center_sigma: Gaussian sigma for center pyramid
            surround_sigma: Gaussian sigma for surround pyramid
            n_scales: Number of scales per pyramid octave
            normalize: Whether to normalize output to [0,1]
            pyramid_structure: Pyramid structure to use
            orientation: Whether to compute orientation features
            combined_features: Whether to combine color channels before fusion
            center_bias: Strength of center bias (None to disable)
        """
        self._c_space = c_space
        self._fuse_feature = fuse_feature
        self._fuse_conspicuity = fuse_conspicuity
        self._start_layer = start_layer
        self._stop_layer = stop_layer
        self._center_sigma = center_sigma
        self._surround_sigma = surround_sigma
        self._n_scales = n_scales
        self._normalize = normalize
        self._pyramid_structure = pyramid_structure
        self._orientation = orientation
        self._combined_features = combined_features
        self._center_bias = center_bias

        # Orientation features
        self.patches = []
        self.gabor = []  # [orientation][octave*scale]
        self.pyr_laplace = []

    def _clear(self):
        """Clear all internal state."""
        self.patches = []
        self.gabor = []
        self.pyr_laplace = []

    def _prepare_input(self, rgba: np.ndarray) -> list[np.ndarray]:
        """Convert image to desired color space and split into planes.

        Args:
            rgb: RGB image. Should be uint8 in [0, 255].

        Returns:
            List of 3 planes in the target color space (e.g., [L, a, b] for LAB).

        Raises:
            ValueError: If unsupported color space is specified.
        """
        # Get into uint8 0-255 to ensure correct colorspace conversion.
        if rgba.dtype == np.uint8:
            pass
        elif np.issubdtype(rgba.dtype, np.integer):
            rgba = rgba.astype(np.uint8)
        else:
            rgba = (rgba * 255).astype(np.uint8)

        # TODO: Upsample here?

        # Convert colorspace.
        rgb = rgba[:, :, :3]
        if self._c_space == ColorSpace.LAB:
            # Convert to LAB
            converted = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab) / 255.0
            planes = cv2.split(converted)

        elif self._c_space == ColorSpace.OPPONENT_CODI:
            # Opponent color space (CoDi style)
            converted = rgb.astype(np.float32)
            R, G, B = cv2.split(converted)
            L = (R + G + B) / (3 * 255.0)
            a = (R - G) / 255.0
            b = (B - (G + R) / 2.0) / 255.0
            planes = [L, a, b]

        elif self._c_space == ColorSpace.OPPONENT:
            # Opponent color space (shifted to [0,1])
            converted = rgb.astype(np.float32)
            R, G, B = cv2.split(converted)
            L = (R + G + B) / (3 * 255.0)
            a = (R - G + 255.0) / (2 * 255.0)
            b = (B - (G + R) / 2.0 + 255.0) / (2 * 255.0)
            planes = [L, a, b]

        else:
            raise ValueError(f"Unsupported color space: {self._c_space}")

        planes = [p.astype(np.float32) for p in planes]
        return cv2.merge(planes)

    """
    - Pyramids
    ------------------------------------------------------------------------------------
    """

    def _compute_pyramids(
        self, image: np.ndarray
    ) -> dict[str, dict[str, list[list[np.ndarray]]]]:
        """Compute pyramids.

        Args:
            image: Input image in the user's target colorspace.

        Returns:
            Pyramids

        Raises:
            ValueError: If unsupported pyramid structure is specified.
        """
        if self._pyramid_structure == PyramidStructure.VOCUS2:
            return self._pyramid_new(image)
        if self._pyramid_structure == PyramidStructure.CODI:
            return self._pyramid_codi(image)
        if self._pyramid_structure == PyramidStructure.CLASSIC:
            return self._pyramid_classic(image)
        raise ValueError(f"Unsupported pyramid structure: {self._pyramid_structure}")

    def _pyramid_new(self, image: np.ndarray):
        """Build pyramids using VOCUS2 structure (CVPR 2015 default).

        Builds center pyramid first, then derives surround pyramid from it.
        """
        # Build center pyramids
        planes = cv2.split(image)
        center_sigma = self._center_sigma
        surround_sigma = self._surround_sigma
        start_layer = self._start_layer
        stop_layer = self._stop_layer
        n_scales = self._n_scales

        L_center = build_multiscale_pyramid(
            planes[0],
            center_sigma,
            n_scales,
            start_layer,
            stop_layer,
        )
        a_center = build_multiscale_pyramid(
            planes[1],
            center_sigma,
            n_scales,
            start_layer,
            stop_layer,
        )
        b_center = build_multiscale_pyramid(
            planes[2],
            center_sigma,
            n_scales,
            start_layer,
            stop_layer,
        )

        # Build surround pyramids by additional smoothing of center pyramids
        n_octaves = len(L_center)
        L_surround = np.zeros_like(L_center, dtype=object)
        a_surround = np.zeros_like(L_center, dtype=object)
        b_surround = np.zeros_like(L_center, dtype=object)

        adapted_sigma = np.sqrt(surround_sigma**2 - center_sigma**2)
        for octave in range(n_octaves):
            for scale in range(n_scales):
                scaled_sigma = adapted_sigma * (2.0 ** (scale / n_scales))

                L_sur = cv2.GaussianBlur(
                    L_center[octave, scale],
                    (0, 0),
                    scaled_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )
                a_sur = cv2.GaussianBlur(
                    a_center[octave, scale],
                    (0, 0),
                    scaled_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )
                b_sur = cv2.GaussianBlur(
                    b_center[octave, scale],
                    (0, 0),
                    scaled_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )

                L_surround[octave, scale] = L_sur
                a_surround[octave, scale] = a_sur
                b_surround[octave, scale] = b_sur

        return {
            "L": {
                "center": L_center,
                "surround": L_surround,
            },
            "a": {
                "center": a_center,
                "surround": a_surround,
            },
            "b": {
                "center": b_center,
                "surround": b_surround,
            },
        }

    def _pyramid_classic(
        self,
        image: np.ndarray,
    ) -> dict[str, dict[str, list[list[np.ndarray]]]]:
        """Build pyramids using CLASSIC structure.

        Builds center and surround pyramids independently.
        """
        planes = cv2.split(image)
        center_sigma = self._center_sigma
        surround_sigma = self._surround_sigma
        start_layer = self._start_layer
        stop_layer = self._stop_layer
        n_scales = self._n_scales
        # Build center and surround pyramids independently
        pyramids = {}
        for i, feat in enumerate(["L", "a", "b"]):
            pyramids[feat] = {
                "center": build_multiscale_pyramid(
                    planes[i], center_sigma, n_scales, start_layer, stop_layer
                ),
                "surround": build_multiscale_pyramid(
                    planes[i], surround_sigma, n_scales, start_layer, stop_layer
                ),
            }
        return pyramids

    def _pyramid_codi(self, image: np.ndarray):
        """Build pyramids using CODI structure.

        Builds base pyramid first, then derives center and surround from it.
        """
        planes = cv2.split(image)

        # Build base pyramids with sigma=1
        L_base = self._build_multiscale_pyr(planes[0], 1.0)
        a_base = self._build_multiscale_pyr(planes[1], 1.0)
        b_base = self._build_multiscale_pyr(planes[2], 1.0)

        # Compute adapted sigmas
        adapted_center_sigma = np.sqrt(self._center_sigma**2 - 1.0)
        adapted_surround_sigma = np.sqrt(self._surround_sigma**2 - 1.0)

        # Initialize center and surround pyramids
        L_center = []
        a_center = []
        b_center = []
        L_surround = []
        a_surround = []
        b_surround = []

        # For each octave
        for octave in range(len(L_base)):
            L_center.append([])
            a_center.append([])
            b_center.append([])
            L_surround.append([])
            a_surround.append([])
            b_surround.append([])
            # For each scale
            for scale in range(self._n_scales):
                scaled_center_sigma = adapted_center_sigma * (
                    2.0 ** (scale / self._n_scales)
                )
                scaled_surround_sigma = adapted_surround_sigma * (
                    2.0 ** (scale / self._n_scales)
                )

                # Smooth base pyramid to get center and surround
                L_cen = cv2.GaussianBlur(
                    L_base[octave][scale],
                    (0, 0),
                    scaled_center_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )
                L_sur = cv2.GaussianBlur(
                    L_base[octave][scale],
                    (0, 0),
                    scaled_surround_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )

                a_cen = cv2.GaussianBlur(
                    a_base[octave][scale],
                    (0, 0),
                    scaled_center_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )
                a_sur = cv2.GaussianBlur(
                    a_base[octave][scale],
                    (0, 0),
                    scaled_surround_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )

                b_cen = cv2.GaussianBlur(
                    b_base[octave][scale],
                    (0, 0),
                    scaled_center_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )
                b_sur = cv2.GaussianBlur(
                    b_base[octave][scale],
                    (0, 0),
                    scaled_surround_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )

                L_center[octave].append(L_cen)
                a_center[octave].append(a_cen)
                b_center[octave].append(b_cen)
                L_surround[octave].append(L_sur)
                a_surround[octave].append(a_sur)
                b_surround[octave].append(b_sur)

        return {
            "L": {
                "center": L_center,
                "surround": L_surround,
            },
            "a": {
                "center": a_center,
                "surround": a_surround,
            },
            "b": {
                "center": b_center,
                "surround": b_surround,
            },
        }

    """
    - Color (L, a, b)
    ------------------------------------------------------------------------------------
    """

    def _compute_color_feature_maps(
        self, pyramids: dict[str, dict[str, list[list[np.ndarray]]]]
    ):
        """Compute center-surround differences (on-off and off-on)."""

        feature_maps = {}
        for feat in pyramids.keys():
            diffs = pyramids[feat]["center"] - pyramids[feat]["surround"]
            on = np.array(
                [np.clip(arr, 0, np.inf) for arr in diffs.flatten()], dtype=object
            ).reshape(diffs.shape)
            off = np.array(
                [-np.clip(arr, -np.inf, 0) for arr in diffs.flatten()], dtype=object
            ).reshape(diffs.shape)
            feature_maps[feat] = {
                "on": on,
                "off": off,
            }
        return feature_maps

    def _compute_color_conspicuity_maps(
        self,
        feature_maps: dict[str, dict[str, list[np.ndarray]]],
    ) -> dict[str, np.ndarray]:
        """Get conspicuity maps for color features."""
        conspicuity_maps = {}
        L_on = feature_maps["L"]["on"].flat
        L_off = feature_maps["L"]["off"].flat
        a_on = feature_maps["a"]["on"].flat
        a_off = feature_maps["a"]["off"].flat
        b_on = feature_maps["b"]["on"].flat
        b_off = feature_maps["b"]["off"].flat

        # Intensity/luminance
        conspicuity_maps["L"] = self._fuse(
            [
                self._fuse(L_on, self._fuse_feature),
                self._fuse(L_off, self._fuse_feature),
            ],
            self._fuse_conspicuity,
        )

        # Color opponency
        if self._combined_features:
            conspicuity_maps["ab"] = self._fuse(
                [
                    self._fuse(a_on, self._fuse_feature),
                    self._fuse(a_off, self._fuse_feature),
                    self._fuse(b_on, self._fuse_feature),
                    self._fuse(b_off, self._fuse_feature),
                ],
                self._fuse_conspicuity,
            )
        else:
            conspicuity_maps["a"] = self._fuse(
                [
                    self._fuse(a_on, self._fuse_feature),
                    self._fuse(a_off, self._fuse_feature),
                ],
                self._fuse_conspicuity,
            )
            conspicuity_maps["b"] = self._fuse(
                [
                    self._fuse(b_on, self._fuse_feature),
                    self._fuse(b_off, self._fuse_feature),
                ],
                self._fuse_conspicuity,
            )

        return conspicuity_maps

    """
    - Orientation
    ------------------------------------------------------------------------------------
    """

    def _compute_orientation_feature_maps(
        self, pyramids: dict[str, dict[str, list[list[np.ndarray]]]]
    ):
        """Compute orientation features using Gabor filters."""
        image = pyramids["L"]["center"]
        n_octaves = len(image)
        n_scales = self._n_scales

        # Create Gabor kernels
        filter_size = int(11 * self._center_sigma + 1)
        if filter_size % 2 == 0:
            filter_size += 1

        # Build Laplacian pyramid
        pyramid = []
        for octave in range(n_octaves - 1):
            shape_at_octave = image[octave][0].shape
            if shape_at_octave[0] < filter_size or shape_at_octave[1] < filter_size:
                break
            pyramid.append([])
            if octave == n_octaves - 1:
                for scale in range(n_scales):
                    pyramid[octave].append(image[octave][scale])
            else:
                for scale in range(n_scales):
                    src1 = image[octave][scale]
                    src2 = image[octave + 1][scale]
                    # Resize next octave to current size
                    tmp = upsample(src2, src1.shape)
                    laplacian = src1 - tmp
                    pyramid[octave].append(laplacian)

        self.gabor = [[] for _ in range(4)]  # 4 orientations
        self.gabor_patches = []

        feature_maps = {}
        for ori in range(4):
            theta = ori * np.pi / 4
            gabor_kernel = cv2.getGaborKernel(
                (filter_size, filter_size),
                sigma=self._center_sigma * 0.75,
                theta=theta,
                lambd=self._center_sigma * 2,
                gamma=0.75,
                psi=np.pi / 2,
                ktype=cv2.CV_32F,
            )

            # Normalize kernel
            k_sum = np.sum(np.abs(gabor_kernel))
            gabor_kernel /= k_sum

            # Apply Gabor filter to each scale
            res = np.zeros((len(pyramid), n_scales), dtype=object)
            for octave in range(len(pyramid)):
                for scale in range(n_scales):
                    src = pyramid[octave][scale]
                    dst = cv2.filter2D(src, cv2.CV_32F, gabor_kernel)
                    dst = np.abs(dst)
                    res[octave][scale] = dst
            feature_maps[f"orientation_{ori}"] = res
            self.gabor_patches.append(gabor_kernel)

        self.pyr_laplace = np.zeros((len(pyramid), n_scales), dtype=object)
        for octave in range(len(pyramid)):
            for scale in range(n_scales):
                self.pyr_laplace[octave][scale] = pyramid[octave][scale]
        return feature_maps

    def _compute_orientation_conspicuity_maps(
        self,
        feature_maps: dict[str, dict[str, list[np.ndarray]]],
    ) -> dict[str, np.ndarray]:
        """Get conspicuity maps for orientation features."""

        conspicuity_maps = {
            key: self._fuse(val.flat, self._fuse_feature)
            for key, val in feature_maps.items()
        }
        if self._combined_features:
            conspicuity_maps = {
                "orientation": self._fuse(
                    list(conspicuity_maps.values()), self._fuse_conspicuity
                )
            }
        return conspicuity_maps

    """
    - Salience
    ------------------------------------------------------------------------------------
    """

    def _compute_uniqueness_weight(
        self,
        img: np.ndarray,
        threshold_rel: float = 0.5,
    ) -> float:
        """Compute uniqueness weight by counting local maxima.

        Args:
            img: Single channel image
            threshold_rel: Threshold as fraction of maximum value

        Returns:
            Uniqueness weight (1/sqrt(n_maxima))
        """
        assert len(img.shape) == 2, "Input must be single channel"

        # Find maximum
        global_max = np.max(img)

        # Ignore map if global max is too small
        if global_max < 0.05:
            return 0.0

        # Threshold
        threshold_abs = global_max * threshold_rel
        coords = peak_local_max(
            img,
            min_distance=4,
            threshold_abs=threshold_abs,
            exclude_border=False,
        )
        n_maxima = len(coords)
        # This can be zero if the maxima are all on the border. If we want to allow
        # border maxima, we can so peak_local_max(..., exclude_border=False).
        return 0.0 if n_maxima == 0 else 1.0 / np.sqrt(n_maxima)

    def _fuse(self, maps: Sequence[np.ndarray], op: FusionOperation) -> np.ndarray:
        """Fuse multiple maps using specified operation.

        Args:
            maps: List of maps to fuse
            op: Fusion operation to use

        Returns:
            Fused map
        """
        if not len(maps):
            return np.zeros((1, 1), dtype=np.float32)

        target_size = maps[0].shape[:2]
        fused = np.zeros(target_size, dtype=np.float32)

        # # Resize all maps to target size
        # resized = np.zeros((len(maps),) + target_size, dtype=np.float32)
        # for i, m in enumerate(maps):
        #     if m.shape[:2] != target_size:
        #         r = cv2.resize(
        #             m,
        #             (target_size[1], target_size[0]),
        #             interpolation=cv2.INTER_CUBIC,
        #         )
        #         resized[i] = r
        #     else:
        #         resized[i] = m

        if op == FusionOperation.ARITHMETIC_MEAN:
            # Simple average
            fused = fuse_mean(maps)

        elif op == FusionOperation.MAX:
            # Maximum value
            fused = fuse_max(maps)

        elif op == FusionOperation.UNIQUENESS_WEIGHT:
            # Weight by uniqueness
            weights = []
            weighted_maps = []

            for m in maps:
                w = self._compute_uniqueness_weight(m)
                weights.append(w)
                if w > 0:
                    # Resize weighted map
                    weighted = m * w
                    if weighted.shape[:2] != target_size:
                        weighted = cv2.resize(
                            weighted,
                            (target_size[1], target_size[0]),
                            interpolation=cv2.INTER_CUBIC,
                        )
                    weighted_maps.append(weighted)

            sum_weights = sum(weights)
            if sum_weights > 0:
                for wm in weighted_maps:
                    fused += wm
                fused /= sum_weights

        return fused

    def _compute_salience_map(
        self,
        conspicuity_maps: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Get final saliency map.

        Returns:
            Saliency map normalized to [0,1]

        """
        # Final saliency map
        salience_map = self._fuse(
            list(conspicuity_maps.values()), self._fuse_conspicuity
        )

        # Normalize to [0,1]
        if self._normalize:
            min_val = np.min(salience_map)
            max_val = np.max(salience_map)
            if max_val > min_val:
                salience_map = (salience_map - min_val) / (max_val - min_val)

        return salience_map

    def _add_center_bias(
        self,
        salience_map: np.ndarray,
        center_bias: float,
    ) -> np.ndarray:
        """Add center bias to saliency map.

        Args:
            lambda_param: Strength of center bias

        Returns:
            Center-biased saliency map
        """
        salience_map = salience_map.copy()

        # Apply Gaussian weighting
        h, w = salience_map.shape
        cr, cc = h // 2, w // 2
        for r in range(h):
            for c in range(w):
                d = np.sqrt((r - cr) ** 2 + (c - cc) ** 2)
                fak = np.exp(-center_bias * d * d)
                salience_map[r, c] *= fak

        # Normalize
        if self._normalize:
            min_val = np.min(salience_map)
            max_val = np.max(salience_map)
            if max_val > min_val:
                salience_map = (salience_map - min_val) / (max_val - min_val)

        return salience_map

    def call(self, rgba: np.ndarray, depth: np.ndarray | None = None) -> VOCUS2Result:
        """Compute VOCUS2 saliency map (SalienceStrategy interface).

        Args:
            rgba: RGBA image (uint8 or float32)
            depth: Depth image (optional, not used)

        Returns:
            Saliency map in [0, 1] range
        """
        result = VOCUS2Result(rgba=rgba, depth=depth)

        # Convert to uint8 0-255 if needed.
        input_shape = rgba.shape[:2]

        image = self._prepare_input(rgba)  # e.g. [L, a, b] for LAB
        pyramids = self._compute_pyramids(image)

        color_feature_maps = self._compute_color_feature_maps(pyramids)
        color_conspicuity_maps = self._compute_color_conspicuity_maps(
            color_feature_maps
        )

        if self._orientation:
            orientation_feature_maps = self._compute_orientation_feature_maps(pyramids)
            orientation_conspicuity_maps = self._compute_orientation_conspicuity_maps(
                orientation_feature_maps
            )
        else:
            orientation_feature_maps = {}
            orientation_conspicuity_maps = {}

        feature_maps = {}
        feature_maps.update(color_feature_maps)
        feature_maps.update(orientation_feature_maps)
        conspicuity_maps = {}
        conspicuity_maps.update(color_conspicuity_maps)
        conspicuity_maps.update(orientation_conspicuity_maps)

        salience_map = self._compute_salience_map(conspicuity_maps)

        if self._center_bias:
            salience_map = self._add_center_bias(salience_map, self._center_bias)

        if salience_map.shape[:2] != input_shape:
            salience_map = cv2.resize(
                salience_map,
                (input_shape[1], input_shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        result.image = image
        result.pyramids = pyramids
        result.feature_maps = feature_maps
        result.conspicuity_maps = conspicuity_maps
        result.salience_map = salience_map

        result.gabor_patches = self.gabor_patches
        result.pyramids["orientation"] = self.pyr_laplace

        return result

    def __call__(
        self, rgba: np.ndarray, depth: np.ndarray | None = None
    ) -> VOCUS2Result:
        """Compute VOCUS2 saliency map (SalienceStrategy interface)."""
        return self.call(rgba, depth).salience_map


def merge_octaves(maps: Sequence[np.ndarray]) -> list[np.ndarray]:
    target_size = maps[0].shape[:2]
    merged = np.zeros((len(maps),) + target_size, dtype=np.float32)
    for i, m in enumerate(maps):
        if m.shape[:2] != target_size:
            merged[i] = upsample(m, target_size)
        else:
            merged[i] = m

    return merged


def fuse_mean(maps: Sequence[np.ndarray]) -> np.ndarray:
    return merge_octaves(maps).mean(axis=0)


def fuse_max(maps: Sequence[np.ndarray]) -> np.ndarray:
    return merge_octaves(maps).max(axis=0)


def build_multiscale_pyramid(
    image: np.ndarray,
    sigma: float,
    n_scales: int,
    start_layer: int,
    stop_layer: int,
) -> np.ndarray:
    """Build multi-scale pyramid following Lowe 2004.

    This creates a 2D pyramid structure:
    - Dimension 1 (octaves): Different resolutions (each half the previous)
    - Dimension 2 (scales): Different smoothing levels within each octave

    Args:
        src: Input image (single channel, float32)
        sigma: Base sigma for Gaussian smoothing

    Returns:
        2D object-type array with shape (n_octaves, n_scales)
    """
    # Calculate maximum number of octaves
    min_dim = min(image.shape[0], image.shape[1])
    max_octaves = min(int(np.log2(min_dim)), stop_layer) + 1
    n_octaves = max_octaves - start_layer

    # Quickly resize dummy data until we get to the first octave we want to use.
    src = image
    for _ in range(start_layer):
        src = cv2.GaussianBlur(
            src, (0, 0), 2.0 * sigma, borderType=cv2.BORDER_REPLICATE
        )
        src = cv2.resize(src, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

    # Compute pyramid as in Lowe 2004
    sig_prev = sig_total = 0.0
    pyr = np.zeros((n_octaves, n_scales + 1), dtype=object)
    for octave in range(n_octaves):
        # Compute n_scales + 1 (extra scale used as first of next octave)
        for scale in range(n_scales + 1):
            # First scale of first octave: smooth tmp
            if octave == 0 and scale == 0:
                sig_total = (2.0 ** (scale / n_scales)) * sigma
                dst = cv2.GaussianBlur(
                    src, (0, 0), sig_total, borderType=cv2.BORDER_REPLICATE
                )
                sig_prev = sig_total

            # First scale of other octaves: subsample additional scale of previous
            elif octave != 0 and scale == 0:
                src = pyr[octave - 1][n_scales]
                dst = cv2.resize(
                    src,
                    (src.shape[1] // 2, src.shape[0] // 2),
                    interpolation=cv2.INTER_NEAREST,
                )
                sig_prev = sigma

            # Intermediate scales: smooth previous scale
            else:
                sig_total = (2.0 ** (scale / n_scales)) * sigma
                sig_diff = np.sqrt(sig_total**2 - sig_prev**2)
                src = pyr[octave][scale - 1]
                dst = cv2.GaussianBlur(
                    src, (0, 0), sig_diff, borderType=cv2.BORDER_REPLICATE
                )
                sig_prev = sig_total

            pyr[octave, scale] = dst

    # Erase the temporary scale in each octave that was just used to
    # compute the sigmas for the next octave.
    return pyr[:, :-1]
