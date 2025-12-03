# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .common import resize_to
from .strategies import SalienceStrategy


@dataclass
class IttiKochResult:
    rgba: np.ndarray | None = None
    depth: np.ndarray | None = None
    feature_maps: dict = field(default_factory=dict)
    conspicuity_maps: dict = field(default_factory=dict)
    salience_map: np.ndarray | None = None
    info: dict = field(default_factory=dict)


class IttiKoch(SalienceStrategy):
    """Implementation of the Itti-Koch visual attention model.

    Based on "A model of saliency-based visual attention for rapid scene analysis"
    by L. Itti, C. Koch, and E. Niebur, IEEE TPAMI 1998.

    This model computes saliency using intensity, color, and orientation conspicuity maps
    through center-surround operations on multi-scale image pyramids.
    """

    def __init__(
        self,
        upsample_size: tuple[int, int] | None = None,
        weight_intensity: float = 0.33,
        weight_color: float = 0.33,
        weight_orientation=0.33,
        pyramid_levels: int = 9,
    ):
        self.upsample_size = upsample_size
        self.weight_intensity = weight_intensity
        self.weight_color = weight_color
        self.weight_orientation = weight_orientation
        self.pyramid_levels = pyramid_levels

        self.GaborKernel0 = np.array(
            [
                [
                    1.85212e-06,
                    1.28181e-05,
                    -0.000350433,
                    -0.000136537,
                    0.002010422,
                    -0.000136537,
                    -0.000350433,
                    1.28181e-05,
                    1.85212e-06,
                ],
                [
                    2.80209e-05,
                    0.000193926,
                    -0.005301717,
                    -0.002065674,
                    0.030415784,
                    -0.002065674,
                    -0.005301717,
                    0.000193926,
                    2.80209e-05,
                ],
                [
                    0.000195076,
                    0.001350077,
                    -0.036909595,
                    -0.014380852,
                    0.211749204,
                    -0.014380852,
                    -0.036909595,
                    0.001350077,
                    0.000195076,
                ],
                [
                    0.000624940,
                    0.004325061,
                    -0.118242318,
                    -0.046070008,
                    0.678352526,
                    -0.046070008,
                    -0.118242318,
                    0.004325061,
                    0.000624940,
                ],
                [
                    0.000921261,
                    0.006375831,
                    -0.174308068,
                    -0.067914552,
                    1.000000000,
                    -0.067914552,
                    -0.174308068,
                    0.006375831,
                    0.000921261,
                ],
                [
                    0.000624940,
                    0.004325061,
                    -0.118242318,
                    -0.046070008,
                    0.678352526,
                    -0.046070008,
                    -0.118242318,
                    0.004325061,
                    0.000624940,
                ],
                [
                    0.000195076,
                    0.001350077,
                    -0.036909595,
                    -0.014380852,
                    0.211749204,
                    -0.014380852,
                    -0.036909595,
                    0.001350077,
                    0.000195076,
                ],
                [
                    2.80209e-05,
                    0.000193926,
                    -0.005301717,
                    -0.002065674,
                    0.030415784,
                    -0.002065674,
                    -0.005301717,
                    0.000193926,
                    2.80209e-05,
                ],
                [
                    1.85212e-06,
                    1.28181e-05,
                    -0.000350433,
                    -0.000136537,
                    0.002010422,
                    -0.000136537,
                    -0.000350433,
                    1.28181e-05,
                    1.85212e-06,
                ],
            ],
            dtype=np.float32,
        )

        self.GaborKernel45 = np.array(
            [
                [
                    4.04180e-06,
                    2.25320e-05,
                    -0.000279806,
                    -0.001028923,
                    3.79931e-05,
                    0.000744712,
                    0.000132863,
                    -9.04408e-06,
                    -1.01551e-06,
                ],
                [
                    2.25320e-05,
                    0.000925120,
                    0.002373205,
                    -0.013561362,
                    -0.022947700,
                    0.000389916,
                    0.003516954,
                    0.000288732,
                    -9.04408e-06,
                ],
                [
                    -0.000279806,
                    0.002373205,
                    0.044837725,
                    0.052928748,
                    -0.139178011,
                    -0.108372072,
                    0.000847346,
                    0.003516954,
                    0.000132863,
                ],
                [
                    -0.001028923,
                    -0.013561362,
                    0.052928748,
                    0.460162150,
                    0.249959607,
                    -0.302454279,
                    -0.108372072,
                    0.000389916,
                    0.000744712,
                ],
                [
                    3.79931e-05,
                    -0.022947700,
                    -0.139178011,
                    0.249959607,
                    1.000000000,
                    0.249959607,
                    -0.139178011,
                    -0.022947700,
                    3.79931e-05,
                ],
                [
                    0.000744712,
                    0.003899160,
                    -0.108372072,
                    -0.302454279,
                    0.249959607,
                    0.460162150,
                    0.052928748,
                    -0.013561362,
                    -0.001028923,
                ],
                [
                    0.000132863,
                    0.003516954,
                    0.000847346,
                    -0.108372072,
                    -0.139178011,
                    0.052928748,
                    0.044837725,
                    0.002373205,
                    -0.000279806,
                ],
                [
                    -9.04408e-06,
                    0.000288732,
                    0.003516954,
                    0.000389916,
                    -0.022947700,
                    -0.013561362,
                    0.002373205,
                    0.000925120,
                    2.25320e-05,
                ],
                [
                    -1.01551e-06,
                    -9.04408e-06,
                    0.000132863,
                    0.000744712,
                    3.79931e-05,
                    -0.001028923,
                    -0.000279806,
                    2.25320e-05,
                    4.04180e-06,
                ],
            ],
            dtype=np.float32,
        )

        self.GaborKernel90 = np.array(
            [
                [
                    1.85212e-06,
                    2.80209e-05,
                    0.000195076,
                    0.000624940,
                    0.000921261,
                    0.000624940,
                    0.000195076,
                    2.80209e-05,
                    1.85212e-06,
                ],
                [
                    1.28181e-05,
                    0.000193926,
                    0.001350077,
                    0.004325061,
                    0.006375831,
                    0.004325061,
                    0.001350077,
                    0.000193926,
                    1.28181e-05,
                ],
                [
                    -0.000350433,
                    -0.005301717,
                    -0.036909595,
                    -0.118242318,
                    -0.174308068,
                    -0.118242318,
                    -0.036909595,
                    -0.005301717,
                    -0.000350433,
                ],
                [
                    -0.000136537,
                    -0.002065674,
                    -0.014380852,
                    -0.046070008,
                    -0.067914552,
                    -0.046070008,
                    -0.014380852,
                    -0.002065674,
                    -0.000136537,
                ],
                [
                    0.002010422,
                    0.030415784,
                    0.211749204,
                    0.678352526,
                    1.000000000,
                    0.678352526,
                    0.211749204,
                    0.030415784,
                    0.002010422,
                ],
                [
                    -0.000136537,
                    -0.002065674,
                    -0.014380852,
                    -0.046070008,
                    -0.067914552,
                    -0.046070008,
                    -0.014380852,
                    -0.002065674,
                    -0.000136537,
                ],
                [
                    -0.000350433,
                    -0.005301717,
                    -0.036909595,
                    -0.118242318,
                    -0.174308068,
                    -0.118242318,
                    -0.036909595,
                    -0.005301717,
                    -0.000350433,
                ],
                [
                    1.28181e-05,
                    0.000193926,
                    0.001350077,
                    0.004325061,
                    0.006375831,
                    0.004325061,
                    0.001350077,
                    0.000193926,
                    1.28181e-05,
                ],
                [
                    1.85212e-06,
                    2.80209e-05,
                    0.000195076,
                    0.000624940,
                    0.000921261,
                    0.000624940,
                    0.000195076,
                    2.80209e-05,
                    1.85212e-06,
                ],
            ],
            dtype=np.float32,
        )

        self.GaborKernel135 = np.array(
            [
                [
                    -1.01551e-06,
                    -9.04408e-06,
                    0.000132863,
                    0.000744712,
                    3.79931e-05,
                    -0.001028923,
                    -0.000279806,
                    2.2532e-05,
                    4.0418e-06,
                ],
                [
                    -9.04408e-06,
                    0.000288732,
                    0.003516954,
                    0.000389916,
                    -0.022947700,
                    -0.013561362,
                    0.002373205,
                    0.00092512,
                    2.2532e-05,
                ],
                [
                    0.000132863,
                    0.003516954,
                    0.000847346,
                    -0.108372072,
                    -0.139178011,
                    0.052928748,
                    0.044837725,
                    0.002373205,
                    -0.000279806,
                ],
                [
                    0.000744712,
                    0.000389916,
                    -0.108372072,
                    -0.302454279,
                    0.249959607,
                    0.46016215,
                    0.052928748,
                    -0.013561362,
                    -0.001028923,
                ],
                [
                    3.79931e-05,
                    -0.022947700,
                    -0.139178011,
                    0.249959607,
                    1.000000000,
                    0.249959607,
                    -0.139178011,
                    -0.0229477,
                    3.79931e-05,
                ],
                [
                    -0.001028923,
                    -0.013561362,
                    0.052928748,
                    0.460162150,
                    0.249959607,
                    -0.302454279,
                    -0.108372072,
                    0.000389916,
                    0.000744712,
                ],
                [
                    -0.000279806,
                    0.002373205,
                    0.044837725,
                    0.052928748,
                    -0.139178011,
                    -0.108372072,
                    0.000847346,
                    0.003516954,
                    0.000132863,
                ],
                [
                    2.25320e-05,
                    0.000925120,
                    0.002373205,
                    -0.013561362,
                    -0.022947700,
                    0.000389916,
                    0.003516954,
                    0.000288732,
                    -9.04408e-06,
                ],
                [
                    4.04180e-06,
                    2.25320e-05,
                    -0.000279806,
                    -0.001028923,
                    3.79931e-05,
                    0.000744712,
                    0.000132863,
                    -9.04408e-06,
                    -1.01551e-06,
                ],
            ],
            dtype=np.float32,
        )

    def _extract_RGBI(
        self, src: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract R, G, B, and intensity channels.

        Args:
            input_image: float32 image in [0, 1] range.

        Returns:
            R: Red channel
            G: Green channel
            B: Blue channel
            I: Intensity channel
        """
        (B, G, R) = cv2.split(src)
        I = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return R, G, B, I

    def _create_gaussian_pyramid(self, src):
        """Create Gaussian pyramid with levels limited by image size."""
        h, w = src.shape[:2]
        min_dim = min(h, w)
        max_levels_from_size = 1 + int(np.floor(np.log2(min_dim)))
        effective_levels = min(self.pyramid_levels, max_levels_from_size)

        dst = [src]
        quick_method = False
        for i in range(1, effective_levels):
            # method 1
            if quick_method:
                now_dst = cv2.pyrDown(dst[i - 1])
            else:
                blurred = cv2.GaussianBlur(
                    dst[-1], (5, 5), 0, 0, borderType=cv2.BORDER_REFLECT_101
                )
                # Decimate by taking every other pixel starting at 0
                now_dst = blurred[i % 2 :: 2, i % 2 :: 2]
            dst.append(now_dst)

        return dst

    def _center_surround_diff(self, gaussian_maps):
        """Compute center-surround differences using whatever pyramid levels exist."""
        dst = []
        num_levels = len(gaussian_maps)
        if num_levels < 2:
            return dst

        if num_levels >= 9:
            center_levels = [2, 3, 4]
        else:
            upper_bound = max(2, num_levels - 3)
            center_levels = list(range(2, upper_bound))

        for center_level in center_levels:
            if center_level >= num_levels:
                continue
            target_shape = gaussian_maps[center_level].shape[:2]

            for delta in (3, 4):
                surround_level = center_level + delta
                if surround_level < num_levels:
                    resized_surround = resize_to(
                        gaussian_maps[surround_level],
                        target_shape,
                    )
                    diff = cv2.absdiff(gaussian_maps[center_level], resized_surround)
                    dst.append(diff)

        return dst

    def _gaussian_pyr_center_surround_diff(self, src):
        """Create Gaussian pyramid and compute center-surround differences"""
        gaussian_maps = self._create_gaussian_pyramid(src)
        dst = self._center_surround_diff(gaussian_maps)
        return dst

    def _get_intensity_feat_maps(self, I):
        """Get intensity feature maps"""
        return self._gaussian_pyr_center_surround_diff(I)

    def _get_color_feat_maps(
        self,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get color feature maps"""
        rgb_max = cv2.max(cv2.max(r, g), b)
        rgb_max[rgb_max <= 0] = 0.0001

        rg_min = cv2.min(r, g)

        rg = (r - g) / rgb_max
        by = (b - rg_min) / rgb_max

        rg[rg < 0] = 0
        by[by < 0] = 0

        rg_feat_maps = self._gaussian_pyr_center_surround_diff(rg)
        by_feat_maps = self._gaussian_pyr_center_surround_diff(by)

        return rg_feat_maps, by_feat_maps

    def _get_orientation_feat_maps(self, src: np.ndarray) -> list[list[np.ndarray]]:
        """Get orientation feature maps, skipping Gabor filters on small levels."""
        gaussian_I = self._create_gaussian_pyramid(src)

        gabor_output_0 = [np.empty((1, 1)), np.empty((1, 1))]
        gabor_output_45 = [np.empty((1, 1)), np.empty((1, 1))]
        gabor_output_90 = [np.empty((1, 1)), np.empty((1, 1))]
        gabor_output_135 = [np.empty((1, 1)), np.empty((1, 1))]

        for level_idx in range(2, len(gaussian_I)):
            h, w = gaussian_I[level_idx].shape[:2]
            if h < 9 or w < 9:
                dummy = np.zeros((h, w), dtype=np.float32)
                gabor_output_0.append(dummy)
                gabor_output_45.append(dummy)
                gabor_output_90.append(dummy)
                gabor_output_135.append(dummy)
                continue

            gabor_output_0.append(
                cv2.filter2D(gaussian_I[level_idx], cv2.CV_32F, self.GaborKernel0)
            )
            gabor_output_45.append(
                cv2.filter2D(gaussian_I[level_idx], cv2.CV_32F, self.GaborKernel45)
            )
            gabor_output_90.append(
                cv2.filter2D(gaussian_I[level_idx], cv2.CV_32F, self.GaborKernel90)
            )
            gabor_output_135.append(
                cv2.filter2D(gaussian_I[level_idx], cv2.CV_32F, self.GaborKernel135)
            )

        CSD_0 = self._center_surround_diff(gabor_output_0)
        CSD_45 = self._center_surround_diff(gabor_output_45)
        CSD_90 = self._center_surround_diff(gabor_output_90)
        CSD_135 = self._center_surround_diff(gabor_output_135)

        dst = list(CSD_0)
        dst.extend(CSD_45)
        dst.extend(CSD_90)
        dst.extend(CSD_135)

        return dst

    def _range_normalize(self, src):
        """Standard range normalization"""
        min_val, max_val, _, _ = cv2.minMaxLoc(src)
        if max_val != min_val:
            dst = src / (max_val - min_val) + min_val / (min_val - max_val)
        else:
            dst = src - min_val
        return dst

    def _avg_local_max(self, src):
        """Compute average of local maxima with adaptive stepsize."""
        n_patches = 8
        h, w = src.shape[:2]
        stepsize = max(n_patches, min(h, w) // n_patches)

        num_local = 0
        lmax_sum = 0

        for y in range(0, h - stepsize, stepsize):
            for x in range(0, w - stepsize, stepsize):
                local_patch = src[y : y + stepsize, x : x + stepsize]
                _, local_max, _, _ = cv2.minMaxLoc(local_patch)
                lmax_sum += local_max
                num_local += 1

        return lmax_sum / num_local if num_local > 0 else 0

    def _feat_map_normalization(self, src: np.ndarray) -> np.ndarray:
        """Normalization specific for saliency map model"""
        dst = self._range_normalize(src)
        lmax_mean = self._avg_local_max(dst)
        norm_coeff = (1 - lmax_mean) * (1 - lmax_mean)
        return dst * norm_coeff

    def _normalize_feat_maps(
        self,
        feature_maps: list[np.ndarray],
        shape: tuple[int, int],
    ) -> list[np.ndarray]:
        """Normalize feature maps and resize to target shape"""
        normalized_maps = []
        for feature_map in feature_maps:
            normalized = self._feat_map_normalization(feature_map)
            resized = resize_to(normalized, shape)
            normalized_maps.append(resized)
        return normalized_maps

    def _get_intensity_con_map(
        self,
        feat_maps: list[np.ndarray],
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Get intensity conspicuity map"""
        normed = self._normalize_feat_maps(feat_maps, shape)
        return sum(normed) if normed else np.zeros(shape)

    def _get_color_con_maps(
        self,
        rg_feat_maps: list[np.ndarray],
        by_feat_maps: list[np.ndarray],
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get color conspicuity map"""
        rg_con_map = self._get_intensity_con_map(rg_feat_maps, shape)
        by_con_map = self._get_intensity_con_map(by_feat_maps, shape)
        con_map = rg_con_map + by_con_map
        return rg_con_map, by_con_map, con_map

    def _get_orientation_con_map(
        self,
        feat_maps: list[np.ndarray],
        shape: tuple[int, int],
        ) -> np.ndarray:
        """Get orientation conspicuity map"""
        con_map = np.zeros(shape)
        maps_per_orientation = len(feat_maps) // 4 if len(feat_maps) >= 4 else 0

        for orientation_idx in range(4):
            start_idx = orientation_idx * maps_per_orientation
            end_idx = start_idx + maps_per_orientation

            if start_idx < len(feat_maps) and end_idx <= len(feat_maps):
                orientation_maps = feat_maps[start_idx:end_idx]
                if orientation_maps:
                    angle_con_mat = self._get_intensity_con_map(
                        orientation_maps, shape,
                    )
                    normalized_angle_con_mat = self._feat_map_normalization(angle_con_mat)
                    con_map += normalized_angle_con_mat

        return con_map

    def call(self, rgba: np.ndarray, depth: np.ndarray | None = None) -> IttiKochResult:
        """Compute Itti-Koch saliency map for the given observation.

        Args:
            rgba: RGBA image array (uint8 [0, 255] or float32 [0, 1])
            depth: Depth image array (optional). Ignored.

        Returns:
            sal: Saliency map normalized to [0, 1]
        """
        result = IttiKochResult(rgba=rgba, depth=depth)

        input_shape = rgba.shape[:2]

        if self.upsample_size is not None and self.upsample_size != input_shape:
            rgba = resize_to(rgba, self.upsample_size)

        src = as_rgba_float32(rgba)[:, :, :3]
        base_shape = src.shape[:2]

        result.info["src"] = src

        red, green, blue, intensity = self._extract_RGBI(src)
        result.info["RGBI"] = (red, green, blue, intensity)

        # intensity
        intensity_feat_maps = self._get_intensity_feat_maps(intensity)
        intensity_con_map = self._get_intensity_con_map(intensity_feat_maps, base_shape)
        result.feature_maps["intensity"] = intensity_feat_maps
        result.conspicuity_maps["intensity"] = intensity_con_map

        # color
        rg_feat_maps, by_feat_maps = self._get_color_feat_maps(red, green, blue)
        rg_con_map, by_con_map, color_con_map = self._get_color_con_maps(
            rg_feat_maps, by_feat_maps, base_shape
        )
        result.feature_maps["rg"] = rg_feat_maps
        result.feature_maps["by"] = by_feat_maps
        result.conspicuity_maps["rg"] = rg_con_map
        result.conspicuity_maps["by"] = by_con_map
        result.conspicuity_maps["color"] = color_con_map

        # orientation
        orientation_feat_maps = self._get_orientation_feat_maps(intensity)
        orientation_con_map = self._get_orientation_con_map(
            orientation_feat_maps, base_shape
        )
        result.feature_maps["orientation"] = orientation_feat_maps
        result.conspicuity_maps["orientation"] = orientation_con_map

        salience_matrix = (
            self.weight_intensity * intensity_con_map
            + self.weight_color * color_con_map
            + self.weight_orientation * orientation_con_map
        )

        normalized_salience = self._range_normalize(salience_matrix).astype(np.float32)
        smoothed_salience = cv2.bilateralFilter(normalized_salience, 7, 3, 1.55)
        if smoothed_salience.shape != input_shape:
            smoothed_salience = resize_to(smoothed_salience, input_shape)
        if np.max(smoothed_salience) > 1:
            # smoothed_salience = smoothed_salience / np.max(smoothed_salience)
            raise ValueError("should already be normalized")

        salience_map = smoothed_salience.astype(np.float32)
        result.salience_map = salience_map

        return result

    def __call__(self, rgba: np.ndarray, depth: np.ndarray | None = None) -> np.ndarray:
        return self.call(rgba, depth).salience_map




def as_rgba_float32(rgba: np.ndarray) -> np.ndarray:
    if np.issubdtype(rgba.dtype, np.integer):
        return (rgba / 255.0).astype(np.float32)
    return np.asarray(rgba, dtype=np.float32)

