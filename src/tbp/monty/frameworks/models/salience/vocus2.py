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

import cv2
import numpy as np
from skimage.feature import peak_local_max

from .common import resize_to
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


class PyrStructure(Enum):
    """Pyramid structure options."""

    CLASSIC = 0  # Two independent pyramids
    CODI = 1  # Two pyramids derived from a base pyramid
    NEW = 2  # Surround pyramid derived from center pyramid (CVPR 2015)


class VOCUS2Config:
    """Configuration for VOCUS2 model."""

    def __init__(
        self,
        c_space: ColorSpace = ColorSpace.OPPONENT_CODI,
        fuse_feature: FusionOperation = FusionOperation.ARITHMETIC_MEAN,
        fuse_conspicuity: FusionOperation = FusionOperation.ARITHMETIC_MEAN,
        start_layer: int = 0,
        stop_layer: int = 4,
        center_sigma: float = 3.0,
        surround_sigma: float = 13.0,
        n_scales: int = 2,
        normalize: bool = True,
        pyr_struct: PyrStructure = PyrStructure.NEW,
        orientation: bool = False,
        combined_features: bool = False,
        center_bias: float | None = None,
    ):
        """Initialize VOCUS2 configuration.

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
            pyr_struct: Pyramid structure to use
            orientation: Whether to compute orientation features
            combined_features: Whether to combine color channels before fusion
        """
        self.c_space = c_space
        self.fuse_feature = fuse_feature
        self.fuse_conspicuity = fuse_conspicuity
        self.start_layer = start_layer
        self.stop_layer = stop_layer
        self.center_sigma = center_sigma
        self.surround_sigma = surround_sigma
        self.n_scales = n_scales
        self.normalize = normalize
        self.pyr_struct = pyr_struct
        self.orientation = orientation
        self.combined_features = combined_features
        self.center_bias = center_bias

@dataclass
class VOCUS2Result:
    rgba: np.ndarray | None = None
    depth: np.ndarray | None = None
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

    def __init__(self, cfg: VOCUS2Config | None = None):
        """Initialize VOCUS2 with configuration.

        Args:
            cfg: VOCUS2 configuration. If None, uses defaults.
        """
        self.cfg = cfg if cfg is not None else VOCUS2Config()

        # Internal state
        self.input = None
        self.salmap = None
        self.salmap_ready = False
        self.splitted_ready = False
        self.processed = False

        # Feature pyramids (2D: [octave][scale])
        self.pyr_center_L = []
        self.pyr_center_a = []
        self.pyr_center_b = []
        self.pyr_surround_L = []
        self.pyr_surround_a = []
        self.pyr_surround_b = []

        # Center-surround contrast pyramids (1D: flattened octave*scale)
        self.on_off_L = []
        self.off_on_L = []
        self.on_off_a = []
        self.off_on_a = []
        self.on_off_b = []
        self.off_on_b = []

        # Orientation features
        self.gabor = []  # [orientation][octave*scale]
        self.pyr_laplace = []

        # Color planes
        self.planes = []

    def _clear(self):
        """Clear all internal state."""
        self.salmap = None
        self.on_off_L = []
        self.off_on_L = []
        self.on_off_a = []
        self.off_on_a = []
        self.on_off_b = []
        self.off_on_b = []
        self.pyr_center_L = []
        self.pyr_surround_L = []
        self.pyr_center_a = []
        self.pyr_surround_a = []
        self.pyr_center_b = []
        self.pyr_surround_b = []
        self.gabor = []
        self.pyr_laplace = []

    def _prepare_input(self, rgb: np.ndarray) -> list[np.ndarray]:
        """Convert image to desired color space and split into planes.

        Args:
            rgb: RGB image. Should be uint8 in [0, 255].

        Returns:
            List of 3 planes in the target color space (e.g., [L, a, b] for LAB).

        Raises:
            ValueError: If unsupported color space is specified.
        """
        if self.cfg.c_space == ColorSpace.LAB:
            # Convert to LAB
            converted = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab) / 255.0
            planes = cv2.split(converted)

        elif self.cfg.c_space == ColorSpace.OPPONENT_CODI:
            # Opponent color space (CoDi style)
            converted = rgb.astype(np.float32)
            R, G, B = cv2.split(converted)
            L = (R + G + B) / (3 * 255.0)
            a = (R - G) / 255.0
            b = (B - (G + R) / 2.0) / 255.0
            planes = [L, a, b]

        elif self.cfg.c_space == ColorSpace.OPPONENT:
            # Opponent color space (shifted to [0,1])
            converted = rgb.astype(np.float32)
            R, G, B = cv2.split(converted)
            L = (R + G + B) / (3 * 255.0)
            a = (R - G + 255.0) / (2 * 255.0)
            b = (B - (G + R) / 2.0 + 255.0) / (2 * 255.0)
            planes = [L, a, b]

        else:
            raise ValueError(f"Unsupported color space: {self.cfg.c_space}")

        return [p.astype(np.float32) for p in planes]


    def _build_multiscale_pyr(
        self,
        src: np.ndarray,
        sigma: float,
    ) -> list[list[np.ndarray]]:
        """Build multi-scale pyramid following Lowe 2004.

        This creates a 2D pyramid structure:
        - Dimension 1 (octaves): Different resolutions (each half the previous)
        - Dimension 2 (scales): Different smoothing levels within each octave

        Args:
            src: Input image (single channel, float32)
            sigma: Base sigma for Gaussian smoothing

        Returns:
            2D list [octave][scale] of pyramid images
        """
        # Calculate maximum number of octaves
        min_dim = min(src.shape[0], src.shape[1])
        max_octaves = min(int(np.log2(min_dim)), self.cfg.stop_layer) + 1
        n_octaves = max_octaves - self.cfg.start_layer

        # Quickly resize dummy data until we get to the first octave we want to use.
        tmp = src.copy()  # necessary?
        for _ in range(self.cfg.start_layer):
            tmp = cv2.GaussianBlur(
                tmp, (0, 0), 2.0 * sigma, borderType=cv2.BORDER_REPLICATE
            )
            tmp = cv2.resize(tmp, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

        # Compute pyramid as in Lowe 2004
        sig_prev = 0.0
        sig_total = 0.0
        pyr = []
        for octave in range(n_octaves):
            pyr.append([])

            # Compute n_scales + 1 (extra scale used as first of next octave)
            for scale in range(self.cfg.n_scales + 1):
                # First scale of first octave: smooth tmp
                if octave == 0 and scale == 0:
                    src = tmp
                    sig_total = (2.0 ** (scale / self.cfg.n_scales)) * sigma
                    dst = cv2.GaussianBlur(
                        src, (0, 0), sig_total, borderType=cv2.BORDER_REPLICATE
                    )
                    sig_prev = sig_total

                # First scale of other octaves: subsample additional scale of previous
                elif octave != 0 and scale == 0:
                    src = pyr[octave - 1][self.cfg.n_scales]
                    dst = cv2.resize(
                        src,
                        (src.shape[1] // 2, src.shape[0] // 2),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    sig_prev = sigma

                # Intermediate scales: smooth previous scale
                else:
                    sig_total = (2.0 ** (scale / self.cfg.n_scales)) * sigma
                    sig_diff = np.sqrt(sig_total**2 - sig_prev**2)
                    src = pyr[octave][scale - 1]
                    dst = cv2.GaussianBlur(
                        src, (0, 0), sig_diff, borderType=cv2.BORDER_REPLICATE
                    )
                    sig_prev = sig_total

                pyr[octave].append(dst)

        # Erase the temporary scale in each octave that was just used to
        # compute the sigmas for the next octave.
        for octave in range(len(pyr)):
            pyr[octave].pop()

        return pyr

    def _pyramid_new(self, img: np.ndarray):
        """Build pyramids using NEW structure (CVPR 2015 default).

        Builds center pyramid first, then derives surround pyramid from it.
        """
        self._clear()

        # Prepare input
        self.planes = self._prepare_input(img)

        # Build center pyramids
        self.pyr_center_L = self._build_multiscale_pyr(
            self.planes[0], float(self.cfg.center_sigma)
        )
        self.pyr_center_a = self._build_multiscale_pyr(
            self.planes[1], float(self.cfg.center_sigma)
        )
        self.pyr_center_b = self._build_multiscale_pyr(
            self.planes[2], float(self.cfg.center_sigma)
        )

        # Compute adapted sigma for surround
        adapted_sigma = np.sqrt(
            self.cfg.surround_sigma**2 - self.cfg.center_sigma**2
        )

        # Build surround pyramids by additional smoothing of center pyramids
        self.pyr_surround_L = []
        self.pyr_surround_a = []
        self.pyr_surround_b = []

        for octave in range(len(self.pyr_center_L)):
            self.pyr_surround_L.append([])
            self.pyr_surround_a.append([])
            self.pyr_surround_b.append([])

            for scale in range(self.cfg.n_scales):
                scaled_sigma = adapted_sigma * (2.0 ** (scale / self.cfg.n_scales))

                surr_L = cv2.GaussianBlur(
                    self.pyr_center_L[octave][scale],
                    (0, 0),
                    scaled_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )
                surr_a = cv2.GaussianBlur(
                    self.pyr_center_a[octave][scale],
                    (0, 0),
                    scaled_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )
                surr_b = cv2.GaussianBlur(
                    self.pyr_center_b[octave][scale],
                    (0, 0),
                    scaled_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )

                self.pyr_surround_L[octave].append(surr_L)
                self.pyr_surround_a[octave].append(surr_a)
                self.pyr_surround_b[octave].append(surr_b)

    def _pyramid_classic(self, img: np.ndarray):
        """Build pyramids using CLASSIC structure.

        Builds center and surround pyramids independently.
        """
        self._clear()
        self.salmap_ready = False
        self.splitted_ready = False

        # Prepare input
        self.planes = self._prepare_input(img)

        # Build center and surround pyramids independently
        self.pyr_center_L = self._build_multiscale_pyr(
            self.planes[0], float(self.cfg.center_sigma)
        )
        self.pyr_center_a = self._build_multiscale_pyr(
            self.planes[1], float(self.cfg.center_sigma)
        )
        self.pyr_center_b = self._build_multiscale_pyr(
            self.planes[2], float(self.cfg.center_sigma)
        )

        self.pyr_surround_L = self._build_multiscale_pyr(
            self.planes[0], float(self.cfg.surround_sigma)
        )
        self.pyr_surround_a = self._build_multiscale_pyr(
            self.planes[1], float(self.cfg.surround_sigma)
        )
        self.pyr_surround_b = self._build_multiscale_pyr(
            self.planes[2], float(self.cfg.surround_sigma)
        )

    def _pyramid_codi(self, img: np.ndarray):
        """Build pyramids using CODI structure.

        Builds base pyramid first, then derives center and surround from it.
        """
        self._clear()
        self.salmap_ready = False
        self.splitted_ready = False

        # Prepare input
        self.planes = self._prepare_input(img)

        # Build base pyramids with sigma=1
        pyr_base_L = self._build_multiscale_pyr(self.planes[0], 1.0)
        pyr_base_a = self._build_multiscale_pyr(self.planes[1], 1.0)
        pyr_base_b = self._build_multiscale_pyr(self.planes[2], 1.0)

        # Compute adapted sigmas
        adapted_center_sigma = np.sqrt(self.cfg.center_sigma**2 - 1.0)
        adapted_surround_sigma = np.sqrt(self.cfg.surround_sigma**2 - 1.0)

        # Initialize center and surround pyramids
        self.pyr_center_L = []
        self.pyr_center_a = []
        self.pyr_center_b = []
        self.pyr_surround_L = []
        self.pyr_surround_a = []
        self.pyr_surround_b = []

        # For each octave
        for octave in range(len(pyr_base_L)):
            self.pyr_center_L.append([])
            self.pyr_center_a.append([])
            self.pyr_center_b.append([])
            self.pyr_surround_L.append([])
            self.pyr_surround_a.append([])
            self.pyr_surround_b.append([])

            # For each scale
            for scale in range(self.cfg.n_scales):
                scaled_center_sigma = adapted_center_sigma * (
                    2.0 ** (scale / self.cfg.n_scales)
                )
                scaled_surround_sigma = adapted_surround_sigma * (
                    2.0 ** (scale / self.cfg.n_scales)
                )

                # Smooth base pyramid to get center and surround
                center_L = cv2.GaussianBlur(
                    pyr_base_L[octave][scale],
                    (0, 0),
                    scaled_center_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )
                surround_L = cv2.GaussianBlur(
                    pyr_base_L[octave][scale],
                    (0, 0),
                    scaled_surround_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )

                center_a = cv2.GaussianBlur(
                    pyr_base_a[octave][scale],
                    (0, 0),
                    scaled_center_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )
                surround_a = cv2.GaussianBlur(
                    pyr_base_a[octave][scale],
                    (0, 0),
                    scaled_surround_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )

                center_b = cv2.GaussianBlur(
                    pyr_base_b[octave][scale],
                    (0, 0),
                    scaled_center_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )
                surround_b = cv2.GaussianBlur(
                    pyr_base_b[octave][scale],
                    (0, 0),
                    scaled_surround_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )

                self.pyr_center_L[octave].append(center_L)
                self.pyr_center_a[octave].append(center_a)
                self.pyr_center_b[octave].append(center_b)
                self.pyr_surround_L[octave].append(surround_L)
                self.pyr_surround_a[octave].append(surround_a)
                self.pyr_surround_b[octave].append(surround_b)

    def _center_surround_diff(self):
        """Compute center-surround differences (on-off and off-on)."""
        on_off_size = len(self.pyr_center_L) * self.cfg.n_scales

        self.on_off_L = [None] * on_off_size
        self.off_on_L = [None] * on_off_size
        self.on_off_a = [None] * on_off_size
        self.off_on_a = [None] * on_off_size
        self.on_off_b = [None] * on_off_size
        self.off_on_b = [None] * on_off_size

        # Compute DoG by subtracting pyramids
        for octave in range(len(self.pyr_center_L)):
            for scale in range(self.cfg.n_scales):
                pos = octave * self.cfg.n_scales + scale

                # L channel
                diff = self.pyr_center_L[octave][scale] - self.pyr_surround_L[octave][scale]
                self.on_off_L[pos] = np.maximum(diff, 0)
                self.off_on_L[pos] = np.maximum(-diff, 0)

                # a channel
                diff = self.pyr_center_a[octave][scale] - self.pyr_surround_a[octave][scale]
                self.on_off_a[pos] = np.maximum(diff, 0)
                self.off_on_a[pos] = np.maximum(-diff, 0)

                # b channel
                diff = self.pyr_center_b[octave][scale] - self.pyr_surround_b[octave][scale]
                self.on_off_b[pos] = np.maximum(diff, 0)
                self.off_on_b[pos] = np.maximum(-diff, 0)

    def _orientation(self):
        """Compute orientation features using Gabor filters."""
        # Build Laplacian pyramid
        self.pyr_laplace = []
        for _ in range(len(self.pyr_center_L)):
            self.pyr_laplace.append([])

        # Build all layers except last
        for octave in range(len(self.pyr_center_L) - 1):
            for scale in range(len(self.pyr_center_L[octave])):
                src1 = self.pyr_center_L[octave][scale]
                src2 = self.pyr_center_L[octave + 1][scale]

                # Resize next octave to current size
                # tmp = cv2.resize(src2, (src1.shape[1], src1.shape[0]), interpolation=cv2.INTER_NEAREST)
                tmp = resize_to(src2, src1.shape)
                laplacian = src1 - tmp
                self.pyr_laplace[octave].append(laplacian)

        # Copy last layer
        for scale in range(self.cfg.n_scales):
            last_octave = len(self.pyr_center_L) - 1
            self.pyr_laplace[last_octave].append(
                self.pyr_center_L[last_octave][scale].copy()
            )

        # Create Gabor kernels
        filter_size = int(11 * self.cfg.center_sigma + 1)
        if filter_size % 2 == 0:
            filter_size += 1

        self.gabor = [[] for _ in range(4)]  # 4 orientations

        for ori in range(4):
            theta = ori * np.pi / 4
            gabor_kernel = cv2.getGaborKernel(
                (filter_size, filter_size),
                sigma=self.cfg.center_sigma,
                theta=theta,
                lambd=self.cfg.center_sigma * 2,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            )

            # Apply Gabor filter to each scale
            for octave in range(len(self.pyr_laplace)):
                for scale in range(len(self.pyr_laplace[octave])):
                    if self.pyr_laplace[octave][scale] is not None:
                        src = self.pyr_laplace[octave][scale]
                        if src.shape[0] >= filter_size and src.shape[1] >= filter_size:
                            dst = cv2.filter2D(src, cv2.CV_32F, gabor_kernel)
                            dst = np.abs(dst)
                            self.gabor[ori].append(dst)

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


    def _fuse(self, maps: list[np.ndarray], op: FusionOperation) -> np.ndarray:
        """Fuse multiple maps using specified operation.

        Args:
            maps: List of maps to fuse
            op: Fusion operation to use

        Returns:
            Fused map
        """
        if not maps:
            return np.zeros((1, 1), dtype=np.float32)

        # Get target size from first map
        target_size = maps[0].shape[:2]
        fused = np.zeros(target_size, dtype=np.float32)

        # Resize all maps to target size
        resized = []
        for m in maps:
            if m.shape[:2] != target_size:
                r = cv2.resize(
                    m,
                    (target_size[1], target_size[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
                resized.append(r)
            else:
                resized.append(m)

        if op == FusionOperation.ARITHMETIC_MEAN:
            # Simple average
            for r in resized:
                fused += r
            fused /= len(resized)

        elif op == FusionOperation.MAX:
            # Maximum value
            fused = resized[0].copy()
            for r in resized[1:]:
                fused = np.maximum(fused, r)

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
        if self.cfg.normalize:
            min_val = np.min(salience_map)
            max_val = np.max(salience_map)
            if max_val > min_val:
                salience_map = (salience_map - min_val) / (max_val - min_val)

        return salience_map

    def _compute_feature_maps(self, rgb: np.ndarray):
        """Generate pyramids and compute center-surround difference feature-maps.

        Args:
            rgb: Input rgb image (uint8)
        """
        self.input = rgb.copy()

        # Build pyramids according to structure
        if self.cfg.pyr_struct == PyrStructure.NEW:
            self._pyramid_new(rgb)
        elif self.cfg.pyr_struct == PyrStructure.CODI:
            self._pyramid_codi(rgb)
        elif self.cfg.pyr_struct == PyrStructure.CLASSIC:
            self._pyramid_classic(rgb)
        else:
            raise ValueError(f"Unsupported pyramid structure: {self.cfg.pyr_struct}")

        # Compute center-surround differences/feature-maps.
        self._center_surround_diff()

        # Compute orientation features if requested
        if self.cfg.orientation:
            self._orientation()

    def _compute_salience_map(self) -> np.ndarray:
        """Get final saliency map.

        Returns:
            Saliency map normalized to [0,1]

        Raises:
            RuntimeError: If image not yet processed. Call process() first.
        """
        # Intensity feature maps
        feature_intensity = [
            self._fuse(self.on_off_L, self.cfg.fuse_feature),
            self._fuse(self.off_on_L, self.cfg.fuse_feature),
        ]

        # Color feature maps
        if self.cfg.combined_features:
            # Combine all color channels
            feature_color1 = [
                self._fuse(self.on_off_a, self.cfg.fuse_feature),
                self._fuse(self.off_on_a, self.cfg.fuse_feature),
                self._fuse(self.on_off_b, self.cfg.fuse_feature),
                self._fuse(self.off_on_b, self.cfg.fuse_feature),
            ]
            feature_color2 = []
        else:
            # Keep color channels separate
            feature_color1 = [
                self._fuse(self.on_off_a, self.cfg.fuse_feature),
                self._fuse(self.off_on_a, self.cfg.fuse_feature),
            ]
            feature_color2 = [
                self._fuse(self.on_off_b, self.cfg.fuse_feature),
                self._fuse(self.off_on_b, self.cfg.fuse_feature),
            ]

        # Orientation feature maps
        feature_orientation = []
        if self.cfg.orientation and self.cfg.combined_features:
            for i in range(4):
                feature_orientation.append(self._fuse(self.gabor[i], self.cfg.fuse_feature))

        # Conspicuity maps
        conspicuity_maps = []
        conspicuity_maps.append(
            self._fuse(feature_intensity, self.cfg.fuse_conspicuity)
        )

        if self.cfg.combined_features:
            conspicuity_maps.append(
                self._fuse(feature_color1, self.cfg.fuse_conspicuity)
            )
            if self.cfg.orientation:
                conspicuity_maps.append(
                    self._fuse(feature_orientation, self.cfg.fuse_conspicuity)
                )
        else:
            conspicuity_maps.append(
                self._fuse(feature_color1, self.cfg.fuse_conspicuity)
            )
            conspicuity_maps.append(
                self._fuse(feature_color2, self.cfg.fuse_conspicuity)
            )
            if self.cfg.orientation:
                for i in range(4):
                    conspicuity_maps.append(
                        self._fuse(self.gabor[i], self.cfg.fuse_feature)
                    )

        # Final saliency map
        self.salmap = self._fuse(conspicuity_maps, self.cfg.fuse_conspicuity)

        # Normalize to [0,1]
        if self.cfg.normalize:
            min_val = np.min(self.salmap)
            max_val = np.max(self.salmap)
            if max_val > min_val:
                self.salmap = (self.salmap - min_val) / (max_val - min_val)

        # Resize to original image size
        if self.input is not None:
            self.salmap = cv2.resize(
                self.salmap,
                (self.input.shape[1], self.input.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        return self.salmap


    def call(self, rgba: np.ndarray, depth: np.ndarray | None = None) -> VOCUS2Result:
        """Compute VOCUS2 saliency map (SalienceStrategy interface).

        Args:
            rgba: RGBA image (uint8 or float32)
            depth: Depth image (optional, not used)

        Returns:
            Saliency map in [0, 1] range
        """
        result = VOCUS2Result(rgba=rgba, depth=depth)

        # Convert RGBA to BGR if needed
        rgb = rgba[:, :, :3]
        if rgb.dtype == np.uint8:
            pass
        elif np.issubdtype(rgb.dtype, np.integer):
            rgb = rgb.astype(np.uint8)
        else:
            rgb = (rgb * 255).astype(np.uint8)

        planes = self._prepare_input(rgb)

        self._compute_feature_maps(rgb)
        salience_map = self._compute_salience_map()

        if self.cfg.center_bias:
            salience_map = self._add_center_bias(self.cfg.center_bias)

        result.salience_map = salience_map
        return result

    def __call__(self, rgba: np.ndarray, depth: np.ndarray | None = None) -> VOCUS2Result:
        """Compute VOCUS2 saliency map (SalienceStrategy interface)."""
        return self.call(rgba, depth).salience_map


def resize(
    array: np.ndarray,
    shape: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize array to target shape using opencv."""
    return cv2.resize(array, shape[::-1], interpolation=interpolation)
