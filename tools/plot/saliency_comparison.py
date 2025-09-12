# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from tbp.monty.frameworks.models.saliency.spectral_residual_salience import (
    SpectralResidualSalience,
)
from tbp.monty.frameworks.models.saliency.bio import BioSalience
from tbp.monty.frameworks.models.saliency.minimum_barrier_salience import (
    MinimumBarrierSalience,
)
from tbp.monty.frameworks.models.saliency.itti_koch_salience import IttiKochSalience

logger = logging.getLogger(__name__)


@dataclass
class SaliencyConfig:
    """Configuration settings for saliency comparison."""

    figure_size: Tuple[int, int] = (15, 8)
    dpi: int = 150
    grid_shape: Tuple[int, int] = (2, 3)  # 2 rows, 3 columns
    image_dir: Path = Path("~/tbp/model_free_saloon/results/images").expanduser()


class SaliencyComparator:
    """Compares different saliency detection methods."""

    def __init__(self, config: SaliencyConfig = None):
        self.config = config or SaliencyConfig()
        self._setup_logging()
        self._initialize_methods()
        self.current_image_path = None  # Track current image for depth loading

    def _setup_logging(self) -> None:
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def _initialize_methods(self) -> None:
        """Initialize all saliency detection methods."""
        try:
            self.methods = {
                "Spectral Residual": SpectralResidualSalience(),
                "BioSaliency": BioSalience(),
                "Minimum Barrier": MinimumBarrierSalience(),
                "Itti-Koch": IttiKochSalience(),
            }
            logger.info(f"Initialized {len(self.methods)} saliency methods")
        except Exception as e:
            logger.error(f"Failed to initialize methods: {e}")
            raise

    def _extract_rgb_from_obs(self, obs: dict) -> np.ndarray:
        """Extract RGB image from observation dictionary."""
        if "rgba" in obs:
            # Extract RGB from RGBA
            rgba = obs["rgba"]
            if rgba.shape[-1] == 4:
                return rgba[:, :, :3]
            else:
                return rgba
        elif "rgb" in obs:
            return obs["rgb"]
        elif "image" in obs:
            img = obs["image"]
            if img.shape[-1] >= 3:
                return img[:, :, :3]
            else:
                # Convert grayscale to RGB
                return np.stack([img] * 3, axis=-1)
        else:
            raise ValueError("No valid image found in observation dictionary")

    def _load_corresponding_depth(self) -> np.ndarray:
        """Load corresponding depth file for the current image."""
        if not self.current_image_path:
            return None

        try:
            # Convert rgba filename to depth filename
            image_path = Path(self.current_image_path)
            if "_rgba" in image_path.name:
                depth_filename = image_path.name.replace("_rgba.npy", "_depth.npy")
            else:
                # If no _rgba suffix, try adding _depth before .npy
                depth_filename = image_path.stem + "_depth.npy"

            depth_path = image_path.parent / depth_filename

            if depth_path.exists():
                depth_data = np.load(depth_path)
                logger.info(
                    f"Loaded depth file: {depth_path}, shape: {depth_data.shape}"
                )

                # Handle different depth data formats
                if len(depth_data.shape) == 3:
                    if depth_data.shape[2] == 1:
                        # Single channel depth
                        depth_data = depth_data[:, :, 0]
                    else:
                        # Multiple channels, take first
                        depth_data = depth_data[:, :, 0]
                elif len(depth_data.shape) == 4:
                    # Batched depth data
                    depth_data = (
                        depth_data[0, :, :, 0]
                        if depth_data.shape[3] == 1
                        else depth_data[0, :, :, 0]
                    )

                return depth_data
            else:
                logger.warning(f"Depth file not found: {depth_path}")
                return None

        except Exception as e:
            logger.error(f"Error loading depth file: {e}")
            return None

    def compute_saliency_maps(self, obs: dict) -> Dict[str, np.ndarray]:
        """Compute saliency maps for all methods."""
        saliency_results = {}

        for method_name, method in self.methods.items():
            try:
                logger.info(f"Computing saliency using {method_name}")
                saliency_map = method.compute_saliency_map(obs)

                # Normalize to [0, 1] if needed
                if saliency_map.max() > 1.0:
                    saliency_map = saliency_map / 255.0

                saliency_results[method_name] = saliency_map
                logger.info(
                    f"  {method_name}: shape={saliency_map.shape}, "
                    f"range=[{saliency_map.min():.3f}, {saliency_map.max():.3f}]"
                )

            except Exception as e:
                logger.error(f"Error computing {method_name}: {e}")
                # Create a zero map as fallback
                rgb_img = self._extract_rgb_from_obs(obs)
                h, w = rgb_img.shape[:2]
                saliency_results[method_name] = np.zeros((h, w), dtype=np.float32)

        return saliency_results

    def create_comparison_plot(
        self,
        obs: dict,
        saliency_results: Dict[str, np.ndarray],
        save_path: Path = None,
        show_plot: bool = True,
    ) -> Figure:
        """Create and optionally save a comparison plot of all saliency methods."""
        try:
            # Extract original RGB image
            original_rgb = self._extract_rgb_from_obs(obs)

            rows, cols = self.config.grid_shape
            fig = plt.figure(figsize=self.config.figure_size)

            # Create grid layout with original image spanning left column
            gs = fig.add_gridspec(rows, cols)

            # Original image spanning both rows in first column
            ax_orig = fig.add_subplot(gs[:, 0])
            ax_orig.imshow(original_rgb)
            ax_orig.set_title(
                f"Original RGB\n{original_rgb.shape[:2]}",
                fontsize=14,
                fontweight="bold",
            )
            ax_orig.axis("off")

            # Saliency maps in positions (0,1), (0,2), (1,1), (1,2)
            method_names = list(saliency_results.keys())
            positions = [(0, 1), (0, 2), (1, 1), (1, 2)]  # (row, col)

            for idx, method_name in enumerate(method_names):
                if idx < len(positions):
                    row, col = positions[idx]
                    ax = fig.add_subplot(gs[row, col])

                    saliency_map = saliency_results[method_name]
                    im = ax.imshow(saliency_map, cmap="hot", vmin=0, vmax=1)
                    ax.set_title(
                        f"{method_name}\n{saliency_map.shape}",
                        fontsize=12,
                        fontweight="bold",
                    )
                    ax.axis("off")

                    # Add colorbar for saliency maps
                    plt.colorbar(im, ax=ax, shrink=0.8)

            plt.tight_layout(pad=2.0)

            if save_path:
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
                logger.info(f"Saved comparison plot to {save_path}")

            if show_plot:
                plt.show()

            return fig

        except Exception as e:
            logger.error(f"Failed to create comparison plot: {e}")
            raise

    def compare_methods(
        self, obs: dict, save_path: Path = None, show_plot: bool = True
    ) -> Tuple[Figure, Dict[str, np.ndarray]]:
        """Complete pipeline to compare saliency methods."""
        saliency_results = self.compute_saliency_maps(obs)
        fig = self.create_comparison_plot(obs, saliency_results, save_path, show_plot)
        return fig, saliency_results


def load_image(image_path: Path) -> dict:
    img = np.load(image_path)
    logger.info(
        f"Loaded image shape: {img.shape}, dtype: {img.dtype}, range: [{img.min():.3f}, {img.max():.3f}]"
    )
    
    # Ensure data is float32 and in [0,1] range for consistency
    if img.dtype == np.float64:
        img = img.astype(np.float32)
    
    # Normalize to [0,1] if needed
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    
    if img.shape[-1] == 4:
        # RGBD format
        obs = {"rgba": img[:, :, :3], "depth": img[:, :, 3]}
        logger.info("Loaded RGBA image")
    elif img.shape[-1] == 5:
        # RGBA with depth
        obs = {"rgba": img[:, :, :4], "depth": img[:, :, 4]}
        logger.info("Loaded RGBA with depth image")
    else:
        raise ValueError(f"Unexpected number of channels: {img.shape[-1]}")

    return obs


def main(image_name: str = None):
    """Main entry point for testing saliency comparison."""
    config = SaliencyConfig()
    comparator = SaliencyComparator(config)

    for image_path in config.image_dir.glob("*.npy"):
        obs = load_image(image_path)
        image_name = image_path.stem


        output_name = (
            f"saliency_comparison_{image_name}" if image_name else "saliency_comparison"
        )
        fig, results = comparator.compare_methods(
            obs,
            save_path=Path(f"{output_name}.png"),
            show_plot=True,
        )

        logger.info("Saliency comparison completed successfully!")


if __name__ == "__main__":
    main()
