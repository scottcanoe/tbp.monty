import numpy as np
import cv2


class SpectralResidualSalience:
    def extract_gray_image(self, image: np.ndarray) -> np.ndarray:
        """Extract the gray image from the RGBA image."""

        img_rgb = image[:, :, :3]
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    def compute_saliency_map(self, obs: dict) -> np.ndarray:
        """Compute saliency map using OpenCV's Spectral Residual method.

        Args:
            obs: Dictionary containing observation data. Expected to have an 'image' key
                 with RGB image data as numpy array.

        Returns:
            Saliency map as numpy array normalized to [0, 255] as uint8
        """
        saliency_method = cv2.saliency.StaticSaliencySpectralResidual_create()

        rgba = obs.get("rgba")
        image = self.extract_gray_image(rgba)

        _, saliency_map = saliency_method.computeSaliency(image)

        assert saliency_map.shape == image.shape, (
            f"Saliency map shape {saliency_map.shape} does not match image shape {image.shape}"
        )
        assert np.max(saliency_map) <= 1.0, f"Saliency map is not normalized to [0, 1]"
        assert np.min(saliency_map) >= 0.0, f"Saliency map is not normalized to [0, 1]"

        return saliency_map
