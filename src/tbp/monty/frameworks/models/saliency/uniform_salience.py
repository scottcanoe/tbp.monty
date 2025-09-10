import numpy as np


class UniformSalience:
    """Uniform saliency strategy that assigns equal salience to all pixels."""

    def compute_saliency_map(self, obs: dict) -> np.ndarray:
        """Compute uniform saliency map from observation dictionary.

        Args:
            obs: The observation dictionary containing 'depth' key.

        Returns:
            A numpy array with uniform saliency values matching the depth shape.
        """
        return np.ones_like(obs["depth"])
