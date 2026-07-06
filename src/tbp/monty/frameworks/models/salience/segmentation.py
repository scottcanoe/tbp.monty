from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from skimage.segmentation import slic


class SlicMerge:

    def __init__(
        self,
        n_segments: int = 50,
        compactness: float = 10.0,
        max_iter: int = 10,
        sigma: float = 1.0,
        enforce_connectivity: bool = True,
        min_size_factor: float = 0.5,
        max_size_factor: float = 3.0,
        slic_zero: bool = False,
        convert2lab: bool = True,
        merge_threshold: float = 10,
        merge_in_lab: bool = True,
    ) -> None:
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.sigma = sigma
        self.enforce_connectivity = enforce_connectivity
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.slic_zero = slic_zero
        self.convert2lab = convert2lab
        self.merge_threshold = merge_threshold
        self.merge_in_lab = merge_in_lab

    def segment(self, image: np.ndarray) -> np.ndarray:
        y, x = (image.shape[0] // 2, image.shape[1] // 2)  # Fixation is at center of image
        h, w = image.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        compactness = self.compactness

        # Run SLIC with enhanced parameters
        n_seg = max(1, int(self.n_segments))
        labels = slic(
            image,
            n_segments=n_seg,
            compactness=compactness,
            max_num_iter=self.max_iter,
            sigma=self.sigma,
            spacing=None,  # Use default spacing
            convert2lab=self.convert2lab,
            enforce_connectivity=self.enforce_connectivity,
            min_size_factor=self.min_size_factor,
            max_size_factor=self.max_size_factor,
            slic_zero=self.slic_zero,
            start_label=0,
            mask=None,
            channel_axis=-1,
        )

        # Compute mean colors for merging
        if self.merge_in_lab:
            # Convert to LAB for perceptual color distance
            color_space = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        else:
            # Use RGB directly
            color_space = image.astype(np.float32)

        # Compute mean color per superpixel
        n_labels = labels.max() + 1
        mean_colors = np.zeros((n_labels, 3), dtype=np.float32)

        # Vectorized computation of mean colors
        for lbl in range(n_labels):
            mask = labels == lbl
            if mask.any():
                mean_colors[lbl] = color_space[mask].mean(axis=0)

        # Build adjacency graph
        adj: list[set[int]] = [set() for _ in range(n_labels)]

        # Check horizontal neighbors
        h_neighbors = labels[:, :-1] != labels[:, 1:]
        for i, j in zip(*np.where(h_neighbors)):
            a, b = labels[i, j], labels[i, j + 1]
            # assert isinstance(a, int) and isinstance(b, int)  # for type checker
            adj[a].add(b)
            adj[b].add(a)

        # Check vertical neighbors
        v_neighbors = labels[:-1, :] != labels[1:, :]
        for i, j in zip(*np.where(v_neighbors)):
            a, b = labels[i, j], labels[i + 1, j]
            # assert isinstance(a, int) and isinstance(b, int)  # for type checker
            adj[a].add(b)
            adj[b].add(a)

        # BFS merge from seed superpixel
        seed_label = labels[y, x]

        # Use appropriate distance metric
        if self.merge_in_lab:
            # CIE76 color difference for LAB
            # Standard threshold values:
            # < 1: not perceptible
            # 1-2: perceptible through close observation
            # 2-10: perceptible at a glance
            # 10-50: colors are more similar than opposite
            # > 50: colors are more opposite than similar
            # Already in good range for LAB
            thresh = self.merge_threshold
        else:
            # Euclidean distance in RGB
            thresh = self.merge_threshold * 2.55  # Scale to 0-255 range

        visited = {seed_label}
        queue = deque([seed_label])
        foreground = {seed_label}

        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                # Compute color distance
                dist = np.linalg.norm(mean_colors[current] - mean_colors[neighbor])

                if dist < thresh:
                    foreground.add(neighbor)
                    queue.append(neighbor)

        # Create output mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for lbl in foreground:
            mask[labels == lbl] = 1

        return mask
