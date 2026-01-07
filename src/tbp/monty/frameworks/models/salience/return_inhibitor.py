# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit


@njit
def compute_spatial_weight(
    points: np.ndarray,
    kernel_location: np.ndarray,
    tau_s: float,
    spatial_cutoff: float,
) -> np.ndarray:
    """Compute the spatial weight for a given set of points and kernel location.

    Args:
        points: A (num_points, 3) array of points.
        kernel_location: A (3,) array of the kernel location.
        tau_s: The spatial decay rate.
        spatial_cutoff: The spatial cutoff distance. Set to <= 0 for no cutoff.

    Returns:
        An (num_points,) array of the spatial weights.
    """
    out = np.empty(points.shape[0])
    for i in range(points.shape[0]):
        dx = points[i, 0] - kernel_location[0]
        dy = points[i, 1] - kernel_location[1]
        dz = points[i, 2] - kernel_location[2]
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        wt = np.exp(-dist / (tau_s / np.log(2)))
        if spatial_cutoff > 0 and dist > spatial_cutoff:
            wt = 0.0
        out[i] = wt
    return out


def compute_spatial_weight_old(
    points: np.ndarray,
    kernel_location: np.ndarray,
    tau_s: float,
    spatial_cutoff: float,
) -> np.ndarray:
    dist = np.linalg.norm(kernel_location - points, axis=1)
    out = np.exp(-dist / (tau_s / np.log(2)))
    if spatial_cutoff > 0:
        out[dist > spatial_cutoff] = 0.0
    return out


class DecayKernel:
    """Decay kernel represents a previously visited location.

    Returns the product of an time- and space- dependent exponentials.
    """

    def __init__(
        self,
        location: np.ndarray,
        tau_t: float = 4.0,
        tau_s: float = 0.01,
        spatial_cutoff: float | None = 0.01,
        w_t_min: float = 0.1,
        accelerated: bool = True,
    ):
        self._location = location
        self._tau_t = tau_t
        self._tau_s = tau_s
        self._spatial_cutoff = spatial_cutoff
        self._w_t_min = w_t_min
        self._t = 0
        self._accelerated = accelerated

    def w_t(self) -> float:
        """Compute the time-dependent weight at the current step.

        The weight is computed as `exp(-t / lam)`, where `t` is the number of
        steps since the kernel was created, and `lam` is equal to `tau_t / log(2)`.

        Returns:
            The weight, bounded to [0, 1].
        """
        return np.exp(-self._t / (self._tau_t / float(np.log(2))))

    def w_s(self, points: np.ndarray) -> np.ndarray:
        """Compute the distance-dependent weight.

        The weight is computed as `exp(-z / lam)`, where `z` is the distance
        between the kernel's center and the given point(s), and `lam` is equal
        to `tau_s / log(2)`.

        Args:
            points: An (num_points, 3) array of points.

        Returns:
            The weight, bounded to [0, 1]. Has shape (num_points,).
        """
        spatial_cutoff = 0.0 if self._spatial_cutoff is None else self._spatial_cutoff
        if self._accelerated:
            return compute_spatial_weight(
                points, self._location, self._tau_s, spatial_cutoff
            )
        return compute_spatial_weight_old(
            points, self._location, self._tau_s, spatial_cutoff
        )

    def step(self) -> bool:
        """Increment the step counter, and check if the kernel is expired.

        Returns:
            True if the kernel is expired, False otherwise.
        """
        self._t += 1
        return self.w_t() < self._w_t_min

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Compute the time- and distance-dependent weight at a given point.

        Computes the product of the time- and distance-dependent weights. Weights
        are bounded to [0, 1], where values close to 1 indicate the kernel has a
        large influence on the given point(s).

        Args:
            points: An (num_points, 3) array of points.

        Returns:
            The weights, bounded to [0, 1]. Has shape (num_points,).
        """
        assert points.ndim == 2 and points.shape[1] == 3
        return self.w_t() * self.w_s(points)


class DecayKernelFactory:
    def __init__(
        self,
        tau_t: float = 7.50,
        tau_s: float = 0.015,
        spatial_cutoff: float | None = 0.025,
        w_t_min: float = 0.1,
        accelerated: bool = True,
    ):
        self._tau_t = tau_t
        self._tau_s = tau_s
        self._spatial_cutoff = spatial_cutoff
        self._w_t_min = w_t_min
        self._accelerated = accelerated
    def __call__(self, location: np.ndarray) -> DecayKernel:
        return DecayKernel(
            location,
            self._tau_t,
            self._tau_s,
            self._spatial_cutoff,
            self._w_t_min,
            self._accelerated,
        )


class DecayField:
    """Manages a collection of decay kernels."""

    def __init__(
        self,
        kernel_factory_class: type[DecayKernelFactory] = DecayKernelFactory,
        kernel_factory_args: dict[str, Any] | None = None,
    ):
        kernel_factory_args = dict(kernel_factory_args) if kernel_factory_args else {}
        self._kernel_factory = kernel_factory_class(**kernel_factory_args)
        self._kernels: list[DecayKernel] = []

    def reset(self) -> None:
        self._kernels.clear()

    def add(self, location: np.ndarray) -> None:
        """Add a kernel to the field."""
        self._kernels.append(self._kernel_factory(location))

    def step(self) -> None:
        """Step each kernel to increment its counter, and keep only non-expired ones."""
        self._kernels = [k for k in self._kernels if not k.step()]

    def compute_weights(self, points: np.ndarray) -> np.ndarray:
        assert points.ndim == 2 and points.shape[1] == 3
        if not self._kernels:
            return np.zeros(points.shape[0])

        # Stack kernel parameters and compute in batch
        results = np.array([k(points) for k in self._kernels])
        return np.max(results, axis=0)


class ReturnInhibitor:
    def __init__(
        self,
        decay_field_class: type[DecayField] = DecayField,
        decay_field_args: dict[str, Any] | None = None,
    ):
        decay_field_args = dict(decay_field_args) if decay_field_args else {}
        self._decay_field = decay_field_class(**decay_field_args)

    def reset(self) -> None:
        self._decay_field.reset()

    def __call__(
        self,
        visited_location: np.ndarray | None,
        query_locations: np.ndarray,
    ) -> np.ndarray:
        if visited_location is not None:
            self._decay_field.add(visited_location)

        ior_vals = self._decay_field.compute_weights(query_locations)
        self._decay_field.step()
        return ior_vals
