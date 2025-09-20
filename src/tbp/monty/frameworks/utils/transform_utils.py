# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import numpy as np
import quaternion  # noqa: F401 required by numpy-quaternion package
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation


def numpy_to_scipy_quat(quat):
    """Convert from wxyz to xyzw format of quaternions.

    i.e. identity rotation in scipy is (0,0,0,1).

    Args:
        quat: A quaternion in wxyz format

    Returns:
        A quaternion in xyzw format
    """
    new_quat = np.array((quat[1], quat[2], quat[3], quat[0]))

    return new_quat


def scipy_to_numpy_quat(quat: np.ndarray) -> np.quaternion:
    numpy_quat = np.quaternion(quat[3], quat[0], quat[1], quat[2])
    return numpy_quat


def as_scipy_rotation(
    obj: Rotation | quaternion.quaternion | ArrayLike,
    *,
    scalar_first: bool = True,
    axes: str = "xyz",
    degrees: bool = True,
) -> Rotation:
    """Convert a rotation description to a rotation matrix.

    Args:
        obj: The rotation to convert.
        scalar_first: Whether to use scalar-first order. Only used if a 4-element
            sequence is given.
        axes: The axes to use for euler angles. Only used if a 3-element sequence is
            given.
        degrees: Whether to use degrees. Only used if a 3-element sequence is given.

    Returns:
        A scipy.spatial.transform.Rotation instance.

    Raises:
        ValueError: If the argument is array-like but doesn't have the right shape.
    """
    if isinstance(obj, Rotation):
        return obj

    if isinstance(obj, quaternion.quaternion):
        return Rotation.from_quat([obj.x, obj.y, obj.z, obj.w])

    obj = np.asarray(obj)

    # - euler angles
    if obj.shape == (3,):
        return Rotation.from_euler(axes, obj, degrees=degrees)

    # - quaternion
    if obj.shape == (4,):
        if scalar_first:
            return Rotation.from_quat(np.roll(obj, -1))
        return Rotation.from_quat(obj)

    # - 3x3 rotation matrix
    if obj.shape == (3, 3):
        return Rotation.from_matrix(axes, obj, degrees=degrees)

    raise ValueError(f"Invalid rotation description: {obj}")


class RigidTransform:
    """A rigid transform (rotation + translation)."""

    def __init__(
        self,
        translation: ArrayLike,
        rotation: Rotation | quaternion.quaternion | ArrayLike,
    ):
        # cached homogeneous transformation matrix
        self._matrix = None

        # basic
        self.translation = translation
        self.rotation = rotation

    @classmethod
    def from_components(
        cls,
        translation: ArrayLike,
        rotation: quaternion.quaternion | ArrayLike | Rotation,
    ) -> RigidTransform:
        """Implemented for compatibiility with future scipy release.

        Args:
            translation: The translation component.
            rotation: The rotation component.

        Returns:
            A RigidTransform instance.

        Note: this is a convenience method for creating a RigidTransform instance
        from its components. It is not necessary to use this method.
        """
        return cls(translation, rotation)

    @property
    def translation(self) -> np.ndarray:
        return self._translation.copy()

    @translation.setter
    def translation(self, translation: ArrayLike) -> None:
        vec = np.array(translation)
        if vec.shape != (3,):
            raise ValueError(f"Translation must be a 3-element array, got {vec.shape}")
        self._translation = vec
        self._matrix = None

    @property
    def rotation(self) -> Rotation:
        return self._rotation

    @rotation.setter
    def rotation(self, rotation: Rotation | quaternion.quaternion | ArrayLike) -> None:
        self._rotation = as_scipy_rotation(rotation)
        self._matrix = None

    def as_matrix(self) -> np.ndarray:
        """4x4 homogeneous transformation matrix."""
        return self._cached_matrix().copy()

    def apply(self, point: ArrayLike) -> np.ndarray:
        return self(point)

    def inv(self) -> RigidTransform:
        rotation_inv = self.rotation.inv()
        translation_inv = rotation_inv.apply(-self._translation)
        return RigidTransform(translation_inv, rotation_inv)

    def _cached_matrix(self) -> np.ndarray:
        """Returns a a homogeneous transformation matrix, possibly cached."""
        if self._matrix is None:
            matrix = np.eye(4)
            matrix[:3, :3] = self._rotation.as_matrix()
            matrix[:3, 3] = self._translation
            self._matrix = matrix
        return self._matrix

    def __call__(self, point: ArrayLike) -> np.ndarray:
        point = np.asarray(point)
        if point.ndim == 1:
            return self._rotation.apply(point) + self._translation
        else:
            matrix = self._cached_matrix()
            return (matrix[:3, :3] @ point.T).T + self._translation

    def __repr__(self) -> str:
        return (
            f"RigidTransform(translation={self._translation}, "
            f"rotation={self._rotation})"
        )
