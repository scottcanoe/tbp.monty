# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Utility functions for testing."""

import numpy as np
import quaternion as qt

from tbp.monty.frameworks.models.motor_system_state import MotorSystemState


def motor_system_states_equal(
    state1: MotorSystemState,
    state2: MotorSystemState,
) -> bool:
    """Check if two MotorSystemState objects are equal.

    Compares positions and rotations with numerical tolerance to handle
    floating-point precision issues.

    Args:
        state1: First motor system state.
        state2: Second motor system state.

    Returns:
        True if states are equal, False otherwise.
    """
    # Check that both states have the same agent IDs
    if set(state1.keys()) != set(state2.keys()):
        return False

    # Compare each agent's state
    for agent_id in state1.keys():
        agent_state1 = state1[agent_id]
        agent_state2 = state2[agent_id]

        # Compare motor_only_step flag
        if agent_state1.motor_only_step != agent_state2.motor_only_step:
            return False

        # Compare agent position
        if not _positions_equal(agent_state1.position, agent_state2.position):
            return False

        # Compare agent rotation
        if not _rotations_equal(agent_state1.rotation, agent_state2.rotation):
            return False

        # Check that both agents have the same sensor IDs
        if set(agent_state1.sensors.keys()) != set(agent_state2.sensors.keys()):
            return False

        # Compare each sensor's state
        for sensor_id in agent_state1.sensors.keys():
            sensor_state1 = agent_state1.sensors[sensor_id]
            sensor_state2 = agent_state2.sensors[sensor_id]

            # Compare sensor position
            if not _positions_equal(sensor_state1.position, sensor_state2.position):
                return False

            # Compare sensor rotation
            if not _rotations_equal(sensor_state1.rotation, sensor_state2.rotation):
                return False

    return True


def _positions_equal(pos1, pos2) -> bool:
    """Compare two positions with tolerance.

    Handles magnum.Vector3, numpy arrays, tuples, and lists.

    Args:
        pos1: First position.
        pos2: Second position.

    Returns:
        True if positions are equal within tolerance.
    """
    # Convert to numpy arrays for comparison
    pos1_array = np.asarray(pos1)
    pos2_array = np.asarray(pos2)

    return np.allclose(pos1_array, pos2_array, atol=1e-6, rtol=1e-5)


def _rotations_equal(rot1, rot2) -> bool:
    """Compare two rotations with tolerance.

    Handles quaternion.quaternion objects and numpy arrays.
    Note: Quaternions q and -q represent the same rotation, so we check both.

    Args:
        rot1: First rotation.
        rot2: Second rotation.

    Returns:
        True if rotations are equal within tolerance.
    """
    # Convert to numpy arrays for comparison
    # quaternion.quaternion can be converted to [w, x, y, z] format
    if hasattr(rot1, "__array__"):
        rot1_array = qt.as_float_array(rot1)
    else:
        rot1_array = np.asarray(rot1)

    if hasattr(rot2, "__array__"):
        rot2_array = qt.as_float_array(rot2)
    else:
        rot2_array = np.asarray(rot2)

    # Check if quaternions are equal or negatives (both represent same rotation)
    atol = 1e-6
    return np.allclose(rot1_array, rot2_array, atol=atol) or np.allclose(
        rot1_array, -rot2_array, atol=atol
    )
