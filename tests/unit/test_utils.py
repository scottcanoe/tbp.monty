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
from magnum import Vector3

from tbp.monty.frameworks.models.motor_system_state import MotorSystemState


def motor_system_states_equal(
    state_1: MotorSystemState,
    state_2: MotorSystemState,
    atol: float = 1e-6,
) -> bool:
    """Check if two MotorSystemState objects are equal.

    Compares positions and rotations with numerical tolerance to handle
    floating-point precision issues.

    Args:
        state_1: First motor system state.
        state_2: Second motor system state.
        atol: Absolute tolerance for numerical comparison. Defaults to 1e-6.

    Returns:
        True if states are equal, False otherwise.
    """

    def positions_equal(pos_1: Vector3, pos_2: Vector3) -> bool:
        pos_1_array = np.asarray(pos_1)
        pos_2_array = np.asarray(pos_2)
        return np.array_equal(pos_1_array, pos_2_array, atol=atol)

    def rotations_equal(rot_1: qt.quaternion, rot_2: qt.quaternion) -> bool:
        rot_1_array = qt.as_float_array(rot_1)
        rot_2_array = qt.as_float_array(rot_2)
        return np.array_equal(rot_1_array, rot_2_array, atol=atol) or np.array_equal(
            rot_1_array, -rot_2_array, atol=atol
        )

    # Check that both states have the same agent IDs
    if set(state_1.keys()) != set(state_2.keys()):
        return False

    # Compare agent states
    for agent_id in state_1.keys():
        agent_1 = state_1[agent_id]
        agent_2 = state_2[agent_id]

        # Compare the agent' positions and rotations.
        if not positions_equal(agent_1.position, agent_2.position):
            return False
        if not rotations_equal(agent_1.rotation, agent_2.rotation):
            return False

        # Compare their respective sensors.
        if set(agent_1.sensors.keys()) != set(agent_2.sensors.keys()):
            return False
        for sensor_id in agent_1.sensors.keys():
            sensor_1 = agent_1.sensors[sensor_id]
            sensor_2 = agent_2.sensors[sensor_id]
            if not positions_equal(sensor_1.position, sensor_2.position):
                return False
            if not rotations_equal(sensor_1.rotation, sensor_2.rotation):
                return False

    return True
