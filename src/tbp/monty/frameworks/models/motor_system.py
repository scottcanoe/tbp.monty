# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.actions.actions import (
    Action,
    LookDown,
    LookUp,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.models.motor_policies import BasePolicy, MotorPolicy
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)
from tbp.monty.frameworks.models.states import GoalState, State
from tbp.monty.frameworks.utils.transform_utils import (
    RigidTransform,
    as_scipy_rotation,
)


@dataclass
class MotorSystemTelemetry:
    state: MotorSystemState
    driving_goal_state: GoalState | None
    experiment_mode: Literal["train", "eval"] | None
    processed_observations: State | None
    action: Action | None
    policy_id: str | None


class MotorSystem:
    """The basic motor system implementation."""

    def __init__(
        self,
        policy: MotorPolicy,
        agent_id: str = "agent_id_0",
        state: MotorSystemState | None = None,
        save_telemetry: bool = False,
    ) -> None:
        """Initialize the motor system with a motor policy.

        Args:
            policy: The default motor policy to use.
            agent_id: The agent ID of the motor system.
            state: The initial state of the motor system.
                Defaults to None.
            save_telemetry: Whether to save telemetry.
                Defaults to False.
        """
        # TODO: don't default with this... probably want to have on motor system per
        # agent, and policies inherit agent IDs from the motor system.
        self._agent_id = agent_id

        self._default_policy = self._policy = policy
        self._look_at_policy = LookAtPolicy(
            agent_id=self._agent_id,
            sensor_module_id="view_finder",
        )
        self.save_telemetry = save_telemetry

        self.reset(state)

    @property
    def agent_id(self) -> str:
        """Returns the agent ID of the motor system.

        NOTE: this assumes one agent is associated with the motor system.
        When we move to a motor system composed of many motor modules, agent IDs
        will likely be associated with the latter.
        """
        return self._agent_id

    @property
    def last_action(self) -> Action | None:
        """Returns the last action taken by the motor system."""
        return self._last_action

    @property
    def policy(self) -> MotorPolicy:
        """Returns the motor policy."""
        return self._policy

    @property
    def state(self) -> MotorSystemState:
        """Returns the state of the motor system."""
        return self._state

    @state.setter
    def state(self, state: MotorSystemState | None) -> None:
        """Sets the state of the motor system."""
        self._state = state if state else MotorSystemState()

    @property
    def telemetry(self) -> list[MotorSystemTelemetry]:
        """Returns the telemetry of the motor system."""
        return self._telemetry

    def driving_goal_state(self) -> GoalState | None:
        """Returns the driving goal state."""
        return self._driving_goal_state

    def set_driving_goal_state(self, goal_state: GoalState | None) -> None:
        """Sets the driving goal state.

        Args:
            goal_state: The goal state to drive the motor system.
        """
        self._driving_goal_state = goal_state

    def experiment_mode(self) -> Literal["train", "eval"] | None:
        """Returns the experiment mode."""
        return self._experiment_mode

    def set_experiment_mode(self, mode: Literal["train", "eval"] | None) -> None:
        """Sets the experiment mode."""
        self._experiment_mode = mode

    def processed_observations(self) -> State | None:
        """Returns the processed observations."""
        return self._processed_observations

    def set_processed_observations(self, processed_observations: State | None) -> None:
        """Sets the processed observations."""
        self._processed_observations = processed_observations
        self._policy.processed_observations = processed_observations

    def reset(self, state: MotorSystemState | None = None) -> None:
        """Reset the motor system."""
        self._policy = self._default_policy
        self._state = state if state else MotorSystemState()
        self._driving_goal_state = None
        self._experiment_mode = None
        self._processed_observations = None
        self._last_action = None
        self._telemetry = []

    def pre_episode(self) -> None:
        """Pre episode hook."""
        self.reset()
        self._policy.pre_episode()

    def post_episode(self) -> None:
        """Post episode hook."""
        self._policy.post_episode()

    def step(self) -> None:
        """Select a policy, etc.

        This must be called before `__call__()` is used.

        The important thing here is to determine whether the driving goal state
        should be attempted with the data loader's execute_jump_attempt() method.
        If so, we need to set self._policy to the appropriate `InformedPolicy` object
        and then set its `driving_goal_state` attribute. This is what the data loader
        will look for when deciding to use `execute_jump_attempt()`.

        If we don't want to attempt the driving goal state with a jump, we need to
        set self._policy to some other policy but maybe set an attribute like
        `driving_goal_state` but with a different name (or else not have that policy
        inherit from `InformedPolicy`).

        If there is no driving goal state, pick some other policy.
        """
        policy = self._select_policy()
        self._policy = policy
        self._policy.set_experiment_mode(self._experiment_mode)
        if hasattr(self._policy, "set_driving_goal_state"):
            self._policy.set_driving_goal_state(self._driving_goal_state)
        self._policy.processed_observations = self._processed_observations

    def _select_policy(self) -> MotorPolicy:
        """Selects a policy for the motor system.

        Returns:
            The policy to use.
        """
        if self._driving_goal_state:
            if self._driving_goal_state.sender_id == "view_finder":
                return self._look_at_policy
                return self._look_at_policy

        return self._default_policy

    def _post_call(self, action: Action) -> None:
        """Post call hook."""
        if self.save_telemetry:
            self._telemetry.append(
                MotorSystemTelemetry(
                    state=self._state,
                    driving_goal_state=self._driving_goal_state,
                    experiment_mode=self._experiment_mode,
                    processed_observations=self._processed_observations,
                    action=action,
                    policy_id=self._policy.__class__.__name__,
                )
            )

        # Need to keep this in sync with the policy's driving goal state since
        # derive_habitat_goal_state() consumes the goal state.
        # For now, just clear goal states. Figuring out how and when some should
        # persist is unclear to me.
        self._driving_goal_state = None
        self._last_action = action

    def __call__(self) -> Action:
        """Defines the structure for __call__.

        Delegates to the motor policy.

        Returns:
            The action to take.
        """
        action = self._policy(self._state)
        self._post_call(action)
        return action


class LookAtPolicy(BasePolicy):
    """A policy that looks at a target."""

    def __init__(self, agent_id: str, sensor_module_id: str, **kwargs):
        """Initialize the look at policy.

        Args:
            agent_id: The agent ID
            sensor_module_id: The sensor module ID
            kwargs: Additional arguments to pass to the base policy.
        """
        # TODO: The BasePolicy should be refactored. Not all subclasses need the
        # arguments it requires. Here we just add some reasonable values.
        rng = kwargs.pop("rng", np.random.default_rng(42))
        action_sampler_class = kwargs.pop("action_sampler_class", ConstantSampler)
        action_sampler_args = kwargs.pop(
            "action_sampler_args",
            dict(
                actions=[TurnLeft, TurnRight, LookDown, LookUp],
                rotation_degrees=5.0,
            ),
        )
        super().__init__(
            rng=rng,
            action_sampler_class=action_sampler_class,
            action_sampler_args=action_sampler_args,
            agent_id=agent_id,
            switch_frequency=kwargs.pop("switch_frequency", 0.0),
            **kwargs,
        )
        self.sensor_module_id = sensor_module_id
        self.driving_goal_state = None
        self.processed_observations = None  # unused -- here for compatibility

    def get_random_action(self, *args, **kwargs) -> Action:
        """Returns TurnLeft with 0 rotation degrees.

        Reimplemented due to issues with random number generation. And also, should
        all policies be expected to return random actions?
        """
        return TurnLeft(agent_id=self.agent_id, rotation_degrees=0)

    def reset(self) -> None:
        """Reset the look at policy."""
        super().reset()
        self.driving_goal_state = None
        self.processed_observations = None

    def set_driving_goal_state(self, goal_state: GoalState | None) -> None:
        self.driving_goal_state = goal_state

    def dynamic_call(self, state: MotorSystemState) -> tuple[Action, Action]:
        """Return turn left/right and look down/up actions to take.

        Computes two actions -- a yawing action and a pitching action -- that should
        orient the agent and sensor towards the driving goal state. They must be
        applied in the order in which they are returned.

        Args:
            state: The motor system state.

        Returns:
            A tuple of actions, where the first action is one of TurnLeft or TurnRight,
            and the second action is one of LookDown or LookUp.

        """
        # TODO: Remove this once we adhere to a standard format for motor system states.
        state = clean_habitat_motor_system_state(state)

        # Collect necessary agent and sensor pose information.
        # Subscripts: w=world, a=agent, s=sensor.
        agent_pos_w = state[f"{self.agent_id}"]["position"]
        agent_rot_w = as_scipy_rotation(state[f"{self.agent_id}"]["rotation"])
        agent_to_world = RigidTransform(agent_pos_w, agent_rot_w)
        sensor_rot_a = as_scipy_rotation(
            state[f"{self.agent_id}"]["sensors"][f"{self.sensor_module_id}"]["rotation"]
        )

        # Get the target location in world and agent coordinates.
        t_w = np.asarray(self.driving_goal_state.location)
        t_a = agent_to_world.inv()(t_w)

        # Compute the target's azimuth, relative to the agent.
        yaw_a = np.arctan2(t_a[0], -t_a[2])

        # Compute the target's elevation, relative to the agent. Then subtract the
        # sensor's current pitch to get a pitch delta effective for the sensor.
        pitch_a = np.arctan2(t_a[1], np.sqrt(t_a[0] ** 2 + t_a[2] ** 2))
        sensor_pitch_a = sensor_rot_a.as_euler("xyz")[0]
        pitch_s = pitch_a - sensor_pitch_a

        # Create actions to return to the the motor system.
        yaw_degrees = np.degrees(yaw_a)
        if yaw_degrees >= 0:
            turn = TurnLeft(agent_id=self.agent_id, rotation_degrees=-yaw_degrees)
        else:
            turn = TurnRight(agent_id=self.agent_id, rotation_degrees=yaw_degrees)

        pitch_degrees = np.degrees(pitch_s)
        if pitch_degrees >= 0:
            look = LookDown(agent_id=self.agent_id, rotation_degrees=-pitch_degrees)
        else:
            look = LookUp(agent_id=self.agent_id, rotation_degrees=pitch_degrees)

        # For logging purposes only.
        self.driving_goal_state.info["attempted"] = True

        # Drop the reference to the goal state.
        self.driving_goal_state = None

        return turn, look


def clean_habitat_motor_system_state(raw_state: dict) -> MotorSystemState:
    """Clean up a Habitat motor system state dictionaries.

    Function that cleans up Habitat's MotorSystemState to a more usable format.
    For example, a single RGBD camera normally has separate, redundant positions
    and rotations. For example, "patch.depth" and "patch.rgba" are both present
    and contain the same rotation and position data. This function consolidates
    these into a single position and rotation for the sensor. Positions are also
    converted to the more usable numpy arrays (as opposed to magnum.Vector3 objects).

    Args:
        raw_state: The dirty habitat motor system dictionary.

    Returns:
        The cleaned motor system state.

    TODO: This is temporary. We should decide on a standard format for motor
    system states returned by simulators/environments and adhere to it.
    """
    state = MotorSystemState()
    for agent_id, raw_agent_state in raw_state.items():
        pos = raw_agent_state["position"]  # a magnum.Vector3
        rot = raw_agent_state["rotation"]  # a quaternion.quaternion
        agent_state = AgentState(
            {
                "position": np.array([pos.x, pos.y, pos.z]),
                "rotation": rot,
                "sensors": {},
            }
        )
        for sensor_key, raw_sensor_state in raw_agent_state["sensors"].items():
            sensor_id = sensor_key.split(".")[0]
            if sensor_id in agent_state["sensors"]:
                continue
            pos = raw_sensor_state["position"]
            rot = raw_sensor_state["rotation"]
            agent_state["sensors"][sensor_id] = SensorState(
                position=np.array([pos.x, pos.y, pos.z]),
                rotation=rot,
            )
        state[agent_id] = agent_state

    return state
