# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, sentinel

import numpy as np
import numpy.typing as npt
import quaternion as qt
from parameterized import parameterized_class

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.models.salience.sensor_module import (
    SalienceSM,
)
from tbp.monty.frameworks.sensors import SensorID

PATCH = 4


class ArrayEqual:
    def __init__(self, arr: npt.ArrayLike):
        self.arr = arr

    def __eq__(self, other: npt.ArrayLike):
        return np.array_equal(self.arr, other)

    def __hash__(self):
        return hash(np.asarray(self.arr).tobytes())


def observation(on_object: npt.NDArray[np.bool_] | None = None) -> SensorObservation:
    """Build an observation whose semantic channel marks the given pixels.

    Args:
        on_object: ``(PATCH, PATCH)`` mask of on-object pixels. Defaults to all.

    Returns:
        An observation with rgba, depth, and a semantic_3d of distinct locations.

    """
    if on_object is None:
        on_object = np.ones((PATCH, PATCH), dtype=bool)
    num_pixels = on_object.size
    # Distinct coordinates per pixel, so goal locations are identifiable.
    locations = np.arange(num_pixels * 3, dtype=float).reshape(num_pixels, 3)
    semantic_3d = np.concatenate(
        [locations, on_object.reshape(num_pixels, 1).astype(float)], axis=1
    )
    return SensorObservation(
        rgba=np.zeros((PATCH, PATCH, 4), dtype=np.uint8),
        depth=np.zeros((PATCH, PATCH)),
        semantic_3d=semantic_3d,
    )


def locations_of(
    data: SensorObservation, on_object: npt.NDArray[np.bool_]
) -> np.ndarray:
    """Return the world locations of the given pixels.

    Returns:
        A ``(num_selected, 3)`` array of locations.

    """
    semantic_3d = data["semantic_3d"]
    return semantic_3d[:, 0:3].reshape(PATCH, PATCH, 3)[on_object]


@parameterized_class(
    ("save_raw_obs", "is_exploring", "should_snapshot"),
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, False),
    ],
)
class SalienceSMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sensor_module = SalienceSM(
            sensor_module_id="test",
            # Real arrays: the module does arithmetic on whatever these return.
            salience_strategy=MagicMock(return_value=np.zeros((PATCH, PATCH))),
            return_inhibitor=MagicMock(return_value=np.zeros(PATCH * PATCH)),
            snapshot_telemetry=MagicMock(),
        )
        self.default_sensor_state = SensorState(
            position=(0, 0, 0),
            rotation=qt.quaternion(1, 0, 0, 0),
        )
        self.state = AgentState(
            sensors={
                SensorID(self.sensor_module.sensor_module_id): self.default_sensor_state
            },
            position=self.default_sensor_state.position,
            rotation=self.default_sensor_state.rotation,
        )
        self.ctx = RuntimeContext(rng=np.random.RandomState())

    def test_step_snapshots_raw_observation_as_needed(self) -> None:
        self.sensor_module._save_raw_obs = self.save_raw_obs  # type: ignore[attr-defined]
        self.sensor_module.is_exploring = self.is_exploring  # type: ignore[attr-defined]
        data = observation()

        self.sensor_module.update_state(self.state)
        self.sensor_module.step(self.ctx, data)

        raw_observation = self.sensor_module._snapshot_telemetry.raw_observation  # type: ignore[attr-defined]
        if not self.should_snapshot:  # type: ignore[attr-defined]
            raw_observation.assert_not_called()
            return

        raw_observation.assert_called_once()
        args, kwargs = raw_observation.call_args
        self.assertEqual(args[0], data)
        self.assertEqual(args[1], self.state.rotation)
        self.assertTrue(np.array_equal(args[2], self.state.position))
        # No segmentation strategy is configured, so the snapshot carries the
        # salience map and the goals, but no segmentation or region telemetry.
        self.assertEqual(sorted(kwargs["info"]), ["goals", "salience_map"])

    def test_step_returns_no_percept(self) -> None:
        self.assertIsNone(self.sensor_module.step(self.ctx, observation()))

    def test_step_proposes_a_goal_for_every_on_object_location(self) -> None:
        on_object = np.zeros((PATCH, PATCH), dtype=bool)
        on_object[0, :2] = True
        data = observation(on_object)
        weighted = np.array([0.25, 0.75])
        self.sensor_module._weight_salience = MagicMock(return_value=weighted)  # type: ignore[method-assign]

        self.sensor_module.step(self.ctx, data)
        goals = self.sensor_module.propose_goals()

        expected_locations = locations_of(data, on_object)
        self.assertEqual(len(goals), 2)
        for goal, location, confidence in zip(goals, expected_locations, weighted):
            np.testing.assert_array_equal(goal.location, location)
            self.assertEqual(goal.confidence, confidence)
            self.assertTrue(goal.use_state)
            self.assertIsNone(goal.morphological_features)
            self.assertIsNone(goal.non_morphological_features)
            self.assertIsNone(goal.goal_tolerances)
            self.assertEqual(goal.sender_id, "test")
            self.assertEqual(goal.sender_type, "SM")

    def test_step_proposes_no_goals_when_nothing_is_on_object(self) -> None:
        data = observation(np.zeros((PATCH, PATCH), dtype=bool))
        self.sensor_module._weight_salience = MagicMock(return_value=np.array([]))  # type: ignore[method-assign]

        self.sensor_module.step(self.ctx, data)

        self.assertEqual(self.sensor_module.propose_goals(), [])

    def test_step_weights_the_salience_of_on_object_pixels(self) -> None:
        on_object = np.zeros((PATCH, PATCH), dtype=bool)
        on_object[0, :2] = True
        data = observation(on_object)
        salience_map = np.arange(PATCH * PATCH, dtype=float).reshape(PATCH, PATCH)
        self.sensor_module._salience_strategy.return_value = salience_map  # type: ignore[attr-defined]
        ior_weights = np.array([0.0, 0.0])
        self.sensor_module._return_inhibitor.return_value = ior_weights  # type: ignore[attr-defined]
        self.sensor_module._weight_salience = MagicMock(  # type: ignore[method-assign]
            return_value=np.zeros(2)
        )

        self.sensor_module.step(self.ctx, data)

        self.sensor_module._salience_strategy.assert_called_once_with(  # type: ignore[attr-defined]
            ctx=self.ctx, rgba=data["rgba"], depth=data["depth"]
        )
        ctx, salience, weights = self.sensor_module._weight_salience.call_args.args
        self.assertEqual(ctx, self.ctx)
        np.testing.assert_array_equal(salience, salience_map[on_object])
        np.testing.assert_array_equal(weights, ior_weights)

    def test_step_inhibits_return_relative_to_the_fixated_location(self) -> None:
        # The centre pixel is on-object, so it is the fixation.
        data = observation()
        self.sensor_module._weight_salience = MagicMock(  # type: ignore[method-assign]
            return_value=np.zeros(PATCH * PATCH)
        )

        self.sensor_module.step(self.ctx, data)

        center, locations = self.sensor_module._return_inhibitor.call_args.args  # type: ignore[attr-defined]
        expected = data["semantic_3d"][:, 0:3].reshape(PATCH, PATCH, 3)
        np.testing.assert_array_equal(center, expected[PATCH // 2, PATCH // 2])
        np.testing.assert_array_equal(locations, expected.reshape(-1, 3))

    def test_step_has_no_fixation_when_the_centre_is_off_object(self) -> None:
        on_object = np.ones((PATCH, PATCH), dtype=bool)
        on_object[PATCH // 2, PATCH // 2] = False
        data = observation(on_object)
        self.sensor_module._weight_salience = MagicMock(  # type: ignore[method-assign]
            return_value=np.zeros(int(on_object.sum()))
        )

        self.sensor_module.step(self.ctx, data)

        center, _ = self.sensor_module._return_inhibitor.call_args.args  # type: ignore[attr-defined]
        self.assertIsNone(center)


class SalienceSMPrivateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sensor_module = SalienceSM(
            sensor_module_id="test",
            # Real arrays: the module does arithmetic on whatever these return.
            salience_strategy=MagicMock(return_value=np.zeros((PATCH, PATCH))),
            return_inhibitor=MagicMock(return_value=np.zeros(PATCH * PATCH)),
            snapshot_telemetry=MagicMock(),
        )
        self.ctx = RuntimeContext(rng=np.random.RandomState())

    def test_normalize_salience_does_clips_uniform_salience_between_0_and_1(
        self,
    ) -> None:
        salience = 2 * np.ones(10)
        normalized = self.sensor_module._normalize_salience(salience)
        np.testing.assert_array_equal(normalized, np.ones(10))

    def test_normalize_salience_normalizes_empty_salience(self) -> None:
        salience = np.array([])
        normalized = self.sensor_module._normalize_salience(salience)
        np.testing.assert_array_equal(normalized, np.array([]))

    def test_weight_salience_decays_randomizes_and_normalizes_salience_in_that_order(
        self,
    ) -> None:
        salience = np.array([1, 2, 3])
        ior_weights = np.array([0.1, 0.2, 0.3])
        self.sensor_module._decay_salience = MagicMock(return_value=sentinel.decayed)  # type: ignore[method-assign]
        self.sensor_module._randomize_salience = MagicMock(  # type: ignore[method-assign]
            return_value=sentinel.randomized
        )
        self.sensor_module._normalize_salience = MagicMock(  # type: ignore[method-assign]
            return_value=sentinel.normalized
        )

        weighted = self.sensor_module._weight_salience(self.ctx, salience, ior_weights)

        self.sensor_module._decay_salience.assert_called_once_with(
            salience, ior_weights
        )
        self.sensor_module._randomize_salience.assert_called_once_with(
            self.ctx, sentinel.decayed
        )
        self.sensor_module._normalize_salience.assert_called_once_with(
            sentinel.randomized
        )
        self.assertEqual(weighted, sentinel.normalized)
