# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import unittest
from typing import Any, Callable

import numpy as np
import quaternion
import torch
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.actions.actions import (
    ActionJSONEncoder,
    LookDown,
    LookUp,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.models.buffer import BufferEncoder


class BufferEncoderTest(unittest.TestCase):
    def setUp(self):
        # Define class and subclass with independent encoders.
        class Dummy:
            def __init__(self, data=0):
                self.data = data

        def dummy_encoder(obj: Any) -> Any:
            return obj.data

        class DummyEncoderClass(json.JSONEncoder):
            def default(self, obj: Any) -> Any:
                if isinstance(obj, Dummy):
                    return obj.data
                return super().default(obj)

        class DummySubclass(Dummy):
            pass

        def dummy_subclass_encoder(obj):
            return dict(data=obj.data)

        self.dummy_class = Dummy
        self.dummy_subclass = DummySubclass
        self.dummy_encoder = dummy_encoder
        self.dummy_subclass_encoder = dummy_subclass_encoder
        self.dummy_encoder_class = DummyEncoderClass

        for cls in [
            self.dummy_class,
            self.dummy_subclass,
        ]:
            BufferEncoder.encoders.pop(cls, None)

    def test_base_state(self):
        self.assertTrue(hasattr(BufferEncoder, "encoders"))
        self.assertIsInstance(BufferEncoder.encoders, dict)
        for key, val in BufferEncoder.encoders.items():
            self.assertIsInstance(key, type)
            self.assertIsInstance(val, Callable)

    def test_register(self):
        # Test registering a a function.
        BufferEncoder.register(self.dummy_class, self.dummy_encoder)
        self.assertEqual(BufferEncoder.encoders[self.dummy_class], self.dummy_encoder)

        # Test registering a subclass of JSONEncoder.
        BufferEncoder.encoders.pop(self.dummy_class, None)
        BufferEncoder.register(self.dummy_class, self.dummy_encoder_class)

        # Test attempting to register an invalid encoder.
        with self.assertRaises(ValueError):
            BufferEncoder.register(self.dummy_class, None)

    def test_find_class_and_subclass(self):
        dummy_1 = self.dummy_class(data=0)
        dummy_2 = self.dummy_subclass(data=1)

        # Test (not) finding an encoder for a class that has not been registered.
        self.assertIsNone(BufferEncoder._find(dummy_1))
        self.assertIsNone(BufferEncoder._find(dummy_2))

        # Test finding a newly registered encoder used for class and subclass.
        BufferEncoder.register(self.dummy_class, self.dummy_encoder)
        self.assertEqual(BufferEncoder._find(dummy_1), self.dummy_encoder)
        self.assertEqual(BufferEncoder._find(dummy_2), self.dummy_encoder)

        # Test finding encoder for a subclass after registering one for it.
        BufferEncoder.register(self.dummy_subclass, self.dummy_subclass_encoder)
        self.assertEqual(
            BufferEncoder._find(dummy_2),
            self.dummy_subclass_encoder,
        )
        self.assertNotEqual(
            BufferEncoder._find(dummy_1),
            BufferEncoder._find(dummy_2),
        )

    def test_encode_class_and_subclass(self):
        dummy_1 = self.dummy_class(data=0)
        dummy_2 = self.dummy_subclass(data=1)

        # Test encode, same for class and subclass.
        BufferEncoder.register(self.dummy_class, self.dummy_encoder)
        self.assertEqual("0", json.dumps(dummy_1, cls=BufferEncoder))
        self.assertEqual("1", json.dumps(dummy_2, cls=BufferEncoder))

        # Repeat using subclass of JSONEncoder.
        BufferEncoder.encoders.pop(self.dummy_class, None)
        BufferEncoder.register(self.dummy_class, self.dummy_encoder_class)
        self.assertEqual("0", json.dumps(dummy_1, cls=BufferEncoder))
        self.assertEqual("1", json.dumps(dummy_2, cls=BufferEncoder))

        # Test encoding after setting new encoder for subclass.
        BufferEncoder.register(self.dummy_subclass, self.dummy_subclass_encoder)
        expected = json.dumps(self.dummy_subclass_encoder(dummy_2))
        self.assertEqual(expected, json.dumps(dummy_2, cls=BufferEncoder))

    def test_numpy(self):
        # Test arrays
        array = np.array([0, 1], dtype=int)
        array_string = json.dumps(array.tolist())
        self.assertEqual(array_string, json.dumps(array, cls=BufferEncoder))

        # Test ints
        for val in array:
            val_string = str(val)
            self.assertEqual(val_string, json.dumps(val, cls=BufferEncoder))

        # Test bools
        for val in array.astype(bool):
            val_string = str(val).lower()
            self.assertEqual(val_string, json.dumps(val, cls=BufferEncoder))

    def test_scipy_rotation(self):
        rot = Rotation.from_euler("xyz", [30, 45, 60], degrees=True)
        rot_array = rot.as_euler("xyz", degrees=True)
        rot_string = json.dumps(rot_array.tolist())
        self.assertEqual(rot_string, json.dumps(rot, cls=BufferEncoder))

    def test_torch_tensors(self):
        tensor = torch.tensor([0, 1], dtype=torch.int32)
        tensor_string = json.dumps(tensor.tolist())
        self.assertEqual(tensor_string, json.dumps(tensor, cls=BufferEncoder))

    def test_quaternion(self):
        quat = quaternion.quaternion(0, 1, 0, 0)
        quat_array = quaternion.as_float_array(quat)
        quat_string = json.dumps(quat_array.tolist())
        self.assertEqual(quat_string, json.dumps(quat, cls=BufferEncoder))

    def test_action(self):
        actions = [
            LookDown(agent_id="test", rotation_degrees=47),
            LookUp(agent_id="test", rotation_degrees=77),
            TurnLeft(agent_id="test", rotation_degrees=90),
            TurnRight(agent_id="test", rotation_degrees=90),
        ]
        for obj in actions:
            a = json.dumps(obj, cls=ActionJSONEncoder)
            b = json.dumps(obj, cls=BufferEncoder)
            self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
