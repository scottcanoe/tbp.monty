import fnmatch
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union

import dask.array as da
import imageio
import magnum
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import quaternion
import wrapt
import zarr
from configs import CONFIGS
from model_utils import HabitatSceneDataset
from run import main

from tbp.monty.frameworks.experiments import MontyExperiment
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.run import config_to_dict

BufferEncoder.register(magnum.Vector3, lambda v: [v.x, v.y, v.z])

RESULTS_DIR = Path("~/tbp/results/compositional_objects/results").expanduser()
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

MAX_STEPS = 25


class PubSubSystem:
    """
    A pub/sub system for peeking into processes by injecting publishers.

    This system allows you to:
    1. Subscribe to topics to receive messages
    2. Publish messages to topics
    3. Inject publishers into existing code using wrapt decorators
    4. Filter and process messages in real-time

    Note: This system is purely event-driven and does not store messages.
    """

    def __init__(self):
        self._subscribers: Dict[str, Set[Callable]] = defaultdict(set)
        self._lock = threading.Lock()

        # Counters
        self._message_id_counter = 0

    def subscribe(self, topic: str, handler):
        """Subscribe to a topic."""
        with self._lock:
            self._subscribers[topic].add(handler)

    def unsubscribe(self, topic: str, handler):
        """Unsubscribe from a topic."""
        with self._lock:
            self._subscribers[topic].discard(handler)

    def publish(self, msg: dict):
        """Publish a message to a topic."""
        # if not self._enabled:
        # return

        topic = msg.get("topic", "")
        # message_id = self._message_id_counter
        # self._message_id_counter += 1
        # time = time.time()

        with self._lock:
            # Notify subscribers
            if topic in self._subscribers:
                for handler in self._subscribers[topic]:
                    try:
                        handler(msg)
                    except Exception as e:
                        logging.error(f"Error in subscriber {handler} callback: {e}")


# Global pub/sub instance
_pubsub = PubSubSystem()


def get_pubsub() -> PubSubSystem:
    return _pubsub


publish = _pubsub.publish
subscribe = _pubsub.subscribe
unsubscribe = _pubsub.unsubscribe


"""
================================================================================
- Handlers/Subscribers
"""


class Filter:
    def __init__(
        self,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ):
        self.include = list(np.atleast_1d(include).astype(object)) if include else []
        self.exclude = list(np.atleast_1d(exclude).astype(object)) if exclude else []

    def match(self, obj: Any, key: Optional[Callable] = None) -> bool:
        string = key(obj) if key else obj
        if self.include:
            if not any(fnmatch.fnmatch(string, pattern) for pattern in self.include):
                return False
        if self.exclude:
            if any(fnmatch.fnmatch(string, pattern) for pattern in self.exclude):
                return False
        return True

    def filter_iterable(
        self, iterable: Iterable[Any], key: Optional[Callable] = None
    ) -> List[Any]:
        return [item for item in iterable if self.match(item, key)]

    def filter_dict(
        self, dct: Dict[str, Any], key: Optional[Callable] = None
    ) -> Dict[str, Any]:
        return {k: v for k, v in dct.items() if self.match(v, key)}


def maybe_rename_existing(path: os.PathLike) -> None:
    path = Path(path).expanduser()
    if path.exists():
        if path.is_dir():
            old_path = path.with_suffix(".old")
            if old_path.exists():
                shutil.rmtree(old_path)
            path.rename(old_path)
        else:
            old_name = path.stem + "_old" + "".join(path.suffixes)
            old_path = path.parent / old_name
            if old_path.exists():
                old_path.unlink()
            path.rename(old_path)


class ArrayGroupLogger:
    """A callback class for writing array data to disk during an experiment.

    Brought to you by ChatGPT.

    Args:
        base_dir: Path to a directory where each stream will be stored as a .zarr group.
        compressor: Compression codec ID (e.g. 'zlib', 'blosc', 'zstd') or 'default' for none.
    """

    def __init__(
        self,
        path: os.PathLike,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
        compressor: Optional[str] = None,
        adapter: Optional[Callable] = None,
    ):
        self.path = Path(path).expanduser()
        maybe_rename_existing(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.include = list(np.atleast_1d(include).astype(object)) if include else []
        self.exclude = list(np.atleast_1d(exclude).astype(object)) if exclude else []
        self.compressor = zarr.get_codec({"id": compressor}) if compressor else None
        self.adapter = adapter
        self.array_loggers = {}

    def receive(self, msg: dict):
        """Receive an observation dict message from the pub/sub system."""
        data = self.adapter(msg) if self.adapter else msg
        self.write(data)

    def write(self, data: dict):
        data = self.filter(data)
        for name, array in data.items():
            try:
                logger = self.array_loggers[name]
            except KeyError:
                path = self.path / f"{name}.zarr"
                self.array_loggers[name] = ArrayLogger(path, self.compressor)
                logger = self.array_loggers[name]
            logger.write(array)

    def filter(self, data: dict) -> dict:
        if self.include:
            data = {k: v for k, v in data.items() if k in self.include}
        if self.exclude:
            data = {k: v for k, v in data.items() if k not in self.exclude}
        return data


class RawObservationLogger(ArrayGroupLogger):
    def receive(self, data: Any):
        data = self.adapter(data) if self.adapter else data
        flat = {}
        for agent_id in data:
            for sensor_module_id in data[agent_id]:
                for dset in data[agent_id][sensor_module_id]:
                    name = f"{agent_id}.{sensor_module_id}.{dset}"
                    flat[name] = data[agent_id][sensor_module_id][dset]

        self.write(flat)


class ArrayLogger:
    def __init__(
        self,
        path: os.PathLike,
        compressor: Optional[str] = None,
        adapter: Optional[Callable] = None,
    ):
        self.path = path
        maybe_rename_existing(self.path)
        self.compressor = zarr.get_codec({"id": compressor}) if compressor else None
        self.adapter = adapter
        self.dataset = None

    def receive(self, data):
        if self.adapter:
            data = self.adapter(data)
        self.write(data)

    def write(self, array: np.ndarray):
        if self.dataset is None:
            shape = (0,) + array.shape
            chunks = (1,) + array.shape
            self.dataset = zarr.open(
                self.path,
                mode="w",
                shape=shape,
                chunks=chunks,
                dtype=array.dtype,
                compressor=self.compressor,
                # append_dim=0,
            )
        self.dataset.append(array[None, ...])


class ActionLogger:
    def __init__(self, path: os.PathLike, adapter: Optional[Callable] = None):
        self.path = Path(path).expanduser()
        maybe_rename_existing(self.path)
        self.encoder = BufferEncoder()
        self.adapter = adapter

    def receive(self, data: Any):
        action = self.adapter(data) if self.adapter else data
        self.write(action)

    def write(self, action: Any):
        with open(self.path, "a") as f:
            f.write(self.encoder.encode(action) + "\n")


class ProprioceptiveStateLogger:
    def __init__(self, path: os.PathLike, adapter: Optional[Callable] = None):
        self.path = Path(path).expanduser()
        maybe_rename_existing(self.path)
        self.encoder = BufferEncoder()
        self.adapter = adapter

    def receive(self, data: Any):
        state = self.adapter(data) if self.adapter else data
        self.write(state)

    def write(self, state: Any):
        with open(self.path, "a") as f:
            f.write(self.encoder.encode(state) + "\n")


def wrap_method(cls, method_name: str, topic: str) -> None:
    method = getattr(cls, method_name)
    if hasattr(method, "__wrapped__"):
        print(f"Method {method_name} is already wrapped")
        return

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        result = wrapped(*args, **kwargs)
        msg = {
            "topic": topic,
            "wrapped": wrapped,
            "instance": instance,
            "args": args,
            "kwargs": kwargs,
            "result": result,
        }
        publish(msg)
        return result

    setattr(cls, method_name, wrapper(method))


def import_model(object_name: str):
    object_dataset = HabitatSceneDataset(
        Path(os.environ.get("MONTY_DATA", "~/tbp/data")).expanduser()
        / "compositional_objects"
    )
    blender_dir = Path.home() / "Google Drive/My Drive/Blender/compositional_objects"
    object_path = blender_dir / "objects" / f"{object_name}.glb"
    object_dataset.import_model(object_path, replace=True)


def run_experiment(object_name: str):
    config = CONFIGS["dist_agent_1lm"]
    config["eval_dataloader_args"].object_names = [object_name]
    config["logging_config"].run_name = object_name
    config["logging_config"].output_dir = RESULTS_DIR
    config["experiment_args"].max_train_steps = 1
    config["experiment_args"].max_eval_steps = MAX_STEPS
    config["monty_config"].monty_args.num_exploratory_steps = MAX_STEPS
    config = config_to_dict(config)

    # Do all wrapping here.
    dataset_cls = config["dataset_class"]
    wrap_method(dataset_cls, "reset", "dataset.reset")
    wrap_method(dataset_cls, "__getitem__", "dataset.__getitem__")

    # Do all pubsub subscriptions here.
    pubsub = get_pubsub()
    experiment_dir = (
        Path(config["logging_config"]["output_dir"])
        / config["logging_config"]["run_name"]
    )

    # # dataset.__getitem__ -> frame_logger.receive
    raw_obs_logger = RawObservationLogger(experiment_dir / "raw_observations")
    raw_obs_logger.include = [
        "agent_id_0.patch.rgba",
        "agent_id_0.patch.depth",
        "agent_id_0.view_finder.rgba",
        "agent_id_0.view_finder.depth",
    ]
    raw_obs_logger.adapter = lambda msg: msg["result"][0]
    pubsub.subscribe("dataset.__getitem__", raw_obs_logger.receive)

    action_logger = ActionLogger(experiment_dir / "actions.jsonl")
    action_logger.adapter = lambda msg: msg["args"][0]
    pubsub.subscribe("dataset.__getitem__", action_logger.receive)

    proprioceptive_state_logger = ProprioceptiveStateLogger(
        experiment_dir / "proprioceptive_states.jsonl"
    )
    proprioceptive_state_logger.adapter = lambda msg: msg["result"][1]
    pubsub.subscribe("dataset.__getitem__", proprioceptive_state_logger.receive)

    main(CONFIGS, experiments=["dist_agent_1lm"])
    return experiment_dir


def make_gifs(experiment_dir: Path, max_steps: Optional[int] = MAX_STEPS):
    gifs_dir = experiment_dir / "gifs"
    gifs_dir.mkdir(parents=True, exist_ok=True)

    raw_obs_dir = experiment_dir / "raw_observations"
    dataset_paths = list(raw_obs_dir.glob("*.zarr"))

    for path in dataset_paths:
        if "rgba" in path.name:
            print(f"Converting {path} to gif")
            frames = zarr.open(path, mode="r")[:max_steps]
        elif "depth" in path.name:
            raw_frames = zarr.open(path, mode="r")[:max_steps]
            norm = colors.Normalize(vmin=0, vmax=0.5)
            scalar_map = plt.cm.ScalarMappable(norm=norm, cmap="gray_r")
            frames = []
            for i, raw_frame in enumerate(raw_frames):
                frame = scalar_map.to_rgba(raw_frame)
                frame = (frame * 255).astype(np.uint8)
                frames.append(frame)
        else:
            continue

        gif_name = path.name[: -len(".zarr")] + ".gif"
        gif_path = gifs_dir / gif_name
        imageio.mimsave(gif_path, frames, duration=100)


def show_frame(experiment_dir: Path):
    raw_obs_dir = experiment_dir / "raw_observations"
    dataset_path = raw_obs_dir / "agent_id_0.view_finder.rgba.zarr"
    frames = zarr.open(dataset_path, mode="r")[:]
    array = frames[0]
    fig, ax = plt.subplots()
    ax.imshow(array)
    ax.axis("off")
    plt.show()


def get_object_name(default: str):
    """Run the experiment with command line arguments."""
    import argparse

    if sys.argv[0].endswith("ipykernel_launcher.py"):
        return default

    parser = argparse.ArgumentParser(description="Run compositional objects experiment")
    parser.add_argument(
        "object_name",
        type=str,
        nargs="?",
        default="",
        help="Name of object to use in experiment",
    )
    args = parser.parse_args()

    if args.object_name:
        return args.object_name
    return default


if __name__ == "__main__":
    object_name = "015_cylinder_numenta_vert"

    object_name = get_object_name(object_name)
    # print(f"object_name: {object_name}")

    import_model(object_name[4:])
    experiment_dir = run_experiment(object_name)
    make_gifs(experiment_dir)
    show_frame(experiment_dir)
