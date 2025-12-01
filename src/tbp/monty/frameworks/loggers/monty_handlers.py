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

import abc
import copy
import fnmatch
import json
import logging
import os
from pathlib import Path
from pprint import pformat
from typing import Any, Container, Iterable, Literal

from typing_extensions import override

from tbp.monty.frameworks.actions.actions import ActionJSONEncoder
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.utils.logging_utils import (
    lm_stats_to_dataframe,
    maybe_rename_existing_file,
)

logger = logging.getLogger(__name__)

###
# Template for MontyHandler
###


class MontyHandler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def report_episode(self, **kwargs):
        pass

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractclassmethod
    def log_level(self):
        """Handlers filter information from the data they receive.

        This class method specifies the level they filter at.
        """
        pass


###
# Handler classes
###


class DetailedJSONHandler(MontyHandler):
    """Grab any logs at the DETAILED level and append to a json file."""

    def __init__(
        self,
        detailed_episodes_to_save: Container[int] | Literal["all"] = "all",
        detailed_save_per_episode: bool = False,
        episode_id_parallel: int | None = None,
        detailed_filters: list | None = None,
    ) -> None:
        """Initialize the DetailedJSONHandler.

        Args:
            detailed_episodes_to_save: Container of episodes to save or
                the string ``"all"`` (default) to include every episode.
            detailed_save_per_episode: Whether to save individual episode files or
                consolidate into a single detailed_run_stats.json file.
                Defaults to False.
            episode_id_parallel: Episode id associated with current run,
                used to identify the episode when using run_parallel.
        """
        self.already_renamed = False
        self.detailed_episodes_to_save = detailed_episodes_to_save
        self.detailed_save_per_episode = detailed_save_per_episode
        self.episode_id_parallel = episode_id_parallel

        self.filters = []
        if detailed_filters:
            self.filters.extend(detailed_filters)

    @classmethod
    def log_level(cls):
        return "DETAILED"

    def _should_save_episode(self, global_episode_id: int) -> bool:
        """Check if episode should be saved.

        Returns:
            True if episode should be saved, False otherwise.
        """
        return (
            self.detailed_episodes_to_save == "all"
            or global_episode_id in self.detailed_episodes_to_save
        )

    def get_episode_id(
        self, local_episode, mode: Literal["train", "eval"], **kwargs
    ) -> int:
        """Get global episode id corresponding to a mode-local episode index.

        This function is needed to determine correct episode id when using
        run_parallel.

        Returns:
            global_episode_id: Combined train+eval episode id.
        """
        return (
            kwargs[f"{mode}_episodes_to_total"][local_episode]
            if self.episode_id_parallel is None
            else self.episode_id_parallel
        )

    def get_detailed_stats(
        self,
        data,
        global_episode_id: int,
        local_episode: int,
        mode: Literal["train", "eval"],
    ) -> dict:
        """Get detailed episode stats.

        Returns:
            stats: Episode stats.
        """
        output_data = {}

        basic_stats = data["BASIC"][f"{mode}_stats"][local_episode]
        detailed_pool = data["DETAILED"]
        detailed_stats = detailed_pool.get(local_episode)
        if detailed_stats is None:
            detailed_stats = detailed_pool.get(global_episode_id)

        output_data[global_episode_id] = copy.deepcopy(basic_stats)
        output_data[global_episode_id].update(detailed_stats)

        return output_data

    def report_episode(self, data, output_dir, local_episode, mode="train", **kwargs):
        """Report episode data."""
        global_episode_id = self.get_episode_id(local_episode, mode, **kwargs)

        if not self._should_save_episode(global_episode_id):
            logger.debug(
                "Skipping detailed JSON for episode %s (not requested)",
                global_episode_id,
            )
            return

        stats = self.get_detailed_stats(data, global_episode_id, local_episode, mode)

        for episode_key in stats.keys():
            for filt in self.filters:
                stats[episode_key] = filt.apply(stats[episode_key])

        if self.detailed_save_per_episode:
            self._save_per_episode(output_dir, global_episode_id, stats)
        else:
            self._save_all(global_episode_id, output_dir, stats)

    def _save_per_episode(self, output_dir: str, global_episode_id: int, stats: dict):
        """Save detailed stats for a single episode.

        Args:
            output_dir: Directory where results are written.
            global_episode_id: Combined train+eval episode id used for DETAILED logs.
            stats: Dictionary containing episode stats keyed by global episode id.
        """
        episodes_dir = Path(output_dir) / "detailed_run_stats"
        os.makedirs(episodes_dir, exist_ok=True)

        episode_file = episodes_dir / f"episode_{global_episode_id:06d}.json"
        maybe_rename_existing_file(episode_file)

        with open(episode_file, "w") as f:
            json.dump(
                {global_episode_id: stats[global_episode_id]},
                f,
                cls=BufferEncoder,
            )

        logger.debug(
            "Saved detailed JSON for episode %s to %s",
            global_episode_id,
            str(episode_file),
        )

    def _save_all(self, global_episode_id: int, output_dir: str, stats: dict):
        """Save detailed stats for all episodes."""
        save_stats_path = Path(output_dir) / "detailed_run_stats.json"
        if not self.already_renamed:
            maybe_rename_existing_file(save_stats_path)
            self.already_renamed = True

        with open(save_stats_path, "a") as f:
            json.dump(
                {global_episode_id: stats[global_episode_id]},
                f,
                cls=BufferEncoder,
            )
            f.write(os.linesep)

        logger.debug(
            "Appended detailed stats for episode %s to %s",
            global_episode_id,
            str(save_stats_path),
        )

    def close(self):
        pass

class TestFilter:
    def __init__(self, value: str = "test_value"):
        self.value = value

    # def __call__(self, dct: dict[str, Any]) -> dict[str, Any]:
    #     return {k: v for k, v in dct.items() if v == self.test_value}


class IncludeExcludeFilter:
    """Simple include/exclude filter for strings. Supports glob patterns.

    Designed to operate like the include/exclude arguments for programs like rsync.
    The rules are:
      - If 'include' is given, ONLY items that match include patterns won't be
        filtered out. If `include` is empty or None, then all items will be included.
      - If 'exclude' is given, items that match exclude patterns will be filtered out.
        If `exclude` is empty or None, then no items will be excluded.
      - `include` is evaluated first, then `exclude`.
    """

    def __init__(
        self,
        include: str | Iterable[str] = (),
        exclude: str | Iterable[str] = (),
    ):
        """Initialize the filter.

        Args:
            include: Strings/patterns to include.
            exclude: Strings/patterns to exclude.
        """
        include = [include] if isinstance(include, str) else include
        exclude = [exclude] if isinstance(exclude, str) else exclude
        self._include = set(include)
        self._exclude = set(exclude)

    def match(self, text: str) -> bool:
        """Check if text should be included.

        Returns True if included, False if excluded.
        First matching rule wins.

        If both include and exclude are provided, the include takes precedence.

        Returns:
            True if text should be included, False if excluded.
        """
        if self._include:
            return any(fnmatch.fnmatch(text, pattern) for pattern in self._include)
        if self._exclude:
            return not any(fnmatch.fnmatch(text, pattern) for pattern in self._exclude)
        return True

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in data.items() if self.match(k)}

class RawObservationsFilter:
    """Filter for including/excluding sensor module raw observation data.

    This filter applies an include/exclude filter to each sensor module's
    raw observations dictionaries. For example, raw observations typically have the
    keys "rgba", "depth", and "semantic_3d", "world_coords", etc. but we may only
    want to save the "rgba" data. This filter with `include=["rgba"]` will only save
    the "rgba" data.

    """

    def __init__(
        self,
        include: str | Iterable[str] = (),
        exclude: str | Iterable[str] = (),
    ):
        self._filter = IncludeExcludeFilter(include, exclude)

    def apply(self, episode_data: dict[str, Any]) -> dict[str, Any]:
        sm_ids = [k for k in episode_data.keys() if k.startswith("SM_")]
        for sm_id in sm_ids:
            sm_dict = episode_data[sm_id]
            raw_observations = sm_dict.get("raw_observations", [])
            for i, row in enumerate(raw_observations):
                raw_observations[i] = self._filter.apply(row)
        return episode_data


class BasicCSVStatsHandler(MontyHandler):
    """Grab any logs at the BASIC level and append to train or eval CSV files."""

    @classmethod
    def log_level(cls):
        return "BASIC"

    def __init__(self):
        """Initialize with empty dictionary to keep track of writes per file.

        We only want to include the header the first time we write to a file. This
        keeps track of writes per file so we can format the file properly.
        """
        self.reports_per_file = {}

    @override
    def report_episode(self, data, output_dir, episode, mode="train", **kwargs):
        # episode is ignored when reporting stats to CSV
        # Look for train_stats or eval_stats under BASIC logs
        basic_logs = data["BASIC"]
        mode_key = f"{mode}_stats"
        output_file = Path(output_dir) / f"{mode}_stats.csv"
        stats = basic_logs.get(mode_key, {})
        logger.debug(pformat(stats))

        # Remove file if it existed before to avoid appending to previous results file
        if output_file not in self.reports_per_file:
            self.reports_per_file[output_file] = 0
            maybe_rename_existing_file(output_file)
        else:
            self.reports_per_file[output_file] += 1

        # Format stats for a single episode as a dataframe
        dataframe = lm_stats_to_dataframe(stats)
        # Move most relevant columns to front
        if "most_likely_object" in dataframe:
            top_columns = [
                "primary_performance",
                "stepwise_performance",
                "num_steps",
                "rotation_error",
                "result",
                "most_likely_object",
                "primary_target_object",
                "stepwise_target_object",
                "highest_evidence",
                "time",
                "symmetry_evidence",
                "monty_steps",
                "monty_matching_steps",
                "individual_ts_performance",
                "individual_ts_reached_at_step",
                "primary_target_position",
                "primary_target_rotation_euler",
                "most_likely_rotation",
            ]
        else:
            top_columns = [
                "primary_performance",
                "stepwise_performance",
                "num_steps",
                "rotation_error",
                "result",
                "primary_target_object",
                "stepwise_target_object",
                "time",
                "symmetry_evidence",
                "monty_steps",
                "monty_matching_steps",
                "primary_target_position",
                "primary_target_rotation_euler",
            ]
        dataframe = self.move_columns_to_front(
            dataframe,
            top_columns,
        )

        # Only include header first time you write to this file
        header = self.reports_per_file[output_file] < 1
        dataframe.to_csv(output_file, mode="a", header=header)

    def move_columns_to_front(self, df, columns):
        for c_key in reversed(columns):
            df.insert(0, c_key, df.pop(c_key))
        return df

    def close(self):
        pass


class ReproduceEpisodeHandler(MontyHandler):
    @classmethod
    def log_level(cls):
        return "BASIC"

    @override
    def report_episode(self, data, output_dir, episode, mode="train", **kwargs):
        # Set up data directory with reproducibility info for each episode
        if not hasattr(self, "data_dir"):
            self.data_dir = os.path.join(output_dir, "reproduce_episode_data")
            os.makedirs(self.data_dir, exist_ok=True)

        # TODO: store a pointer to the training model
        # something like if train_epochs == 0:
        #   use model_name_or_path
        # else:
        #   get checkpoint of most up to date model

        # Write data to action file
        action_file = f"{mode}_episode_{episode}_actions.jsonl"
        action_file_path = os.path.join(self.data_dir, action_file)
        actions = data["BASIC"][f"{mode}_actions"][episode]
        with open(action_file_path, "w") as f:
            f.writelines(
                f"{json.dumps(action[0], cls=ActionJSONEncoder)}\n"
                for action in actions
            )

        # Write data to object params / targets file
        object_file = f"{mode}_episode_{episode}_target.txt"
        object_file_path = os.path.join(self.data_dir, object_file)
        target = data["BASIC"][f"{mode}_targets"][episode]
        with open(object_file_path, "w") as f:
            json.dump(target, f, cls=BufferEncoder)

    def close(self):
        pass
