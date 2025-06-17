import argparse
import functools
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np

from compositional_objects.configs import CONFIGS
from tbp.monty.frameworks.experiments import MontyExperiment
from tbp.monty.frameworks.run import (
    config_to_dict,
    create_cmd_parser,
    print_config,
)
from tbp.monty.frameworks.run_env import setup_env


class Runner:
    """
    This class is used to run the experiment.

    run_name:
      - If provided via constructor, overrides the experiment config's `run_name`.
      - If not provided...
         - If started from CLI...
           - If experiment config has a `run_name`, use it. Otherwise, use the
             -e command line argument.
         - If NOT started from CLI...
           - If experiment config has a `run_name`, use it. Otherwise, raise an
             error.

    out_dir: Final output directory, reflecting the logging config's `output_dir`
      and `run_name`.
    """

    # Experiment configs/instance.
    config: dict  # Original config used when creating the runner.
    config_dict: Optional[dict] = None  # Config after preparing it and converting
    exp: Optional[MontyExperiment] = None  # MontyExperiment instance.

    # Output options
    run_name: Optional[str] = None
    out_dir: Optional[Path] = None

    # CLI-related information
    _cli: bool = False
    _cli_experiment: str = ""
    _cli_quiet_habitat_logs: bool = True

    def __init__(
        self,
        config: Dict[str, dict],
        run_name: Optional[str] = None,
        print_config: bool = True,
    ):
        self.config = config
        self.run_name = run_name
        self.print_config = print_config

    @classmethod
    def from_cli(
        cls,
        all_configs: Dict[str, dict],
        run_name: Optional[str] = None,
        print_config: bool = True,
    ) -> "Runner":
        cmd_parser = create_cmd_parser(experiments=list(all_configs.keys()))
        cmd_args = cmd_parser.parse_args()

        if len(cmd_args.experiments) != 1:
            raise ValueError("Exactly one experiment must be specified")
        experiment = cmd_args.experiments[0]

        obj = cls(all_configs[experiment], run_name=run_name, print_config=print_config)
        obj._cli = True
        obj._cli_experiment = experiment
        obj._cli_quiet_habitat_logs = cmd_args.quiet_habitat_logs

        return obj

    def prepare(self):
        """Use this as "main" function when running monty experiments.

        A typical project `run.py` shoud look like this::

            # Load all experiment configurations from local project
            from experiments import CONFIGS
            from tbp.monty.frameworks.run import main

            if __name__ == "__main__":
                main(all_configs=CONFIGS)

        Args:
            all_configs: Dict containing all available experiment configurations.
                Usually each project would have its own list of experiment
                configurations
            experiments: Optional list of experiments to run, used to bypass the
                command line args
        """

        # Setup environment variables.
        setup_env()

        if self._cli_quiet_habitat_logs:
            os.environ["MAGNUM_LOG"] = "quiet"
            os.environ["HABITAT_SIM_LOG"] = "quiet"

        self.config_dict = config_to_dict(self.config)

        # Update run_name and output dir with experiment name
        # NOTE: wandb args are further processed in monty_experiment

        # Figure out run_name, accounting for CLI vs. code-based runs and overrides.
        config_run_name = self.config_dict["logging_config"]["run_name"]
        if self.run_name:
            run_name = self.run_name
        else:
            if self._cli:
                run_name = config_run_name or self._cli_experiment
            else:
                run_name = config_run_name

        # Figure out output directory.
        config_out_dir = self.config_dict["logging_config"]["output_dir"]
        output_dir = os.path.join(config_out_dir, run_name)

        # Finally, set the config's run_name and output_dir. Also create
        # the output directory, and keep these attributes on this instance.
        self.config_dict["logging_config"]["run_name"] = run_name
        self.config_dict["logging_config"]["output_dir"] = output_dir
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # We're not running in parallel, this should always be False
        self.config_dict["logging_config"]["log_parallel_wandb"] = False
        if self.print_config:
            print_config(self.config_dict)

        # Finally, instantiate the experiment.
        self.exp = self.config_dict["experiment_class"](self.config_dict)

    def run(self):
        if self.exp is None:
            self.prepare()

        start_time = time.time()
        with self.exp as exp:
            # TODO: Later will want to evaluate every x episodes or epochs
            # this could probably be solved with just setting the logging frequency
            # Since each trainng loop already does everything that eval does.
            if exp.do_train:
                print("---------training---------")
                exp.train()

            if exp.do_eval:
                print("---------evaluating---------")
                exp.evaluate()

        secs_elapsed = time.time() - start_time
        mins, secs = divmod(secs_elapsed, 60)
        msg = f"Done running '{self.run_name}' in {int(mins)}m {int(secs)}s"
        logging.info(msg)
        print(msg)


# if __name__ == "__main__":
#     """
#     Run from CLI:

#        python runner.py -e dist_agent_1lm

#     Run directly:

#        runner = Runner(CONFIGS["dist_agent_1lm"])
#        runner.run()
#     """

#     # Run from CLI.
#     runner = Runner.from_cli(CONFIGS)
#     runner.run()


class MyClass:
    def __init__(self, val=0):
        self.val = val

    def get_val(self):
        return self.val


a = MyClass()


def wrap_get_val(method):
    @functools.wraps(method)
    def wrapper(self):
        print("wrapper called")
        val = method(self)
        return f"wrapped {val}"

    return wrapper


MyClass.get_val = wrap_get_val(MyClass.get_val)
b = MyClass()

_wrapped_classes = set()
_wrapped_methods = set()

t0 = time.time()


def wrap_method(cls, method_name):
    method = getattr(cls, method_name)
    if method in _wrapped_methods:
        print(f"Method {cls.__name__}.{method_name} already wrapped")
        return

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        print(f"  +  wrapper called for {method_name} at {time.time() - t0:.2f}s")
        return method(self, *args, **kwargs)

    setattr(cls, method_name, wrapper)
    _wrapped_methods.add(getattr(cls, method_name))


def list_methods(cls):
    methods = inspect.getmembers(cls, predicate=inspect.isfunction)
    print("\nMethods of experiment class:")
    for method_name, method in methods:
        print(f"  {method_name}")


_getitem = []


def wrap_dataset__getitem__(cls):
    method = cls.__getitem__

    @functools.wraps(method)
    def wrapper(self, action):
        out = method(self, action)
        _getitem.append(out)
        return out

    cls.__getitem__ = wrapper

    # def __getitem__(self, action: Action):
    #     observation = self.env.step(action)
    #     state = self.env.get_state()
    #     if self.transform is not None:
    #         observation = self.apply_transform(self.transform, observation, state)
    #     return observation, ProprioceptiveState(state) if state else None


runner = Runner(CONFIGS["dist_agent_1lm"], print_config=False)
runner.prepare()
config = runner.config_dict
exp = runner.exp

cls = config["experiment_class"]
# Get all methods of the experiment class
methods = inspect.getmembers(cls, predicate=inspect.isfunction)
print("\nMethods of experiment class:")
for method_name, method in methods:
    wrap_method(cls, method_name)
    # print(f"  {name}")


dataset_cls = config["dataset_class"]
wrap_dataset__getitem__(dataset_cls)

# dataset_methods = inspect.getmembers(dataset_cls, predicate=inspect.isfunction)
# print("\nMethods of dataset class:")
# for method_name, method in dataset_methods:
#     wrap_method(dataset_cls, method_name)


# for method_name in ["pre_epoch", "pre_episode", "post_episode", "post_epoch"]:
#     wrap_method(cls, method_name)
# wrap_method(exp.dataset.__class__, "__getitem__")

runner.run()

images = []
for i, elt in enumerate(_getitem):
    observation, state = elt
    im = observation["agent_id_0"]["view_finder"]["rgba"]
    images.append(im)


imageio.mimsave("images.gif", images, duration=100)
