# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from tbp.monty.frameworks.run_env import setup_env
from tbp.monty.hydra import register_resolvers


@hydra.main(config_path=".", config_name="experiment", version_base=None)
def printer(cfg: DictConfig):
    # Force resolving of the config to get errors
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg.experiment))


def monty_class_resolver(class_name: str) -> str:
    """Returns the class name as a string instead of the actual class."""
    return class_name


def ndarray_resolver(list_or_tuple: list | tuple) -> str:
    """Returns a string representation of the array instead of actual numpy array."""
    return f"np.array({list(list_or_tuple)})"

def numpy_list_eval_resolver(expr_list: list) -> list[float]:
    # call str() on each item so we can use number literals
    return str([eval(str(item)) for item in expr_list])  # noqa: S307


def numpy_pi_resolver(obj: str) -> float:
    if "np.pi" in obj:
        return str(np.pi)
    return obj
    # return float(obj)

def ones_resolver(n: int) -> str:
    """Returns a string representation instead of actual numpy array."""
    return f"np.ones({n})"


if __name__ == "__main__":
    setup_env()
    register_resolvers()
    # Override resolvers that return non-serializable objects
    OmegaConf.clear_resolver("monty.class")
    OmegaConf.register_new_resolver("monty.class", monty_class_resolver)

    OmegaConf.clear_resolver("np.array")
    OmegaConf.register_new_resolver("np.array", ndarray_resolver)

    OmegaConf.clear_resolver("np.ones")
    OmegaConf.register_new_resolver("np.ones", ones_resolver)

    OmegaConf.clear_resolver("np.list_eval")
    OmegaConf.register_new_resolver("np.list_eval", numpy_list_eval_resolver)

    OmegaConf.clear_resolver("np.pi")
    OmegaConf.register_new_resolver("np.pi", numpy_pi_resolver)

    printer()
