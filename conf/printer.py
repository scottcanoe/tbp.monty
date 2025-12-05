# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import os
import re
from pathlib import Path

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf

from tbp.monty.frameworks.run_env import setup_env
from tbp.monty.hydra import register_resolvers

MONTY_DIR = Path.home() / "tbp" / "tbp.monty"
CONF_DIR = MONTY_DIR / "conf"
SALIENCE_DIR = CONF_DIR / "experiment" / "salience"


# Matches `${something:...}` where `something` has no `:` or `}` in it
_RESOLVER_INTERP = re.compile(r"\$\{[^}:]+:.*?\}")


def _escape_resolvers_in_str(s: str) -> str:
    # Turn `${oc.env:VAR}` into `\${oc.env:VAR}`
    return _RESOLVER_INTERP.sub(lambda m: "\\" + m.group(0), s)


def _escape_resolvers_in_container(obj):
    """Recursively escape resolver interpolations in a plain Python object."""
    if isinstance(obj, dict):
        return {k: _escape_resolvers_in_container(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_escape_resolvers_in_container(item) for item in obj]
    elif isinstance(obj, str):
        return _escape_resolvers_in_str(obj)
    else:
        return obj


def resolve_nodes_only(cfg: DictConfig) -> dict:
    # Convert to plain Python container without resolving
    raw_container = OmegaConf.to_container(
        cfg, resolve=False, structured_config_mode=None
    )
    # Escape all resolver interpolations in the container
    escaped_container = _escape_resolvers_in_container(raw_container)
    # Return the plain container - YAML serialization will handle it correctly
    return escaped_container


@hydra.main(config_path=".", config_name="experiment", version_base=None)
def printer(cfg: DictConfig):
    # Get the config with resolver interpolations escaped
    cfg_dict = resolve_nodes_only(cfg)
    # Print just the experiment section as YAML
    import yaml

    print(yaml.dump(cfg_dict["experiment"], default_flow_style=False, sort_keys=False))


# def monty_class_resolver(class_name: str) -> str:
#     """Returns the class name as a string instead of the actual class."""
#     return class_name


# def ndarray_resolver(list_or_tuple: list | tuple) -> str:
#     """Returns a string representation of the array instead of actual numpy array."""
#     return f"np.array({list(list_or_tuple)})"

# def numpy_list_eval_resolver(expr_list: list) -> list[float]:
#     # call str() on each item so we can use number literals
#     return str([eval(str(item)) for item in expr_list])  # noqa: S307


# def numpy_pi_resolver(obj: str) -> float:
#     if "np.pi" in obj:
#         return str(np.pi)
#     return obj
#     # return float(obj)

# def ones_resolver(n: int) -> str:
#     """Returns a string representation instead of actual numpy array."""
#     return f"np.ones({n})"


if __name__ == "__main__":
    # setup_env()
    # register_resolvers()
    # Override resolvers that return non-serializable objects
    # OmegaConf.clear_resolver("monty.class")
    # OmegaConf.register_new_resolver("monty.class", monty_class_resolver)

    # OmegaConf.clear_resolver("np.array")
    # OmegaConf.register_new_resolver("np.array", ndarray_resolver)

    # OmegaConf.clear_resolver("np.ones")
    # OmegaConf.register_new_resolver("np.ones", ones_resolver)

    # OmegaConf.clear_resolver("np.list_eval")
    # OmegaConf.register_new_resolver("np.list_eval", numpy_list_eval_resolver)

    # OmegaConf.clear_resolver("np.pi")
    # OmegaConf.register_new_resolver("np.pi", numpy_pi_resolver)

    printer()

