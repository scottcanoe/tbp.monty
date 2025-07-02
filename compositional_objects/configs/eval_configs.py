# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configs for Figure 3: Robust Sensorimotor Inference.

This module defines the following experiments:
 - `dist_agent_1lm`
 - `dist_agent_1lm_noise_all`
 - `dist_agent_1lm_randrot_14`
 - `dist_agent_1lm_randrot_14_noise_all`
 - `dist_agent_1lm_randrot_14_noise_all_color_clamped`

 Experiments use:
 - 77 objects
 - 14 rotations
 - Goal-state-driven/hypothesis-testing policy active
 - A single LM (no voting)

NOTE: random rotation variants use the random object initializer and 14 rotations.
`dist_agent_1lm_randrot_noise` which uses the 5 predefined "random" rotations
is defined in `fig5_rapid_inference_with_voting.py`.
"""

import numpy as np

from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    PatchAndViewMontyConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.simulators.habitat.environment import ObjectConfig

from .common import (
    MIN_EVAL_STEPS,
    PRETRAIN_DIR,
    DefaultDatasetArgs,
    EvalLoggingConfig,
    get_eval_lm_config,
    get_eval_motor_config,
    get_eval_patch_config,
    get_view_finder_config,
)

# - 14 Rotation used during training (cube faces + corners)
TEST_ROTATIONS_14 = get_cube_face_and_corner_views_rotations()

DEBUG_ROTATIONS = TEST_ROTATIONS_14[:1]

dist_agent_1lm = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=str(PRETRAIN_DIR / "dist_agent_1lm/pretrained"),
        n_eval_epochs=len(DEBUG_ROTATIONS),
        max_total_steps=1,
        max_eval_steps=5,
    ),
    logging_config=EvalLoggingConfig(run_name="dist_agent_1lm"),
    monty_config=PatchAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyArgs(min_eval_steps=MIN_EVAL_STEPS),
        sensor_module_configs=dict(
            sensor_module_0=get_eval_patch_config("dist"),
            sensor_module_1=get_view_finder_config(),
        ),
        learning_module_configs=dict(
            learning_module_0=get_eval_lm_config("dist"),
        ),
        motor_system_config=get_eval_motor_config("dist"),
    ),
    # Set up environment.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=DefaultDatasetArgs(),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        # object_names=["tbp_mug", "numenta_mug", "ycb_mug"],
        object_names=[],
        object_init_sampler=PredefinedObjectInitializer(
            positions=[[0.0, 1.5, 0.0]], rotations=DEBUG_ROTATIONS
        ),
    ),
)

CONFIGS = {
    "dist_agent_1lm": dist_agent_1lm,
}
