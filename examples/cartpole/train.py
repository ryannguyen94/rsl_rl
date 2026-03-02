# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a PPO agent on CartPole-v1 using rsl_rl.

Usage::

    cd examples/cartpole
    python train.py

Logs and checkpoints are saved to ``logs/cartpole_ppo/``.
"""

from __future__ import annotations

import copy
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from cartpole_env import CartPoleVecEnv


def make_train_cfg() -> dict:
    """Return the training configuration dictionary for CartPole PPO."""
    return {
        "num_steps_per_env": 24,
        "save_interval": 50,
        "run_name": "cartpole_ppo",
        "logger": "tensorboard",
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "PPO",
            "learning_rate": 3e-4,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.01,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [64, 64],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [64, 64],
            "activation": "elu",
        },
    }


def main() -> None:
    num_envs = 64
    num_learning_iterations = 300
    seed = 42

    torch.manual_seed(seed)

    log_dir = os.path.join(os.path.dirname(__file__), "logs", "cartpole_ppo")
    os.makedirs(log_dir, exist_ok=True)

    env = CartPoleVecEnv(num_envs=num_envs, device="cpu")

    # Deep-copy config because construct_algorithm mutates it via .pop("class_name")
    cfg = make_train_cfg()
    runner = OnPolicyRunner(env, copy.deepcopy(cfg), log_dir=log_dir, device="cpu")

    runner.learn(num_learning_iterations=num_learning_iterations)
    print(f"\nTraining complete. Logs and checkpoints saved to: {log_dir}")


if __name__ == "__main__":
    main()
