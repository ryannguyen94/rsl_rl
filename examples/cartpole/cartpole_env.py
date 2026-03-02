# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CartPole vectorized environment wrapper for rsl_rl.

Wraps gymnasium's CartPole-v1 into the :class:`rsl_rl.env.VecEnv` interface so it can be
used with :class:`rsl_rl.runners.OnPolicyRunner` and PPO.

CartPole has a discrete action space (0=left, 1=right), but rsl_rl outputs continuous
actions via a Gaussian distribution. We bridge this with a simple threshold:
``action > 0 → push right (1)``, ``action <= 0 → push left (0)``.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv


class CartPoleVecEnv(VecEnv):
    """Vectorized CartPole-v1 environment compatible with rsl_rl."""

    def __init__(self, num_envs: int, device: str = "cpu") -> None:
        self.num_envs = num_envs
        self.num_actions = 1  # single continuous output, thresholded to discrete
        self.max_episode_length = 500
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}

        # Create gymnasium environments
        self._envs = [gym.make("CartPole-v1") for _ in range(num_envs)]

        # Reset all environments and cache initial observations
        self._obs = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
        for i, env in enumerate(self._envs):
            obs, _ = env.reset()
            self._obs[i] = torch.tensor(obs, dtype=torch.float32, device=device)

    def get_observations(self) -> TensorDict:
        """Return the current observations as a TensorDict with a ``"policy"`` key."""
        return TensorDict({"policy": self._obs.clone()}, batch_size=[self.num_envs], device=self.device)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step all environments with the given continuous actions.

        Args:
            actions: Continuous actions of shape ``(num_envs, 1)``.

        Returns:
            Tuple of ``(observations, rewards, dones, extras)``.
        """
        # Threshold continuous actions to discrete: >0 → right (1), <=0 → left (0)
        discrete_actions = (actions.squeeze(-1) > 0).long()

        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        time_outs = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for i, env in enumerate(self._envs):
            obs, reward, terminated, truncated, _ = env.step(discrete_actions[i].item())
            rewards[i] = reward
            self.episode_length_buf[i] += 1

            done = terminated or truncated
            if done:
                # Time-out = truncated but not terminated (hit step limit, not a failure)
                if truncated and not terminated:
                    time_outs[i] = 1.0
                dones[i] = 1.0
                obs, _ = env.reset()
                self.episode_length_buf[i] = 0

            self._obs[i] = torch.tensor(obs, dtype=torch.float32, device=self.device)

        extras = {"time_outs": time_outs}
        return self.get_observations(), rewards, dones, extras
