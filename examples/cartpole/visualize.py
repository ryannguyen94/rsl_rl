# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize CartPole training results.

Produces a self-contained HTML page with:

1. Training reward and episode-length curves (from TensorBoard event files).
2. An animated GIF of the trained agent balancing the pole.

All assets are base64-embedded so the HTML file can be viewed standalone.

Usage::

    cd examples/cartpole
    python visualize.py
    # Then serve the output directory and open in a browser, e.g.:
    python -m http.server 8080 --directory logs/cartpole_ppo
"""

from __future__ import annotations

import base64
import copy
import glob
import io
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rsl_rl.runners import OnPolicyRunner

from cartpole_env import CartPoleVecEnv
from train import make_train_cfg

SCRIPT_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(SCRIPT_DIR, "logs", "cartpole_ppo")


# ---------------------------------------------------------------------------
# Part A: Training Curves
# ---------------------------------------------------------------------------

def plot_training_curves() -> str:
    """Read TensorBoard logs and return the training curve plot as a base64 PNG string."""
    ea = EventAccumulator(LOG_DIR)
    ea.Reload()

    available_scalars = ea.Tags().get("scalars", [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Mean reward
    if "Train/mean_reward" in available_scalars:
        events = ea.Scalars("Train/mean_reward")
        steps = [e.step for e in events]
        values = [e.value for e in events]
        axes[0].plot(steps, values)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Mean Reward")
        axes[0].set_title("Training Reward")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].set_title("Train/mean_reward not found")

    # Mean episode length
    if "Train/mean_episode_length" in available_scalars:
        events = ea.Scalars("Train/mean_episode_length")
        steps = [e.step for e in events]
        values = [e.value for e in events]
        axes[1].plot(steps, values, color="tab:orange")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Mean Episode Length")
        axes[1].set_title("Episode Length")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_title("Train/mean_episode_length not found")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    print("Generated training curves plot.")
    return b64


# ---------------------------------------------------------------------------
# Part B: Render Trained Policy as GIF
# ---------------------------------------------------------------------------

def find_latest_checkpoint() -> str:
    """Find the checkpoint with the highest iteration number in LOG_DIR."""
    pattern = os.path.join(LOG_DIR, "model_*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found matching {pattern}")
    # Sort by iteration number embedded in filename
    checkpoints.sort(key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    return checkpoints[-1]


def render_gif() -> str:
    """Load the trained policy, render a CartPole episode, and return a base64 GIF string."""
    checkpoint_path = find_latest_checkpoint()
    print(f"Loading checkpoint: {checkpoint_path}")

    # Build runner with the same config used during training
    env = CartPoleVecEnv(num_envs=1, device="cpu")
    cfg = copy.deepcopy(make_train_cfg())
    runner = OnPolicyRunner(env, cfg, log_dir=None, device="cpu")
    runner.load(checkpoint_path)

    policy = runner.get_inference_policy(device="cpu")

    # Create a single env with rendering
    render_env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs_np, _ = render_env.reset(seed=0)
    obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)  # (1, 4)

    frames: list[Image.Image] = []
    for _ in range(500):
        frame = render_env.render()
        frames.append(Image.fromarray(frame))

        # Get action from policy
        obs_td = env.get_observations()
        # Overwrite with current render-env observation
        obs_td["policy"] = obs
        action = policy(obs_td)

        # Threshold to discrete
        discrete_action = int(action.squeeze().item() > 0)
        obs_np, _, terminated, truncated, _ = render_env.step(discrete_action)
        obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)

        if terminated or truncated:
            break

    render_env.close()

    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=20, loop=0)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    print(f"Generated GIF ({len(frames)} frames).")
    return b64


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_html(curves_b64: str, gif_b64: str) -> str:
    """Build a self-contained HTML page embedding the training curves and GIF."""
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CartPole PPO Training Results</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ text-align: center; }}
  .section {{ margin: 2rem 0; text-align: center; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
</style>
</head>
<body>
<h1>CartPole PPO &mdash; Training Results</h1>

<div class="section">
  <h2>Training Curves</h2>
  <img src="data:image/png;base64,{curves_b64}" alt="Training curves">
</div>

<div class="section">
  <h2>Trained Agent</h2>
  <img src="data:image/gif;base64,{gif_b64}" alt="CartPole agent GIF">
</div>
</body>
</html>
"""


def main() -> None:
    if not os.path.isdir(LOG_DIR):
        print(f"Log directory not found: {LOG_DIR}")
        print("Run train.py first.")
        return

    curves_b64 = plot_training_curves()
    gif_b64 = render_gif()

    html = build_html(curves_b64, gif_b64)
    out_path = os.path.join(LOG_DIR, "results.html")
    with open(out_path, "w") as f:
        f.write(html)

    print(f"\nSaved self-contained HTML to: {out_path}")
    print("To view it, run:")
    print(f"  python -m http.server 8080 --directory {LOG_DIR}")
    print("Then open http://<your-runpod-ip>:8080/results.html in your browser.")


if __name__ == "__main__":
    main()
