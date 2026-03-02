# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RSL-RL is a GPU-accelerated reinforcement learning library for robotics (PyTorch + TensorDict). Package name on PyPI: `rsl-rl-lib`. Python 3.9+.

## Common Commands

```bash
# Install for development
pip install -e .

# Run all tests
pytest tests/ -v --tb=short

# Run a single test file
pytest tests/algorithms/test_ppo.py -v

# Run a single test
pytest tests/algorithms/test_ppo.py::TestGAEComputation::test_gae_returns_hand_computed -v

# Lint and format (runs ruff, codespell, license header insertion, etc.)
pre-commit run --all-files
```

## Code Style

- Line length: 120 characters
- Google-style docstrings
- Linting/formatting: ruff (configured in `ruff.toml`)
- All `.py` files must have the BSD-3-Clause license header (auto-inserted by pre-commit via `.github/LICENSE_HEADER.txt`)
- Import order: future → stdlib → third-party → first-party → local. Note: `numpy`, `torch`, `tensordict`, `warp`, `typing_extensions`, and `git` are treated as extra standard libraries in isort config (not third-party)
- Type checking: Pyright in basic mode

## Architecture

The library has 8 subpackages under `rsl_rl/`:

- **algorithms/**: RL algorithms — `PPO` (Proximal Policy Optimization with GAE) and `Distillation` (student-teacher)
- **runners/**: Training loop orchestrators — `OnPolicyRunner` (main training loop, multi-GPU support, checkpointing) and `DistillationRunner`
- **models/**: Complete actor-critic models — `MLPModel`, `CNNModel`, `RNNModel`. Each composes modules from `modules/`
- **modules/**: Neural network building blocks — `MLP`, `CNN`, `RNN`, distribution heads (`GaussianDistribution`, `HeteroscedasticGaussianDistribution`), normalization layers (`EmpiricalNormalization`)
- **storage/**: `RolloutStorage` manages trajectory collection using a `Transition` dataclass; provides mini-batch generators for learning
- **env/**: `VecEnv` abstract base class defining the vectorized environment interface
- **extensions/**: Optional add-ons — Random Network Distillation (RND) for exploration, symmetry-based learning
- **utils/**: Logging (stdout, W&B, Neptune), callable/config resolution, trajectory padding utilities

**Key patterns:**
- Observations use TensorDict with named "observation groups" (e.g., `"policy"` for actor, `"privileged"` for critic-only info)
- Runners construct algorithms from config dicts using factory/resolver functions
- Models expose `act()` and `evaluate()` interfaces consumed by algorithms
