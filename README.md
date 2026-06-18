# louis_rl

A deep reinforcement learning library built on PyTorch, designed for use with goal-conditioned vectorised environments (e.g. Isaac Lab, simple_envs).

## Algorithms

- **PPO** (`PPORunner`) — A minimally complex Proximal Policy Optimisation with clipped surrogate objective
- **SAC** (`SACRunner`) — Soft Actor-Critic, optionally with HER

Runners share a `BaseRunner` interface exposing `learn()`, `get_deterministic_action()`, and checkpoint save/load.

## Features

- **Intrinsic motivation** — pluggable module supporting:
  - **RND** (Random Network Distillation) — curiosity signal from predictor/target network error
  - **Counts** — discretised state-visit counts with `1/sqrt(n)` reward shaping
- **HER** (Hindsight Experience Replay) — `final` and `future` goal relabelling strategies (SAC only)
- **TensorBoard / Wandb logging** — rewards, losses, advantages, buffer stats, and episode info

## Installation

```bash
pip install -e .
```

Requires `rl_games` and PyTorch.

## Usage

Construct an environment implementing the `VecEnv` interface, build a config dataclass, and call `learn()`:

```python
from louis_rl.algos.sac import SACRunner, SACRunnerCfg

cfg = SACRunnerCfg(
    gamma=0.99,
    alpha_init=0.2,
    # ... fill all MISSING fields
)
runner = SACRunner(env=env, cfg=cfg, log_dir="runs/my_experiment")
runner.learn()
```

## Structure

```
louis_rl/
  algos/
    base_runner.py       # Abstract base class
    ppo.py               # PPO implementation
    sac.py               # SAC implementation
  implementations/
    intrinsic.py         # RND and Counts intrinsic reward modules
    her.py               # Hindsight Experience Replay
  utils/
    networks.py          # MLP builder, Policy and Q network wrappers
    experience.py        # Vectorised replay buffer
    reward_normaliser.py # Reward scaling utilities
  isaac/
    env_wrapper.py       # Isaac Lab environment adapter
    terminal_obs_env.py  # Wrapper for terminal observation handling
  vec_env.py             # VecEnv interface definition
  rl_runner.py           # Top-level training entry point
```