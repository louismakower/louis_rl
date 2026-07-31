from __future__ import annotations
import torch
from dataclasses import dataclass

from louis_rl.implementations.goal_spec import GoalSpec


def build_hindsight_goals(
        goal_quantity,  # (max_len, num_envs, goal_dim)
        lengths,  # (num_envs,) — valid steps per env
        mode,     # "final" or "future"
    ):
    """Sample hindsight goals from a trajectory.

    "final": every step uses the episode's final state as goal.
    "future": each step samples uniformly from [t, episode_end].
    """
    max_len, num_envs, _ = goal_quantity.shape
    ep_end = (lengths - 1).unsqueeze(0).expand(max_len, num_envs)
    step_idx = torch.arange(max_len, device=goal_quantity.device).unsqueeze(1).expand(max_len, num_envs)
    env_idx = torch.arange(num_envs, device=goal_quantity.device).unsqueeze(0).expand(max_len, num_envs)

    if mode == "final":
        return goal_quantity[ep_end, env_idx]
    elif mode == "future":
        range_size = (ep_end - step_idx + 1).clamp(min=1)
        offsets = (torch.rand(max_len, num_envs, device=goal_quantity.device) * range_size).long()
        return goal_quantity[step_idx + offsets, env_idx]
    else:
        raise ValueError(f"Unknown HER mode: {mode!r}. Expected 'final' or 'future'.")


@dataclass
class HERCfg:
    """Pure configuration. Runtime buffers live on the runner, not in here."""

    mode: str = "future"  # "final" or "future"
    k: int = 1  # relabelled passes per finished episode
    drop_solved_starts: bool = True


def relabel(traj: dict, spec: GoalSpec, cfg: HERCfg) -> dict[str, torch.Tensor]:
    """Rebuild finished episodes against goals they actually achieved.

    The trajectory buffer stores the goal-independent ``context`` rather than an encoded
    observation, so relabelling simply re-encodes against a different goal: there is no
    observation layout to know about here, and the reward comes from the same
    ``spec.reward`` the rollout used.

    traj holds (max_len, num_envs, ...) tensors plus:
      lengths (num_envs,)             valid steps per episode
      valid   (max_len * num_envs,)   flat mask of real steps
    """
    context = traj["context"]
    next_context = traj["next_context"]
    next_achieved = traj["next_achieved"]
    action = traj["action"]
    valid = traj["valid"]

    goal = build_hindsight_goals(next_achieved, traj["lengths"], mode=cfg.mode)

    obs = spec.encode(context, goal)
    next_obs = spec.encode(next_context, goal)

    goal_dim = goal.shape[-1]
    flat_goal = goal.reshape(-1, goal_dim)
    flat_next_achieved = next_achieved.reshape(-1, goal_dim)

    reward = spec.reward(flat_next_achieved, flat_goal)
    success = spec.success(flat_next_achieved, flat_goal)
    distance = spec.distance(flat_next_achieved, flat_goal)

    done = success if spec.terminate_on_success else traj["term"].reshape(-1)

    out = {
        "obs": obs.reshape(-1, obs.shape[-1])[valid],
        "action": action.reshape(-1, action.shape[-1])[valid],
        "reward": reward.reshape(-1, 1)[valid],
        "next_obs": next_obs.reshape(-1, next_obs.shape[-1])[valid],
        "done": done.reshape(-1, 1)[valid],
        "distances": distance.reshape(-1)[valid],
    }
    return out