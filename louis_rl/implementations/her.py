from __future__ import annotations
import torch
from dataclasses import dataclass


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
    k: int = 1

    def __post_init__(self):
        self.policy_obs_dim: int | None = None  # set by SACRunner after env construction

    def get_hindsight_transitions(self, trajectories: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            trajectories: dict of tensors with shape (num_steps, num_envs, element_size)
        Returns:
            dict of tensors with shape (num_hindsight_samples, element_size)
        """
        raise NotImplementedError()