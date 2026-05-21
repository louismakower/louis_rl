from __future__ import annotations
import torch


def build_hindsight_goals(
        obj_pos,  # (max_len, num_envs, 3)
        lengths,  # (num_envs,) — valid steps per env
        mode,     # "final" or "future"
    ):
    """Sample hindsight goals from a trajectory.

    "final": every step uses the episode's final state as goal.
    "future": each step samples uniformly from [t, episode_end].
    """
    max_len, num_envs, _ = obj_pos.shape
    ep_end = (lengths - 1).unsqueeze(0).expand(max_len, num_envs)
    step_idx = torch.arange(max_len, device=obj_pos.device).unsqueeze(1).expand(max_len, num_envs)
    env_idx = torch.arange(num_envs, device=obj_pos.device).unsqueeze(0).expand(max_len, num_envs)

    if mode == "final":
        return obj_pos[ep_end, env_idx]
    elif mode == "future":
        range_size = (ep_end - step_idx + 1).clamp(min=1)
        offsets = (torch.rand(max_len, num_envs, device=obj_pos.device) * range_size).long()
        return obj_pos[step_idx + offsets, env_idx]
    else:
        raise ValueError(f"Unknown HER mode: {mode!r}. Expected 'final' or 'future'.")


class HERCfg:
    def __init__(self):
        self.policy_obs_dim: int | None = None  # set by SACRunner after env construction

    def get_hindsight_transitions(self, trajectories: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            trajectories: dict of tensors with shape (num_steps, num_envs, element_size)
        Returns:
            dict of tensors with shape (num_hindsight_samples, element_size)
        """
        raise NotImplementedError()