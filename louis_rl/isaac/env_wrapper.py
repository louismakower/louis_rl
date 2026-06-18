from __future__ import annotations
import torch
from gymnasium.spaces import Dict as DictSpace

from louis_rl.vec_env import SpaceInfo
from .terminal_obs_env import ReturnTerminalManagerBasedRLEnv


def _convert_space(space):
    """Convert Isaac obs space to VecEnv format, fixing (num_envs, obs_dim) → (obs_dim,)."""
    if isinstance(space, DictSpace):
        return {k: _convert_space(v) for k, v in space.spaces.items()}
    return SpaceInfo(shape=space.shape[1:])


class IsaacEnvWrapper:
    """Adapts a ReturnTerminalManagerBasedRLEnv to the VecEnv protocol.

    Isaac obs spaces report shape (num_envs, obs_dim); this wrapper corrects
    observation_space to (obs_dim,) per the VecEnv convention.

    Episode-phase randomisation is the caller's responsibility before learn():
        env.unwrapped.episode_length_buf = torch.randint_like(
            env.unwrapped.episode_length_buf,
            high=int(env.unwrapped.max_episode_length),
        )
    """

    def __init__(self, env: ReturnTerminalManagerBasedRLEnv):
        self._env = env

    @property
    def device(self) -> torch.device:
        return self._env.unwrapped.device

    @property
    def num_envs(self) -> int:
        return self._env.unwrapped.num_envs

    @property
    def max_episode_length(self) -> int:
        return self._env.unwrapped.max_episode_length

    @property
    def action_space(self) -> SpaceInfo:
        return SpaceInfo(shape=self._env.unwrapped.single_action_space.shape)

    @property
    def observation_space(self) -> dict[str, SpaceInfo]:
        raw = self._env.unwrapped.observation_space
        return {k: _convert_space(v) for k, v in raw.items()}

    def step(self, action: torch.Tensor):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()
