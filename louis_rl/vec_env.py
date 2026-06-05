from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import torch


@dataclass
class SpaceInfo:
    shape: tuple[int, ...]


class VecEnv(Protocol):
    @property
    def device(self) -> torch.device: ...
    @property
    def num_envs(self) -> int: ...
    @property
    def max_episode_length(self) -> int: ...  # used by SAC to size the HER trajectory buffer
    @property
    def action_space(self) -> SpaceInfo: ...
    @property
    def observation_space(self) -> dict[str, SpaceInfo]: ...

    def step(self, action: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Returns (obs, rew, term, timeout, extras).

        extras must always contain "terminal_obs": a full (N, ...) obs dict mirroring
        the obs structure. Rows for envs that did NOT reset this step are nan.
        Callers identify reset envs via term | timeout.
        """
        ...

    def reset(self) -> tuple[dict, dict]: ...

    def randomise_ep_counters(self): ...


class Logger(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...
    def add_histogram(self, tag: str, values: torch.Tensor, global_step: int) -> None: ...
