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
    
    # per-env step count within the current episode, shape (num_envs,)
    ep_counters: torch.Tensor

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


class SnapshotVecEnv(VecEnv, Protocol):
    """
    Capability for archive-based resets (see StableArchive /
    StableCounts.archive.select).
    
    On step, extras dict must also contain "terminal_snapshot" (N, snapshot_dim),
    nan for envs that did NOT reset this step, mirroring "terminal_obs".
    """

    def snapshot_state(self) -> torch.Tensor:
        """(N, snapshot_dim) current full restorable state, for every env."""
        ...

    def restore_state(self, env_ids: torch.Tensor, snapshots: torch.Tensor) -> dict:
        """Mutate sim state for the flagged envs to the given per-row snapshot.

        env_ids: (N,) bool mask of which envs to restore.
        snapshots: (N, snapshot_dim) restore targets (rows for non-flagged envs unused).

        Returns a fresh full obs dict mirroring step()'s obs, so callers can just
        do `next_obs = env.restore_state(...)` and continue as if from step().
        """
        ...


class Logger(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...
    def add_histogram(self, tag: str, values: torch.Tensor, global_step: int) -> None: ...
