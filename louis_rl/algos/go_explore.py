from __future__ import annotations
import os
from louis_rl.algos.ppo import PPORunner, PPORunnerCfg
import torch
from dataclasses import dataclass, MISSING

from louis_rl.vec_env import VecEnv
from .base_runner import BaseRunner
from louis_rl.utils.archive import GoExploreArchive

class GoExploreVecEnv(VecEnv):
    def snapshot_state(self, env_ids=None) -> torch.Tensor:
        pass

    def restore_state(self, env_ids, snapshots) -> None:
        pass

    def terminal_snapshot(self, extras) -> torch.Tensor:
        """Pre-reset snapshot for envs that reset this step (nan for the rest)."""
        pass

class GoExploreRunner(BaseRunner):
    def __init__(
            self,
            env: GoExploreVecEnv,
            cfg: GoExploreRunnerCfg,
            log_dir: str,
    ):
        super().__init__(log_dir)
        self._env = env
        self.device = self._env.device
        self.num_envs = self._env.num_envs
        self.cfg: GoExploreRunnerCfg = cfg

        self.act_dim = self._env.action_space.shape[0]

        self.archive = GoExploreArchive(
            limits=self.cfg.archive_limits,
            resolutions=self.cfg.archive_resolutions,
            snapshot_dim=self.cfg.archive_snapshot_dim,
            device=self.device,
        )
        self.policy = PPORunner(
            self._env,
            self.cfg.policy_cfg,
            log_dir=os.path.join(log_dir, "policy"),
        )

        self.explore_counter = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

    def explore_step(self, to_reset):
        num_resetted = int(to_reset.sum())
        if num_resetted > 0:
            selected = self.archive.select(num_resetted)
            if selected is not None:
                reset_snapshots, chosen = selected
                self._env.restore_state(env_ids=to_reset, snapshots=reset_snapshots)
                # cell becomes parent and inherit depth
                self.prev_cell[to_reset] = chosen
                # score == -depth
                self.depth[to_reset] = (-self.archive.best_scores[chosen]).long()
        self.explore_counter[to_reset] = 0

        random_actions = (torch.rand((self.num_envs, self.act_dim), device=self.device) - 0.5) * 2
        next_obs, rew, term, timeout, extras = self._env.step(random_actions)
        resetted = term | timeout

        current = self._env.snapshot_state()
        terminal = self._env.terminal_snapshot(extras)
        snapshot = torch.where(resetted.unsqueeze(-1), terminal, current)

        self.depth += 1
        scores = -self.depth.float()
        cells = self.archive.update(snapshot, scores, parents=self.prev_cell)

        self.prev_cell = cells
        self.explore_counter += 1

        to_reset = resetted | (self.explore_counter >= self.cfg.explore_horizon)
        return to_reset

    def explore(self):
        obs, extras = self._env.reset()
        self._env.randomise_ep_counters()

        # seed the archive with the start states as roots (depth 0, no parent)
        snapshot = self._env.snapshot_state()
        self.prev_cell = self.archive.grid.to_flat(snapshot)
        self.depth = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        roots = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.archive.update(snapshot, torch.zeros(self.num_envs, device=self.device), roots)

        # at start, reset none (env.reset() just called)
        to_reset = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        for _ in range(self.cfg.num_explore_steps):
            to_reset = self.explore_step(to_reset)

    def robustify(self):
        pass

    def learn(self):
        self.explore()
        self.robustify()


@dataclass
class GoExploreRunnerCfg:
    explore_horizon: int = MISSING
    num_explore_steps: int = MISSING

    policy_cfg: PPORunnerCfg = MISSING

    # archive
    archive_limits: list[tuple[float, float]] = MISSING
    archive_resolutions: float | list[float] = MISSING
    archive_snapshot_dim: int = MISSING
