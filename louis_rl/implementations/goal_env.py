from __future__ import annotations
from abc import abstractmethod

import torch
from torch.utils.tensorboard import SummaryWriter

from louis_rl.implementations.goal_spec import GoalSpec
from louis_rl.utils.grid import CellGrid
from louis_rl.vec_env import SpaceInfo, VecEnv


class GoalSource:
    """Where an episode's goal comes from. Sampling is the only thing a source must do"""

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """(n, goal_dim) goals for n envs that are about to start an episode."""


class GoalVecEnv:
    """Makes a plain env goal-conditioned without the simulator knowing about goals """

    def __init__(
            self,
            env: VecEnv,
            spec: GoalSpec,
            source: GoalSource,
            log_dir: str | None = None,
            grid: CellGrid | None = None,
            log_interval: int = 20,
            map_interval: int = 2000,
    ):
        self._env = env
        self.spec = spec
        self.source = source

        key = spec.desired_key
        self.goal_key = key[0]

        device, num_envs = env.device, env.num_envs
        self.goals = torch.zeros(num_envs, spec.goal_dim, device=device)

        # diagnostics
        self.grid = grid
        self.log_interval = log_interval
        self.map_interval = map_interval
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
        self.global_step = 0
        self._steps = 0
        self._in_goal = torch.zeros((), device=device)
        self._dist = torch.zeros((), device=device)
        self._ep_reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._ep_success = torch.zeros((), device=device)
        self._ep_count = torch.zeros((), device=device)
        self._goal_cells = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._cell_success = torch.zeros(grid.n_cells, device=device)
        self._cell_attempts = torch.zeros(grid.n_cells, device=device)

    @property
    def observation_space(self) -> dict:
        space = dict(self._env.observation_space)
        space[self.goal_key] = SpaceInfo(shape=(self.spec.goal_dim,))
        return space

    def reset(self):
        obs, extras = self._env.reset()
        all_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._ep_reached[:] = False
        self._resample(all_envs)
        obs[self.goal_key] = self.goals.clone()
        return obs, extras

    def step(self, action: torch.Tensor):
        obs, _env_rew, term, timeout, extras = self._env.step(action)
        resetted = term | timeout

        # get achieved from terminal obs
        achieved = self.spec.achieved(obs)
        terminal_obs = extras.get("terminal_obs")
        achieved = torch.where(
            resetted.unsqueeze(-1), self.spec.achieved(terminal_obs), achieved
        )

        rew = self.spec.reward(achieved, self.goals)
        success = self.spec.success(achieved, self.goals)
        self._record(success, achieved, resetted)

        # the goal an episode was pursuing belongs to its terminal obs, not to its
        # successor, so this has to be written before the goals are resampled
        nan = torch.full_like(self.goals, float("nan"))
        terminal_obs[self.goal_key] = torch.where(resetted.unsqueeze(-1), self.goals, nan)

        self._resample(resetted)
        obs[self.goal_key] = self.goals.clone()

        return obs, rew, term, timeout, extras

    def __getattr__(self, name):
        # forward everything from the env
        return getattr(self._env, name)

    def _resample(self, env_ids: torch.Tensor):
        n = int(env_ids.sum())
        if n == 0:
            return
        self.goals[env_ids] = self.source.sample(n)
        self._goal_cells[env_ids] = self.grid.to_flat(self.goals[env_ids])

    @torch.no_grad()
    def _record(self, success: torch.Tensor, achieved: torch.Tensor, resetted: torch.Tensor):
        if self.writer is None:
            return

        self.global_step += self.num_envs
        self._steps += 1
        self._in_goal += success.float().mean()
        self._dist += self.spec.distance(achieved, self.goals).mean()
        self._ep_reached |= success

        if resetted.any():
            reached = self._ep_reached[resetted].float()
            self._ep_success += reached.sum()
            self._ep_count += reached.numel()
            if self.grid is not None:
                cells = self._goal_cells[resetted]
                self._cell_success.index_add_(0, cells, reached)
                self._cell_attempts.index_add_(0, cells, torch.ones_like(reached))
            self._ep_reached[resetted] = False

        if self._steps % self.log_interval == 0:
            self.writer.add_scalar("reach/in_goal", self._in_goal / self._steps, self.global_step)
            self.writer.add_scalar("reach/dist_to_goal", self._dist / self._steps, self.global_step)
            if self._ep_count > 0:
                self.writer.add_scalar(
                    "reach/ep_success", self._ep_success / self._ep_count, self.global_step
                )
            self._in_goal.zero_()
            self._dist.zero_()
            self._ep_success.zero_()
            self._ep_count.zero_()
            self._steps = 0

        if self.grid is not None and self.global_step % (self.map_interval * self.num_envs) == 0:
            self._log_success_map()

    def _log_success_map(self):
        attempts = self._cell_attempts
        rate = torch.where(
            attempts > 0, self._cell_success / attempts.clamp(min=1), torch.zeros_like(attempts)
        )
        self.writer.add_scalar("reach/cells_attempted", (attempts > 0).sum(), self.global_step)
        if len(self.grid.shape) == 2:
            # deliberately the same layout as ExplorerRunner's explore/coverage_map, so
            # the two images can be read against each other: what was ever reached at
            # all, versus what the policy can be sent back to
            self.writer.add_image(
                "reach/success_map", rate.reshape(*self.grid.shape).unsqueeze(0), self.global_step
            )
        self._cell_success.zero_()
        self._cell_attempts.zero_()