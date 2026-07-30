from __future__ import annotations
from dataclasses import dataclass, MISSING
import torch
from torch.utils.tensorboard import SummaryWriter

from louis_rl.vec_env import VecEnv
from louis_rl.utils.grid import CellGrid
from .base_runner import BaseRunner


class ExplorerRunner(BaseRunner):
    """Takes uniform random actions and records the first visit to every grid cell.

    The archive is a sparse, append-only list: a cell is written once, on the step some
    env first lands in it, and never touched again. Capacity is the cell count, so it
    cannot overflow.
    """

    def __init__(self, env: VecEnv, cfg: ExplorerRunnerCfg, log_dir: str):
        super().__init__(log_dir=log_dir)
        self._env = env
        self.cfg = cfg
        self.device = env.device
        self.num_envs = env.num_envs
        self.act_dim = env.action_space.shape[0]

        self.grid = CellGrid(cfg.limits, cfg.resolutions, self.device)
        n = self.grid.n_cells
        self.n_bin = self.grid.n_bin

        self.visited = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.buf_feats = torch.zeros(n, self.n_bin, device=self.device)
        self.size = 0

        self.writer = SummaryWriter(log_dir=log_dir)

    def learn(self):
        obs, extras = self._env.reset()
        self._env.randomise_ep_counters()

        for step in range(self.cfg.num_steps):
            actions = torch.rand((self.num_envs, self.act_dim), device=self.device) * 2 - 1
            next_obs, rew, term, timeout, extras = self._env.step(actions)
            resetted = term | timeout
            self.global_step += self.num_envs

            # take the pre-reset feature for envs that reset this step, so the cells
            # that trigger the out-of-range termination are recorded, not discarded
            feats = next_obs["explore"]
            terminal = extras["terminal_obs"]["explore"]
            feats = torch.where(resetted.unsqueeze(-1), terminal, feats)

            n_new = self.observe(feats)

            self.writer.add_scalar("explore/n_cells", self.size, self.global_step)
            self.writer.add_scalar("explore/coverage", self.size / self.grid.n_cells, self.global_step)
            self.writer.add_scalar("explore/new_cells", n_new, self.global_step)

            if (step + 1) % self.cfg.save_interval == 0:
                self._log_coverage_map()
                self.save_checkpoint()

        self._log_coverage_map()
        self.save_checkpoint()

    @torch.no_grad()
    def observe(self, feats: torch.Tensor) -> int:
        """Append every cell being seen for the first time. Returns how many were added.

        feats: (N, F) - the leading n_bin columns are binned, the rest ignored.
        """
        cells = self.grid.to_flat(feats)
        new = ~self.visited[cells]
        if not bool(new.any()):
            return 0

        # several envs can reach the same new cell on the same step; keep the
        # lowest-index row per cell so the stored feature is order-independent
        new_cells, new_feats = cells[new], feats[new]
        uniq, inverse = torch.unique(new_cells, return_inverse=True)
        rows = torch.arange(new_cells.numel(), device=self.device)
        first = torch.full((uniq.numel(),), new_cells.numel(), dtype=torch.long, device=self.device)
        first.scatter_reduce_(0, inverse, rows, reduce="amin", include_self=True)

        lo, hi = self.size, self.size + uniq.numel()
        self.buf_feats[lo:hi] = new_feats[first, : self.n_bin]
        self.visited[uniq] = True
        self.size = hi
        return uniq.numel()

    def _log_coverage_map(self):
        if self.n_bin != 2:
            return
        img = self.visited.reshape(*self.grid.shape).float().unsqueeze(0)
        self.writer.add_image("explore/coverage_map", img, self.global_step)

    def _get_checkpoint(self) -> dict:
        return {
            "feats": self.buf_feats[: self.size].clone(),
            "limits": self.cfg.limits,
            "resolutions": self.cfg.resolutions,
            "shape": self.grid.shape,
            "global_step": self.global_step,
        }

    def _load_checkpoint(self, state: dict):
        feats = state["feats"].to(self.device)
        k = feats.shape[0]
        self.buf_feats[:k] = feats
        # the cell is a pure function of the feature, so it need not be stored
        self.visited[:] = False
        self.visited[self.grid.to_flat(feats)] = True
        self.size = k
        self.global_step = state["global_step"]

    def get_deterministic_action(self, obs):
        return torch.rand((self.num_envs, self.act_dim), device=self.device) * 2 - 1


@dataclass
class ExplorerRunnerCfg:
    experiment_name: str = MISSING

    num_steps: int = MISSING  # control steps to explore for
    save_interval: int = MISSING  # control steps between buffer saves

    limits: list[tuple[float, float]] = MISSING
    resolutions: float | list[float] = MISSING

    algo_name: str = "explore"