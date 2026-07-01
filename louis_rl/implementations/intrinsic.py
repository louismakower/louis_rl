from __future__ import annotations
import torch.nn as nn
import torch.optim as optim
import torch

from dataclasses import dataclass, MISSING
from typing import Literal
from abc import ABC, abstractmethod
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from louis_rl.utils.networks import build_mlp
from louis_rl.utils.grid import CellGrid

class IntrinsicModule(ABC):
    @classmethod
    def create(cls, cfg: IntrinsicCfg, device, obs_dim) -> IntrinsicModule:
        if cfg.type == "counts":
            return Counts(cfg, device, obs_dim)
        elif cfg.type == "rnd":
            return RND(cfg, device, obs_dim)
        elif cfg.type == "stable_counts":
            return StableCounts(cfg, device, obs_dim)
        raise NotImplementedError(f"Unknown intrinsic type: {cfg.type}")

    @abstractmethod
    def get_intrinsic_rew(self, obs, update_norm_stats=False):
        pass

    @abstractmethod
    def train_one_step(self, obs):
        pass

    def extra_logs(self) -> dict:
        """Optional extra scalars to log each iteration (name -> value). Default: none."""
        return {}

    def _subsample(self, obs):
        if self.use_frac >= 1.:
            return obs
        use_filter = torch.rand((obs.shape[0],), device=obs.device) < self.use_frac
        return obs[use_filter]


class RND(IntrinsicModule):
    def __init__(self, cfg: RNDCfg, device, obs_dim):
        self.cfg: RNDCfg = cfg
        self.device = device
        self.obs_dim = obs_dim
        self.pred_dim = cfg.pred_dim
        self.use_frac = cfg.use_frac
        self._init_networks()
        self.loss = nn.MSELoss(reduction='none')
        self.optim = optim.Adam(self.predictor.parameters(), lr=self.cfg.lr)  # Adam not AdamW - no weight decay so weights aren't pulled towards 0
        self.obs_norm = RunningMeanStd(insize=(self.obs_dim,)).to(self.device)

    def _init_networks(self):
        self.target = build_mlp(
            sizes=(self.obs_dim, *self.cfg.target_hidden_layers, self.pred_dim),
            device=self.device,
        )
        for param in self.target.parameters():
            param.requires_grad = False

        self.predictor = build_mlp(
            sizes=(self.obs_dim, *self.cfg.predictor_hidden_layers, self.pred_dim),
            device=self.device
        )

    def _preproc_obs(self, obs):
        obs_n = self.obs_norm(obs)
        if self.cfg.obs_clip > 0.0:
            obs_c = torch.clamp(obs_n, -self.cfg.obs_clip, self.cfg.obs_clip)
            return obs_c
        else:
            return obs_n
        

    @torch.no_grad()
    def get_intrinsic_rew(self, obs, update_norm_stats=False):
        self.predictor.eval()
        if update_norm_stats:
            self.obs_norm.train()
        else:
            self.obs_norm.eval()
        obs = self._preproc_obs(obs)
        pred = self.predictor(obs)
        target = self.target(obs)
        return self.loss(pred, target).mean(dim=-1, keepdim=True)  # average over prediction dimension

    def train_one_step(self, obs):
        obs = self._subsample(obs)
        if obs.shape[0] == 0:
            # if use_frac removed all observations, don't try training
            return 0.
        self.predictor.train()
        self.obs_norm.eval()
        obs = self._preproc_obs(obs)
        pred = self.predictor(obs)
        y = self.target(obs)
        loss = self.loss(pred, y).mean(dim=-1, keepdim=True)
        mean_loss = loss.mean()
        
        self.optim.zero_grad()
        mean_loss.backward()
        self.optim.step()

        return loss.detach()


class Counts(IntrinsicModule):
    def __init__(self, cfg: CountsCfg, device, obs_dim):
        self.cfg: CountsCfg = cfg
        self.device = device
        self.obs_dim = obs_dim
        self.use_frac = cfg.use_frac
        self.grid = CellGrid(cfg.limits, cfg.resolutions, device)
        self.counts = torch.ones(self.grid.shape, device=device, dtype=torch.int32)

    def get_intrinsic_rew(self, obs, update_norm_stats=False):
        idx = self.grid.to_index(obs)
        counts = self.counts[idx]
        rew = self._counts_to_rew(counts)
        return rew

    def _counts_to_rew(self, counts):
        return (1 / torch.sqrt(counts)).unsqueeze(-1)  # return (N, 1)

    def train_one_step(self, obs):
        obs = self._subsample(obs)
        if obs.shape[0] == 0:
            # if use_frac removed all observations, don't count anything
            return
        idx = self.grid.to_index(obs)
        # index_put_ with accumulate=True sums duplicate indices, so multiple
        # obs landing in the same cell each increment the count
        ones = torch.ones(obs.shape[0], dtype=self.counts.dtype, device=self.device)
        self.counts.index_put_(idx, ones, accumulate=True)


class StableCounts(Counts):
    """Counts, but a cell only counts / rewards when reached *stably*:
    ``‖obs[:, stable_dims]‖ < stable_threshold`` (e.g. stable_dims = velocity)."""

    def __init__(self, cfg: StableCountsCfg, device, obs_dim):
        super().__init__(cfg, device, obs_dim)
        self.cfg: StableCountsCfg = cfg
        self.stable_dims = torch.tensor(cfg.stable_dims, device=device, dtype=torch.long)
        self.stable_threshold = cfg.stable_threshold
        self.ungated_weight = cfg.ungated_weight

        # self.counts (from Counts) counts stable visits only. visit_counts counts
        # every visit, feeding the optional dense exploration term (ungated_weight).
        self.visit_counts = torch.ones(self.grid.shape, device=device, dtype=torch.int32)

        # per-cell archive of the most-stable binning feature seen; inf == unreached
        self.n_bin = len(self.grid.shape)
        self.best_metric = torch.full((self.grid.n_cells,), float("inf"), device=device)
        self.snapshots = torch.zeros(self.grid.n_cells, self.n_bin, device=device)

    def _stability(self, obs):
        metric = obs[:, self.stable_dims].norm(dim=-1)  # 0 == perfectly stable
        return metric, metric < self.stable_threshold

    def get_intrinsic_rew(self, obs, update_norm_stats=False):
        idx = self.grid.to_index(obs)
        _, stable = self._stability(obs)
        rew = stable.float() / torch.sqrt(self.counts[idx].float())  # gated: reward for stopping somewhere novel
        if self.ungated_weight > 0:
            # small dense term on raw visitation so moving toward unexplored cells pays too
            rew = rew + self.ungated_weight / torch.sqrt(self.visit_counts[idx].float())
        return rew.unsqueeze(-1)

    def train_one_step(self, obs):
        obs = self._subsample(obs)
        if obs.shape[0] == 0:
            return
        # every visit feeds the dense exploration count
        v_idx = self.grid.to_index(obs)
        self.visit_counts.index_put_(v_idx, torch.ones(obs.shape[0], dtype=self.visit_counts.dtype, device=self.device), accumulate=True)

        metric, stable = self._stability(obs)
        if not bool(stable.any()):
            return
        obs, metric = obs[stable], metric[stable]  # count / archive stable visits only
        idx = self.grid.to_index(obs)
        self.counts.index_put_(idx, torch.ones(obs.shape[0], dtype=self.counts.dtype, device=self.device), accumulate=True)
        self._archive(obs, metric)

    def _archive(self, obs, metric):
        # keep the lowest-metric feature per cell; collapse duplicate cells in the
        # batch to their best row first, then overwrite only if we beat the stored one
        cells = self.grid.to_flat(obs)
        n, num = self.grid.n_cells, obs.shape[0]
        batch_best = torch.full((n,), float("inf"), device=self.device)
        batch_best.scatter_reduce_(0, cells, metric, reduce="amin", include_self=True)
        is_winner = metric == batch_best[cells]
        rows = torch.arange(num, device=self.device)
        tie_break = torch.where(is_winner, rows, torch.full_like(rows, num))
        win_row = torch.full((n,), num, dtype=torch.long, device=self.device)
        win_row.scatter_reduce_(0, cells, tie_break, reduce="amin", include_self=True)

        present = (win_row < num).nonzero(as_tuple=True)[0]
        win_row = win_row[present]
        better = metric[win_row] < self.best_metric[present]
        upd_cells, upd_rows = present[better], win_row[better]
        self.best_metric[upd_cells] = metric[upd_rows]
        self.snapshots[upd_cells] = obs[upd_rows, : self.n_bin]

    @property
    def occupied(self):
        return torch.isfinite(self.best_metric)

    @property
    def n_stable(self) -> int:
        return int(self.occupied.sum())

    @property
    def coverage(self) -> float:
        return self.n_stable / self.grid.n_cells  # denominator includes unreachable cells

    def stable_states(self):
        # (M, n_bin) archived features, one per stably-reached cell, for reuse as starts/goals
        return self.snapshots[self.occupied]

    def extra_logs(self) -> dict:
        return {"intrinsic/coverage": self.coverage, "intrinsic/n_stable": self.n_stable}

    def save_stable_states(self, path):
        torch.save(
            {"features": self.stable_states().cpu(), "metric": self.best_metric[self.occupied].cpu()},
            path,
        )


@dataclass(kw_only=True)
class IntrinsicCfg(ABC):
    type: Literal["rnd", "counts", "stable_counts"] = MISSING


@dataclass(kw_only=True)
class RNDCfg(IntrinsicCfg):
    type: Literal["rnd"] = "rnd"
    pred_dim: int = MISSING
    target_hidden_layers: list[int] = MISSING
    predictor_hidden_layers: list[int] = MISSING

    lr: float = MISSING
    obs_clip: float = MISSING
    use_frac: float = MISSING


@dataclass(kw_only=True)
class CountsCfg(IntrinsicCfg):
    type: Literal["counts"] = "counts"
    limits: list[tuple[float]] = MISSING
    resolutions: float | list[float] = MISSING
    use_frac: float = MISSING


@dataclass(kw_only=True)
class StableCountsCfg(CountsCfg):
    type: Literal["stable_counts"] = "stable_counts"
    stable_dims: list[int] = MISSING  # obs indices whose norm defines "stable"
    stable_threshold: float = MISSING  # stable iff ‖obs[:, stable_dims]‖ < this
    ungated_weight: float = 0.0  # >0 adds a dense visitation-novelty term (0 = pure gated)

if __name__ == "__main__":
    cfg = CountsCfg(
        limits=[(0, 1.0), (1, 4)],
        resolutions=[0.2, 1],
        use_frac=1.0,
    )
    counts = Counts(cfg, "cpu", 2)

    # two identical obs in the same batch should each be counted
    obs = torch.tensor([[0.1, 1.5], [0.1, 1.5]])
    counts.train_one_step(obs)
    idx = counts.grid.to_index(obs)
    assert counts.counts[idx][0] == 3, counts.counts[idx]  # started at 1, +2
    print("ok, duplicate-in-batch counting works")