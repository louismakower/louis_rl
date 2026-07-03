from __future__ import annotations
import torch.nn as nn
import torch.optim as optim
import torch

from dataclasses import dataclass, MISSING
from typing import Callable, Literal
from abc import ABC, abstractmethod
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from louis_rl.utils.networks import build_mlp
from louis_rl.utils.grid import CellGrid
from louis_rl.utils.archive import StableArchive, norm_threshold_stability

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
        """Gradient-based training step. No-op for modules with no learnable parameters, use observe() instead"""
        pass

    def observe(self, obs, snapshot=None):
        """Non-gradient bookkeeping (ie stats updates), called every step"""
        pass

    def extra_logs(self) -> dict:
        """Optional extra scalars to log each iteration (name -> value). Default: none."""
        return {}

    def _subsample_mask(self, n: int) -> torch.Tensor:
        if self.use_frac >= 1.:
            return torch.ones(n, dtype=torch.bool, device=self.device)
        return torch.rand((n,), device=self.device) < self.use_frac


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
        mask = self._subsample_mask(obs.shape[0])
        obs = obs[mask]
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

    def observe(self, obs, snapshot=None):
        mask = self._subsample_mask(obs.shape[0])
        obs = obs[mask]
        if obs.shape[0] == 0:
            # if use_frac removed all observations, don't count anything
            return
        idx = self.grid.to_index(obs)
        # index_put_ with accumulate=True sums duplicate indices, so multiple
        # obs landing in the same cell each increment the count
        ones = torch.ones(obs.shape[0], dtype=self.counts.dtype, device=self.device)
        self.counts.index_put_(idx, ones, accumulate=True)

    def train_one_step(self, obs):
        pass


class StableCounts(IntrinsicModule):
    """Counts, but a cell only counts / rewards when reached *stably*
    Uses a user-provided stability_fn callable. Requires env with snapshots/resets"""

    def __init__(self, cfg: StableCountsCfg, device, obs_dim):
        self.cfg: StableCountsCfg = cfg
        self.device = device
        self.use_frac = cfg.use_frac
        self.ungated_weight = cfg.ungated_weight
        self.archive = StableArchive(
            limits=cfg.limits,
            resolutions=cfg.resolutions,
            snapshot_dim=cfg.snapshot_dim,
            device=device,
            stability_fn=cfg.stability_fn,
            count_exponent=cfg.count_exponent,
        )

    def get_intrinsic_rew(self, obs, update_norm_stats=False):
        cells = self.archive.grid.to_flat(obs[:, : self.archive.grid.n_bin])
        _, stable = self.archive.stability_fn(obs)
        return self.archive.reward(cells, stable, self.ungated_weight)

    def observe(self, obs, snapshot):
        """Update the archive with a batch of (features, snapshots)"""
        mask = self._subsample_mask(obs.shape[0])
        obs, snapshot = obs[mask], snapshot[mask]
        if obs.shape[0] == 0:
            return
        self.archive.observe(obs, snapshot)

    def train_one_step(self, obs):
        """Noting to trian, no-op"""
        pass

    @property
    def coverage(self) -> float:
        return self.archive.coverage

    @property
    def n_stable(self) -> int:
        return self.archive.n_stable

    def stable_states(self):
        # (M, snapshot_dim) archived snapshots, one per stably-reached cell
        return self.archive.stable_snapshots[self.archive.occupied]

    def extra_logs(self) -> dict:
        return {
            "intrinsic/coverage": self.coverage,
            "intrinsic/n_stable": self.n_stable,
            "intrinsic/n_visited": self.archive.n_visited,
            "intrinsic/visited_coverage": self.archive.visited_coverage,
        }

    def save_stable_states(self, path):
        torch.save(
            {"snapshots": self.stable_states().cpu(), "score": self.archive.best_score[self.archive.occupied].cpu()},
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
    stability_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = MISSING
    snapshot_dim: int = MISSING  # width of the env-restorable state passed to observe() - NOT inferred from obs_dim
    count_exponent: float = 1.0  # higher prefers rarely-visited cells when archive.select()-ing reset points
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
    counts.observe(obs)
    idx = counts.grid.to_index(obs)
    assert counts.counts[idx][0] == 3, counts.counts[idx]  # started at 1, +2
    print("ok, duplicate-in-batch counting works")

    # StableCounts: stability gating, dedup-on-tie, and reset sampling
    stable_cfg = StableCountsCfg(
        limits=[(0, 1.0), (1, 4)],
        resolutions=[0.2, 1],
        use_frac=1.0,
        stability_fn=norm_threshold_stability(dims=[1], threshold=0.5),  # dim 1 here is a stand-in "velocity" dim
        snapshot_dim=2,
    )
    stable_counts = StableCounts(stable_cfg, "cpu", 2)

    unstable_obs = torch.tensor([[0.1, 5.0]])  # dim-1 value 5.0 way over the norm threshold
    stable_counts.observe(unstable_obs, unstable_obs)
    assert stable_counts.n_stable == 0, "unstable rows must never be archived"
    rew = stable_counts.get_intrinsic_rew(unstable_obs)
    assert rew.item() == 0.0, "unstable rows must get zero (gated) reward"

    # resets are decoupled from stability: an unstable-only cell is still a valid
    # reset target (visited), even though it's not archived (occupied/stable)
    unstable_cell = stable_counts.archive.grid.to_flat(unstable_obs)
    assert bool(stable_counts.archive.visited[unstable_cell])
    assert not bool(stable_counts.archive.occupied[unstable_cell])
    snaps, cells = stable_counts.archive.select(1)
    assert cells.item() == unstable_cell.item()
    assert torch.allclose(snaps[0], unstable_obs[0])

    # two stable obs landing in the same cell, differing "stability" (dim-1 magnitude);
    # the archive should keep only the more-stable (lower-magnitude) one
    dup_obs = torch.tensor([[0.1, 0.3], [0.1, 0.1]])
    stable_counts.observe(dup_obs, dup_obs)
    assert stable_counts.n_stable == 1, stable_counts.n_stable
    assert torch.allclose(stable_counts.stable_states()[0], torch.tensor([0.1, 0.1])), stable_counts.stable_states()

    # select() now draws from ANY visited cell (stable or not), and is a no-op on an empty archive
    empty_cfg = StableCountsCfg(
        limits=[(0, 1.0)], resolutions=[0.2], use_frac=1.0,
        stability_fn=norm_threshold_stability(dims=[0], threshold=0.0), snapshot_dim=1,
    )
    empty_archive = StableCounts(empty_cfg, "cpu", 1).archive
    assert empty_archive.select(3) is None
    snaps, cells = stable_counts.archive.select(5)
    assert snaps.shape == (5, 2)
    assert bool(stable_counts.archive.visited[cells].all())
    print("ok, StableCounts gating / dedup / select-from-any-visit all check out")
