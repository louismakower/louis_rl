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
        raise NotImplementedError(f"Unknown intrinsic type: {cfg.type}")

    @abstractmethod
    def get_intrinsic_rew(self, obs, update_norm_stats=False):
        pass

    @abstractmethod
    def train_one_step(self, obs):
        pass

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


@dataclass(kw_only=True)
class IntrinsicCfg(ABC):
    type: Literal["rnd", "counts"] = MISSING


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