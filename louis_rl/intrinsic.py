from __future__ import annotations
import torch.nn as nn
import torch.optim as optim
import torch

from dataclasses import dataclass, MISSING
from typing import Literal
from abc import ABC, abstractmethod
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from .networks import build_mlp

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
        if self.use_frac < 1.:
            n_obs = obs.shape[0]
            use_filter = torch.rand((n_obs,), device=obs.device) < self.use_frac
            obs = obs[use_filter]
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
        self.counts, self.upper_lims = self._init_counts_lims()

    def get_intrinsic_rew(self, obs, update_norm_stats=False):
        idx = self._obs_to_idx(obs)
        counts = self.counts[idx]
        rew = self._counts_to_rew(counts)
        return rew
    
    def _counts_to_rew(self, counts):
        return (1 / counts).unsqueeze(-1)  # return (N, 1)
    
    def _obs_to_idx(self, obs):
        # obs is (N, obs_dim)
        indices = []
        for d, boundaries in enumerate(self.upper_lims):
            idx = torch.bucketize(obs[:, d].contiguous(), boundaries)
            idx = idx.clamp(0, self.counts.shape[d] - 1)
            indices.append(idx)
        return tuple(indices)


    def _init_counts_lims(self):
        dists = [max - min for min, max in self.cfg.limits]
        assert all(d > 0 for d in dists), "The limits must be (min, max)"
        eps = 1e-9
        if isinstance(self.cfg.resolutions, float):
            r = self.cfg.resolutions
            num_el = tuple(int(d // (r-eps)) for d in dists)
        else:
            num_el = tuple(int(d // (r-eps)) for d, r in zip(dists, self.cfg.resolutions))

        print(num_el)
        counts = torch.ones(size=num_el, device=self.device, dtype=torch.int32)
        mins = [x[0] for x in self.cfg.limits]

        upper_lims = []
        for idx, num_steps in enumerate(num_el):
            res = self.cfg.resolutions[idx]
            x = mins[idx]
            row = []
            for n in range(1, num_steps + 1):
                up = x + n * res
                row.append(up)
            upper_lims.append(row)

        upper_lims_tensors = [
            torch.tensor(row, device=self.device, dtype=torch.float32)
            for row in upper_lims
        ]
        return counts, upper_lims_tensors
    
    def train_one_step(self, obs):
        # TODO: fix this as it currently doesn't double count where two obs are the same
        idx = self._obs_to_idx(obs)
        flat_idx = torch.zeros(obs.shape[0], dtype=torch.long, device=self.device)
        strides = self.counts.stride()
        for d, row_idx in enumerate(idx):
            flat_idx += row_idx * strides[d]
        self.counts[idx] = self.counts[idx] + 1

@dataclass
class IntrinsicCfg(ABC):
    type: Literal["rnd", "counts"]

@dataclass
class RNDCfg(IntrinsicCfg):
    pred_dim: int = MISSING
    target_hidden_layers: list[int] = MISSING
    predictor_hidden_layers: list[int] = MISSING

    lr: float = MISSING
    obs_clip: float = MISSING
    use_frac: float = MISSING

@dataclass
class CountsCfg(IntrinsicCfg):
    limits: list[tuple[float]] = MISSING
    resolutions: float | list[float] = MISSING

if __name__ == "__main__":
    cfg = IntrinsicCfg(
        type="counts",
        limits=[(0, 1.0), (1, 4)],
        resolutions=[0.2, 1],
    )
    counts = Counts(cfg, "cpu", 2)