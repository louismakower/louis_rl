from __future__ import annotations
import torch.nn as nn
import torch.optim as optim
import torch

from dataclasses import dataclass, MISSING
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from .networks import build_mlp


class RND:
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
        obs_c = torch.clamp(obs_n, -self.cfg.obs_clip, self.cfg.obs_clip)
        return obs_c

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


@dataclass
class RNDCfg:
    pred_dim: int = MISSING
    target_hidden_layers: list[int] = MISSING
    predictor_hidden_layers: list[int] = MISSING

    lr: float = MISSING
    obs_clip: float = MISSING
    use_frac: float = MISSING
