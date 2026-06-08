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
        self.pred_shape = self.cfg.pred_shape
        self._init_networks()
        self.loss = nn.MSELoss(reduction='none')
        self.optim = optim.AdamW(self.predictor.parameters(), lr=self.cfg.lr)
        self.obs_norm = RunningMeanStd(insize=(self.obs_dim,)).to(self.device)
    
    def set_train(self):
        self.predictor.train()
        self.obs_norm.train()

    def set_eval(self):
        self.predictor.eval()
        self.obs_norm.eval()

    def _init_networks(self):
        self.target = build_mlp(
            sizes=(self.obs_dim, *self.cfg.target_hidden_layers, self.pred_shape),
            device=self.device,
        )
        for param in self.target.parameters():
            param.requires_grad = False

        self.predictor = build_mlp(
            sizes=(self.obs_dim, *self.cfg.predictor_hidden_layers, self.pred_shape),
            device=self.device
        )

    @torch.no_grad()
    def get_intrinsic_rew(self, obs):
        self.set_eval()
        obs_n = self.obs_norm(obs)
        pred = self.predictor(obs_n)
        target = self.target(obs_n)
        return self.loss(pred, target).mean(dim=-1, keepdim=True)  # average over prediction dimension

    def train_one_step(self, obs):
        self.set_train()
        x_n = self.obs_norm(obs)
        pred = self.predictor(x_n)
        y = self.target(x_n)
        loss = self.loss(pred, y).mean(dim=-1, keepdim=True)
        mean_loss = loss.mean()
        
        self.optim.zero_grad()
        mean_loss.backward()
        self.optim.step()

        return loss.detach()

@dataclass
class RNDCfg:
    pred_shape: int = MISSING
    target_hidden_layers: list[int] = MISSING
    predictor_hidden_layers: list[int] = MISSING

    lr: float = MISSING
