from __future__ import annotations
import torch.nn as nn
import torch.optim as optim
import torch

from dataclasses import dataclass, MISSING

from .networks import build_mlp


class RND:
    def __init__(self, cfg: RNDCfg, device, obs_dim):
        self.cfg: RNDCfg = cfg
        self.device = device
        self.obs_dim = obs_dim
        self.pred_shape = self.cfg.pred_shape
        self._init_networks()
        self.loss = nn.MSELoss(reduction='none')
        self.recent_loss = None
        self.fresh_loss = False
        self.optim = optim.AdamW(self.predictor.parameters(), lr=self.cfg.lr)
    
    def set_train(self):
        self.predictor.train()

    def set_eval(self):
        self.predictor.eval()

    def _init_networks(self):
        self.target = build_mlp(
            sizes=(self.obs_dim, *self.cfg.target_hidden_layers, self.pred_shape),
            device=self.device,
        )
        for param in self.target.parameters():
            param.requires_grad = False

        self.predictor = build_mlp(
            sizes=(self.obs_dim, self.cfg.predictor_hidden_layers, self.pred_shape),
            device=self.device
        )

    @torch.no_grad()
    def get_intrinsic_rew(self, obs):
        pred = self.predictor(obs)
        target = self.target(obs)
        return self.loss(pred, target).mean(dim=-1, keepdim=True)  # average over prediction dimension

    def train_one_step(self):
        if not self.fresh_loss:
            raise RuntimeError("Loss is stale, please call get_intrinsic_rew first")
        mean_loss = self.recent_loss.mean()  # .mean() because it is not reduced before
        self.optim.zero_grad()
        mean_loss.backward()
        self.fresh_loss = False
        self.optim.step()
        return mean_loss.item()

@dataclass
class RNDCfg:
    pred_shape: int = MISSING
    target_hidden_layers: list[int] = MISSING
    predictor_hidden_layers: list[int] = MISSING

    lr: float = MISSING
