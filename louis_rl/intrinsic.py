from __future__ import annotations
import torch.nn as nn
import torch.optim as optim
import torch

from dataclasses import dataclass, MISSING

from rl_games.common.experience import VectorizedReplayBuffer

from .vec_env import VecEnv


class RND:
    def __init__(self, cfg: RNDCfg, env: VecEnv):
        self._env: VecEnv = env
        self.cfg: RNDCfg = cfg
        self.device = env.device
        self.obs_dim = self._env.observation_space["rnd"].shape[0]
        self.pred_shape = self.cfg.pred_shape
        self._init_networks()
        self.loss = nn.MSELoss(reduction='none')
        self.recent_loss = None
        self.fresh_loss = False
        self.optim = optim.AdamW(self.predictor.parameters(), lr=self.cfg.lr)

    def _build_mlp(self, in_dim: int, hidden_layers: list[int], out_dim: int) -> nn.Sequential:
        layers = []
        prev = in_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def _init_networks(self):
        self.target = self._build_mlp(self.obs_dim, self.cfg.target_hidden_layers, self.pred_shape).to(self.device)
        for param in self.target.parameters():
            param.requires_grad = False

        self.predictor = self._build_mlp(self.obs_dim, self.cfg.predictor_hidden_layers, self.pred_shape).to(self.device)

    def get_intrinsic_rew(self, obs):
        pred = self.predictor(obs)
        with torch.no_grad():
            target = self.target(obs)
        self.recent_loss = self.loss(pred, target)
        self.fresh_loss = True
        return self.recent_loss.detach().mean(dim=-1, keepdim=True)  # average over prediction dimension

    def train(self):
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

class IntrinsicBuffer(VectorizedReplayBuffer):
    def __init__(self, obs_shape, action_shape, intrns_obs_shape, capacity, device):
        super().__init__(obs_shape, action_shape, capacity, device)
        self.intrns_obs = torch.empty((capacity, *intrns_obs_shape), dtype=torch.float32, device=self.device)

    def add(self, obs, action, reward, next_obs, done, intrns_obs):
        super().add(obs, action, reward, next_obs, done)

        num_observations = obs.shape[0]
        remaining_capacity = min(self.capacity - self.idx, num_observations)
        overflow = num_observations - remaining_capacity
        if remaining_capacity < num_observations:
            self.intrns_obs[0: overflow] = intrns_obs[-overflow:]
        self.intrns_obs[self.idx: self.idx + remaining_capacity] = intrns_obs[:remaining_capacity]

    def sample(self, batch_size):
        idxs = torch.randint(0,
                            self.capacity if self.full else self.idx,
                            (batch_size,), device=self.device)
        obses = self.obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_obses = self.next_obses[idxs]
        dones = self.dones[idxs]
        intrns_obs = self.intrns_obs[idxs]

        return obses, actions, rewards, next_obses, dones, intrns_obs
