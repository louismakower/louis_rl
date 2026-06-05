
import copy
import math

import torch
from torch import optim
import torch.nn as nn

from rl_games.algos_torch.sac_helper import SquashedNormal

def build_mlp(sizes, device) -> nn.Module:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i != len(sizes) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers).to(device=device)

class Policy:
    def __init__(
            self,
            logstd_min: float,
            logstd_max: float,
            obs_size: int,
            hidden_dims: list[int],
            action_shape: int,
            lr: float,
            device: str,
            alpha_init: float,
            alpha_lr: float,
            target_entropy: float,
    ):
        self.device = device
        self.log_alpha = torch.tensor([math.log(alpha_init)], requires_grad=True, device=device)
        self.alpha_lr = alpha_lr
        self.target_entropy = target_entropy
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max
        self.obs_size = obs_size
        self.hidden_dims = hidden_dims
        self.action_shape = action_shape
        self.network = build_mlp(
            sizes=(self.obs_size, *self.hidden_dims, 2*self.action_shape),
            device=self.device
        )
        self.optimiser = optim.AdamW(self.network.parameters(), lr=lr)
        self.log_alpha_optimiser = torch.optim.AdamW(
            [self.log_alpha],
            lr=self.alpha_lr,
        )

    def train_one_step(self, min_q, log_prob):
        # update policy
        curr_alpha = self.log_alpha.exp().clone()
        self.optimiser.zero_grad()
        ent = -(curr_alpha.detach() * log_prob)
        max_this = (min_q + ent).mean()
        min_this = -max_this
        min_this.backward()
        self.optimiser.step()

        # update alpha
        alpha_loss = (
            self.log_alpha.exp() * (-log_prob - self.target_entropy).detach()
        ).mean()
        self.log_alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimiser.step()

        return self.log_alpha.detach().exp().item(), min_this.item(), alpha_loss.item()

    def dist(self, obs):
        mu, logstd = self.network(obs).chunk(2, dim=-1)
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
        std = logstd.exp()
        return SquashedNormal(loc=mu, scale=std)


class Q:
    def __init__(
            self,
            in_size: int,
            hidden_dims: list[int],
            out_size: int,
            lr: float,
            tau: float,
            device: str,
            grad_clip_norm: float = 1.0,
    ):
        self.device = device
        self.network: nn.Module = build_mlp(
            sizes=(in_size, *hidden_dims, out_size),
            device=self.device
        )
        self.optimiser = optim.AdamW(self.network.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.target = copy.deepcopy(self.network)
        self.tau = tau
        self.grad_clip_norm = grad_clip_norm

    def train_one_step(self, x, y):
        self.optimiser.zero_grad()
        out = self.network(x)
        loss = self.loss(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.grad_clip_norm)
        self.optimiser.step()
        return loss.item()

    def update_target(self):
        for param, target_param in zip(
            self.network.parameters(),
            self.target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def target_network_diff(self, x):
        with torch.no_grad():
            q1_out = self.network(x)
            q1_targ_out = self.target(x)
        mse = ((q1_out - q1_targ_out)**2).mean()
        return mse