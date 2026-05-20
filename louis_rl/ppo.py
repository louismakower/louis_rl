from __future__ import annotations
from dataclasses import dataclass, MISSING
import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn
from torch import optim

from .base_runner import BaseRunner
from .vec_env import VecEnv, Logger


class PPORunner(BaseRunner):
    def __init__(
            self,
            env: VecEnv,
            cfg: PPORunnerCfg,
            log_dir: str,
            writer: Logger,
    ):
        super().__init__(log_dir)
        self._env = env
        self.device = self._env.device
        self.num_envs = self._env.num_envs
        self.cfg = cfg
        self.act_dim = self._env.action_space.shape[0]
        self._init_obs()
        self._init_networks()
        self.rew_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, device=self.device)
        self.act_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, self.act_dim, device=self.device)
        self.obs_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, self.obs_shape, device=self.device)
        self.next_obs_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, self.obs_shape, device=self.device)
        self.term_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, dtype=torch.bool, device=self.device)
        self.timeout_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, dtype=torch.bool, device=self.device)
        self.log_prob_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, device=self.device)
        self.V_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, device=self.device)
        self.writer = writer

        self.cov_mat = torch.diag(
            torch.full(size=(self.act_dim,), fill_value=0.5),
        ).to(device=self.device)

    def _init_obs(self):
        self.policy_obs_dim = self._env.observation_space["policy"].shape[0]
        goal_obs = self._env.observation_space.get("goal")
        goal_obs_dim = sum(goal_obs[k].shape[0] for k in goal_obs) if goal_obs else 0
        self.obs_shape = self.policy_obs_dim + goal_obs_dim

    def _build_mlp(self, in_dim: int, out_dim: int, hidden_dims: list[int]) -> nn.Sequential:
        dims = [in_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
        layers.append(nn.Linear(dims[-1], out_dim))
        return nn.Sequential(*layers)

    def _init_networks(self):
        self.policy = self._build_mlp(self.obs_shape, self.act_dim, self.cfg.policy_hidden_dims).to(self.device)
        self.v = self._build_mlp(self.obs_shape, 1, self.cfg.v_hidden_dims).to(self.device)
        self.policy_optim = optim.AdamW(self.policy.parameters(), lr=self.cfg.policy_lr)
        self.v_optim = optim.AdamW(self.v.parameters(), lr=self.cfg.v_lr)
        self.mse = nn.MSELoss()

    def learn(self):
        obs, extras = self._env.reset()
        step = 0
        for iteration_idx in range(self.cfg.num_iterations):
            next_obs, step, ep_infos = self.rollout(obs, step)
            with torch.no_grad():
                rtg = self.calc_rtg()  # (N, L)
                advantages = self.calc_adv(rtg).detach()  # (N, L)
            for _ in range(self.cfg.num_policy_grad_steps):
                policy_loss = self.policy_step(advantages)
            self.writer.add_scalar("policy/policy_loss", policy_loss, self.global_step)
            for _ in range(self.cfg.num_v_grad_steps):
                v_loss = self.value_step(rtg)
            self.writer.add_scalar("value/v_loss", v_loss, self.global_step)
            all_keys = set(k for d in ep_infos for k in d)
            for key in all_keys:
                vals = [d[key] for d in ep_infos if key in d]
                mean_val = sum(float(v) for v in vals) / len(vals)
                self.writer.add_scalar(key, mean_val, self.global_step)
            obs = next_obs
            if (iteration_idx + 1) % self.cfg.save_interval == 0:
                self.save_checkpoint()

    def _get_checkpoint(self) -> dict:
        return {"policy": self.policy.state_dict()}

    def _load_checkpoint(self, state: dict):
        self.policy.load_state_dict(state["policy"])

    def get_deterministic_action(self, obs):
        self.policy.eval()
        obs = self.add_goal_obs(obs)
        return self.policy(obs)

    def get_log_prob(self, obs, act):
        mean = self.policy(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(act)
        return log_prob

    def policy_step(self, advantages):
        old_log_prob = self.log_prob_buf.detach()  # (N, L)
        curr_log_prob = self.get_log_prob(self.obs_buf, self.act_buf)  # (N, L)
        ratio = torch.exp(curr_log_prob - old_log_prob)  # (N, L)
        surr_1 = ratio * advantages  # (N, L)
        surr_2 = torch.clamp(ratio, 1-self.cfg.eps, 1+self.cfg.eps) * advantages
        loss = -torch.min(surr_1, surr_2).mean()

        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()
        return loss.item()

    def value_step(self, rtg):
        v = self.v(self.obs_buf).squeeze(-1)
        loss = self.mse(v, rtg.detach())

        self.v_optim.zero_grad()
        loss.backward()
        self.v_optim.step()
        return loss.item()

    def calc_adv(self, rtg):
        val = self.v(self.obs_buf).squeeze(-1)  # (N, L)
        return rtg - val

    def calc_rtg(self):
        rtgs = torch.zeros_like(self.rew_buf)
        for step_idx in range(self.rew_buf.shape[1]-1, -1, -1):
            next_obs = self.next_obs_buf[:, step_idx]

            rew = self.rew_buf[:, step_idx]

            if step_idx == self.rew_buf.shape[1]-1:
                # bootstrap next reward if not terminated
                with torch.no_grad():
                    bootstrap_val = self.v(next_obs).squeeze(-1)  # (N,)
                    discounted_rew = torch.where(self.timeout_buf[:, step_idx], self.V_buf[:, step_idx], bootstrap_val)
            else:
                # use RTG from next timestep or bootstrap if timeout
                # (i.e. use the value of the last state we have)
                discounted_rew = torch.where(self.timeout_buf[:, step_idx], self.V_buf[:, step_idx], rtgs[:, step_idx+1])

            # no discounted reward if terminated
            discounted_rew = torch.where(self.term_buf[:, step_idx], 0, discounted_rew)
            rtgs[:, step_idx] = rew + self.cfg.gamma*discounted_rew

        return rtgs

    def get_action(self, obs):
        mean = self.policy(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def rollout(self, obs, step):
        ep_infos = []
        if isinstance(obs, dict):
            obs = self.add_goal_obs(obs)
        for step_idx in range(self.cfg.steps_per_rollout):
            with torch.no_grad():
                act, log_prob = self.get_action(obs)
            next_obs, rew, term, timeout, extras = self._env.step(act)
            self.global_step += self.num_envs
            next_obs = self.add_goal_obs(next_obs)

            self.obs_buf[:, step_idx] = obs
            self.next_obs_buf[:, step_idx] = next_obs
            self.rew_buf[:, step_idx] = rew
            self.term_buf[:, step_idx] = term
            self.timeout_buf[:, step_idx] = timeout
            self.log_prob_buf[:, step_idx] = log_prob
            self.act_buf[:, step_idx] = act
            with torch.no_grad():
                self.V_buf[:, step_idx] = self.v(obs).squeeze(-1)
            obs = next_obs

            self.writer.add_scalar("rewards/mean", rew.mean().item(), self.global_step)
            ep_infos.append(extras.get("log", {}))
            step += 1

        return obs, step, ep_infos


@dataclass
class PPORunnerCfg:
    experiment_name: str = MISSING

    num_iterations: int = MISSING
    steps_per_rollout: int = MISSING

    num_policy_grad_steps: int = MISSING
    num_v_grad_steps: int = MISSING

    policy_lr: float = MISSING
    v_lr: float = MISSING

    policy_hidden_dims: list[int] = MISSING
    v_hidden_dims: list[int] = MISSING

    gamma: float = MISSING
    eps: float = MISSING

    save_interval: int = MISSING
    algo_name: str = "ppo"
