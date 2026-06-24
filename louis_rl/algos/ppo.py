from __future__ import annotations
from dataclasses import dataclass, MISSING
import torch
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch import optim

from .base_runner import BaseRunner
from louis_rl.vec_env import VecEnv
from louis_rl.utils.networks import build_mlp
from louis_rl.implementations.intrinsic import IntrinsicCfg, IntrinsicModule

class PPORunner(BaseRunner):
    def __init__(
            self,
            env: VecEnv,
            cfg: PPORunnerCfg,
            log_dir: str,
            inference_only: bool = False,
    ):
        super().__init__(log_dir)
        self.inference_only = inference_only
        self._env = env
        self.device = self._env.device
        self.num_envs = self._env.num_envs
        self.cfg: PPORunnerCfg = cfg
        self.act_dim = self._env.action_space.shape[0]
        self._init_obs()
        self._init_networks()
        if self.inference_only:
            self.writer = None
            self.intrinsic = None
            return
        self.rew_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, device=self.device)
        self.act_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, self.act_dim, device=self.device)
        self.obs_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, self.obs_shape, device=self.device)
        self.next_obs_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, self.obs_shape, device=self.device)
        self.term_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, dtype=torch.bool, device=self.device)
        self.timeout_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, dtype=torch.bool, device=self.device)
        self.log_prob_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, device=self.device)
        self.writer = SummaryWriter(log_dir=log_dir)
        self._init_intrinsic()
            
        self.cov_mat = torch.diag(
            torch.full(size=(self.act_dim,), fill_value=0.5),
        ).to(device=self.device)

    def _init_intrinsic(self):
        if self.cfg.intrinsic is not None:
            self.intrinsic_obs_shape = self._env.observation_space["rnd"].shape[0]
            self.intrinsic = IntrinsicModule.create(self.cfg.intrinsic, self.device, obs_dim=self.intrinsic_obs_shape)
            self.intrinsic_V = build_mlp(
                sizes=(self.obs_shape, *self.cfg.intrinsic_V_hidden_layers, 1),
                device=self.device
            )
            self.intrinsic_optim = optim.Adam(self.intrinsic_V.parameters(), lr=self.cfg.intrinsic_V_lr)
            self.intrinsic_rew_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, device=self.device)
            self.intrinsic_next_obs_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, self.intrinsic_obs_shape, device=self.device)
        else:
            self.intrinsic_obs_shape = None
            self.intrinsic = None
            self.intrinsic_V = None
            self.intrinsic_optim = None
            self.intrinsic_rew_buf = None
            self.intrinsic_next_obs_buf = None

    def _init_obs(self):
        self.policy_obs_dim = sum(
            self._env.observation_space[k].shape[0]
            for k in self._env.observation_space if "policy" in k
        )
        goal_obs = self._env.observation_space.get("goal")
        goal_obs_dim = sum(goal_obs[k].shape[0] for k in goal_obs) if goal_obs else 0
        self.obs_shape = self.policy_obs_dim + goal_obs_dim

    def _init_networks(self):
        self.policy = build_mlp(
            sizes=(self.obs_shape, *self.cfg.policy_hidden_dims, self.act_dim),
            device=self.device,
        )
        self.v = build_mlp(
            sizes=(self.obs_shape, *self.cfg.v_hidden_dims, 1),
            device=self.device
        )
        self.policy_optim = optim.AdamW(self.policy.parameters(), lr=self.cfg.policy_lr)
        self.v_optim = optim.AdamW(self.v.parameters(), lr=self.cfg.v_lr)
        self.mse = nn.MSELoss()

    def learn(self):
        obs, extras = self._env.reset()
        self._env.randomise_ep_counters()
        step = 0
        for iteration_idx in range(self.cfg.num_iterations):
            next_obs, step, ep_infos = self.rollout(obs, step)
            with torch.no_grad():
                rtg = self.calc_rtg()  # (N, L)
                advantages = self.calc_adv(rtg).detach()  # (N, L)
            self.writer.add_scalar("advantages/extrinsic", advantages.mean(), self.global_step)

            # add instrinsic advantages
            if self.intrinsic is not None:
                with torch.no_grad():
                    intrns_rtg = self.calc_intrinsic_rtg()
                    intrins_adv = self.calc_intrinsic_adv(intrns_rtg).detach()
                    intrins_adv = intrins_adv * self.cfg.intrinsic_weight
                    self.writer.add_scalar("advantages/intrinsic", intrins_adv.mean(), self.global_step)
                    advantages = advantages + intrins_adv
                    self.writer.add_scalar("advantages/total", advantages.mean(), self.global_step)
            
            for _ in range(self.cfg.num_policy_grad_steps):
                policy_loss = self.policy_step(advantages)
            self.writer.add_scalar("policy/policy_loss", policy_loss, self.global_step)
            for _ in range(self.cfg.num_v_grad_steps):
                v_loss = self.value_step(rtg)
            if self.intrinsic is not None:
                for _ in range(self.cfg.intrinsic_v_grad_steps):
                    intrinsic_loss = self.intrinsic_step(intrns_rtg)
                self.writer.add_scalar("intrinsic/intrinsic_v_loss", intrinsic_loss, self.global_step)
                self.intrinsic.train_one_step(self.intrinsic_next_obs_buf.reshape(-1, self.intrinsic_obs_shape))  # reshape so use_frac samples transitions not envs
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
    
    def intrinsic_step(self, intrinsic_rtg):
        intrinsic_V = self.intrinsic_V(self.obs_buf).squeeze(-1)
        loss = self.mse(intrinsic_V, intrinsic_rtg)

        self.intrinsic_optim.zero_grad()
        loss.backward()
        self.intrinsic_optim.step()
        return loss.item()

    def calc_adv(self, rtg):
        val = self.v(self.obs_buf).squeeze(-1)  # (N, L)
        return rtg - val
    
    def calc_intrinsic_adv(self, intrinsic_rtg):
        val = self.intrinsic_V(self.obs_buf).squeeze(-1)
        return intrinsic_rtg - val

    def calc_intrinsic_rtg(self):
        rtgs = torch.zeros_like(self.intrinsic_rew_buf)
        for step_idx in range(self.intrinsic_rew_buf.shape[1]-1, -1, -1):
            next_obs = self.next_obs_buf[:, step_idx]

            rew = self.intrinsic_rew_buf[:, step_idx]

            # on last step, always bootstrap
            if step_idx == self.intrinsic_rew_buf.shape[1]-1:
                with torch.no_grad():
                    bootstrap_val = self.intrinsic_V(next_obs).squeeze(-1)  # (N,)
                discounted_rew = bootstrap_val
            # on other steps, use RTG
            else:
                discounted_rew = rtgs[:, step_idx+1]

            rtgs[:, step_idx] = rew + self.cfg.intrinsic_gamma*discounted_rew

        return rtgs

    def calc_rtg(self):
        rtgs = torch.zeros_like(self.rew_buf)
        for step_idx in range(self.rew_buf.shape[1]-1, -1, -1):
            next_obs = self.next_obs_buf[:, step_idx]

            rew = self.rew_buf[:, step_idx]
            with torch.no_grad():
                bootstrap_val = self.v(next_obs).squeeze(-1)  # (N,)

            # on last step, always bootstrap unless a termination
            if step_idx == self.rew_buf.shape[1]-1:
                    # on termination, future reward is 0, otherwise bootstrap
                    discounted_rew = torch.where(self.term_buf[:, step_idx], 0, bootstrap_val)
            # on other steps, bootstrap only on timeout
            else:
                discounted_rew = torch.where(self.timeout_buf[:, step_idx], bootstrap_val, rtgs[:, step_idx+1])
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
        for step_idx in range(self.cfg.steps_per_rollout):
            if isinstance(obs, dict):
                obs = self.add_goal_obs(obs)
            with torch.no_grad():
                act, log_prob = self.get_action(obs)
            next_obs, rew, term, timeout, extras = self._env.step(act)
            resetted = term | timeout
            self.global_step += self.num_envs
            
            if self.intrinsic:
                # handle terminal RND obs and add to buffer
                terminal_intrinsic = extras["terminal_obs"]["rnd"]
                intrinsic_next_obs = torch.where(resetted.unsqueeze(-1), terminal_intrinsic, next_obs["rnd"])
                intrns_rew = self.intrinsic.get_intrinsic_rew(intrinsic_next_obs, update_norm_stats=True)
                self.intrinsic_rew_buf[:, step_idx] = intrns_rew.squeeze(-1)
                self.intrinsic_next_obs_buf[:, step_idx] = intrinsic_next_obs
            
            # handle terminal observations
            next_obs_t = self.add_goal_obs(next_obs)
            terminal_t = self.add_goal_obs(extras["terminal_obs"])
            next_obs_for_buf = torch.where(resetted.unsqueeze(-1), terminal_t, next_obs_t)

            self.obs_buf[:, step_idx] = obs
            self.next_obs_buf[:, step_idx] = next_obs_for_buf
            self.rew_buf[:, step_idx] = rew
            self.term_buf[:, step_idx] = term
            self.timeout_buf[:, step_idx] = timeout
            self.log_prob_buf[:, step_idx] = log_prob
            self.act_buf[:, step_idx] = act
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
    
    # RND
    intrinsic: None | IntrinsicCfg = MISSING
    intrinsic_V_hidden_layers: list[int] = MISSING
    intrinsic_gamma: float = MISSING
    intrinsic_v_grad_steps: int = MISSING
    intrinsic_V_lr: float = MISSING
    intrinsic_weight: float = MISSING
    
    algo_name: str = "ppo"
