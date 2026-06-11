from __future__ import annotations
from dataclasses import dataclass, MISSING
import torch
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch import optim

from .base_runner import BaseRunner
from .vec_env import VecEnv
from .networks import build_mlp
from .intrinsic import RND, RNDCfg
from .reward_normaliser import RewardNormaliser

class PPORunner(BaseRunner):
    def __init__(
            self,
            env: VecEnv,
            cfg: PPORunnerCfg,
            log_dir: str,
    ):
        super().__init__(log_dir)
        self._env = env
        self.device = self._env.device
        self.num_envs = self._env.num_envs
        self.cfg = cfg
        self.act_dim = self._env.action_space.shape[0]
        self._init_obs()

        # RND
        if self.cfg.use_rnd:
            self.rnd_obs_shape = self._env.observation_space["rnd"].shape[0]
            self.rnd = RND(self.cfg.rnd_cfg, device=self.device, obs_dim=self.rnd_obs_shape)
            self.intrinsic_rew_norm = RewardNormaliser(self.cfg.rnd_gamma, G_max=self.cfg.rnd_G_max, device=self.device)
            self.rnd_obs_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, self.rnd_obs_shape, device=self.device)
            self.int_rew_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, device=self.device)
        else:
            self.rnd = None

        self._init_networks()
        self.rew_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, device=self.device)
        self.act_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, self.act_dim, device=self.device)
        self.obs_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, self.obs_shape, device=self.device)
        self.next_obs_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, self.obs_shape, device=self.device)
        self.term_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, dtype=torch.bool, device=self.device)
        self.timeout_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, dtype=torch.bool, device=self.device)
        self.log_prob_buf = torch.zeros(self.num_envs, self.cfg.steps_per_rollout, device=self.device)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.cov_mat = torch.diag(
            torch.full(size=(self.act_dim,), fill_value=0.5),
        ).to(device=self.device)

    def _init_obs(self):
        self.policy_obs_dim = self._env.observation_space["policy"].shape[0]
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
        if self.rnd:
            self.v_int = build_mlp(
                sizes=(self.obs_shape, *self.cfg.rnd_critic_hidden_layers, 1),
                device=self.device
            )
            self.v_int_optim = optim.AdamW(self.v_int.parameters(), lr=self.cfg.rnd_critic_lr)
        self.mse = nn.MSELoss()

    def learn(self):
        obs, extras = self._env.reset()
        step = 0
        for iteration_idx in range(self.cfg.num_iterations):
            next_obs, step, ep_infos = self.rollout(obs, step)
            with torch.no_grad():
                rtg = self.calc_rtg()  # (N, L)
                advantages = self.calc_adv(rtg).detach()  # (N, L)
                if self.rnd:
                    int_rew_n, int_rew_scale = self.intrinsic_rew_norm.normalise_rewards(self.int_rew_buf)
                    int_rew_n = torch.clamp(int_rew_n, -self.cfg.rnd_rew_clip, self.cfg.rnd_rew_clip) if self.cfg.rnd_rew_clip > 0 else int_rew_n
                    rtg_int = self.calc_rtg_int(int_rew_n)  # (N, L)
                    int_advantages = rtg_int - self.v_int(self.obs_buf).squeeze(-1)  # (N, L)
                    advantages = advantages + self.cfg.rnd_rew_weight * int_advantages
                    self.writer.add_scalar("rnd/intrinsic_rew_scaled", int_rew_n.mean(), self.global_step)
                    self.writer.add_scalar("rnd/intrinsic_rew_scale", int_rew_scale, self.global_step)
                    self.writer.add_scalar("rnd/intrinsic_advantage", int_advantages.mean(), self.global_step)
            for _ in range(self.cfg.num_policy_grad_steps):
                policy_loss = self.policy_step(advantages)
            self.writer.add_scalar("policy/policy_loss", policy_loss, self.global_step)
            for _ in range(self.cfg.num_v_grad_steps):
                v_loss = self.value_step(rtg)
                if self.rnd:
                    v_int_loss = self.value_step_int(rtg_int)
            self.writer.add_scalar("value/v_loss", v_loss, self.global_step)
            if self.rnd:
                self.writer.add_scalar("rnd/v_int_loss", v_int_loss, self.global_step)
                flat_rnd_obs = self.rnd_obs_buf.reshape(-1, self.rnd_obs_shape)
                for _ in range(self.cfg.num_rnd_grad_steps):
                    rnd_loss = self.rnd.train_one_step(flat_rnd_obs)
                if torch.is_tensor(rnd_loss):
                    self.writer.add_scalar("rnd/predictor_loss", rnd_loss.mean(), self.global_step)
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

    def value_step_int(self, rtg_int):
        v = self.v_int(self.obs_buf).squeeze(-1)
        loss = self.mse(v, rtg_int.detach())

        self.v_int_optim.zero_grad()
        loss.backward()
        self.v_int_optim.step()
        return loss.item()

    def calc_adv(self, rtg):
        val = self.v(self.obs_buf).squeeze(-1)  # (N, L)
        return rtg - val

    def calc_rtg(self):
        rtgs = torch.zeros_like(self.rew_buf)
        with torch.no_grad():
            # next_obs_buf holds the terminal obs for resetted envs, so this is
            # the correct bootstrap value at both timeouts and the rollout end
            bootstrap_vals = self.v(self.next_obs_buf).squeeze(-1)  # (N, L)
        last_idx = self.rew_buf.shape[1] - 1
        for step_idx in range(last_idx, -1, -1):
            rew = self.rew_buf[:, step_idx]

            if step_idx == last_idx:
                discounted_rew = bootstrap_vals[:, step_idx]
            else:
                # use RTG from next timestep, or bootstrap off the terminal obs if timeout
                discounted_rew = torch.where(self.timeout_buf[:, step_idx], bootstrap_vals[:, step_idx], rtgs[:, step_idx+1])

            # no discounted reward if terminated
            discounted_rew = torch.where(self.term_buf[:, step_idx], 0, discounted_rew)
            rtgs[:, step_idx] = rew + self.cfg.gamma*discounted_rew

        return rtgs

    def calc_rtg_int(self, int_rew):
        # non-episodic: intrinsic returns flow through episode boundaries,
        # so no term/timeout masking, only a bootstrap at the rollout end
        rtgs = torch.zeros_like(int_rew)
        last_idx = int_rew.shape[1] - 1
        with torch.no_grad():
            bootstrap_val = self.v_int(self.next_obs_buf[:, last_idx]).squeeze(-1)  # (N,)
        for step_idx in range(last_idx, -1, -1):
            next_rtg = bootstrap_val if step_idx == last_idx else rtgs[:, step_idx+1]
            rtgs[:, step_idx] = int_rew[:, step_idx] + self.cfg.rnd_gamma*next_rtg
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
            next_obs_t = self.add_goal_obs(next_obs)
            terminal_t = self.add_goal_obs(extras["terminal_obs"])  # (N, obs_dim), nan for non-reset

            # for resets, use the terminal obs rather than next_obs so we bootstrap
            # off the correct obs, rather than an uncorrelated random reset state
            resetted = term | timeout
            next_obs_for_buffer = torch.where(resetted.unsqueeze(-1), terminal_t, next_obs_t)

            self.obs_buf[:, step_idx] = obs
            self.next_obs_buf[:, step_idx] = next_obs_for_buffer
            self.rew_buf[:, step_idx] = rew
            self.term_buf[:, step_idx] = term
            self.timeout_buf[:, step_idx] = timeout
            self.log_prob_buf[:, step_idx] = log_prob
            self.act_buf[:, step_idx] = act

            # get intrinsic rewards with fresh obs stats; normalised after the rollout
            if self.rnd:
                terminal_rnd = extras["terminal_obs"]["rnd"]
                rnd_next = torch.where(resetted.unsqueeze(-1), terminal_rnd, next_obs["rnd"])
                self.rnd_obs_buf[:, step_idx] = rnd_next
                int_rew = self.rnd.get_intrinsic_rew(rnd_next, update_norm_stats=True).squeeze(-1)
                self.intrinsic_rew_norm.update_reward_stats(int_rew, torch.zeros_like(int_rew), torch.zeros_like(int_rew))
                self.int_rew_buf[:, step_idx] = int_rew
                self.writer.add_scalar("rewards/intrinsic_mean", int_rew.mean().item(), self.global_step)

            obs = next_obs_t

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

    # intrinsic rewards (defaults keep existing configs valid)
    use_rnd: bool = False
    rnd_cfg: RNDCfg | None = None
    rnd_critic_hidden_layers: list[int] | None = None
    rnd_critic_lr: float | None = None
    rnd_rew_weight: float | None = None
    rnd_rew_clip: float | None = None  # 0.0 = disabled
    rnd_gamma: float | None = None
    rnd_G_max: float | None = None
    num_rnd_grad_steps: int = 1

    algo_name: str = "ppo"
