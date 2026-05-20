from __future__ import annotations

from dataclasses import dataclass, MISSING
import torch

from rl_games.common.experience import VectorizedReplayBuffer
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from .networks import Policy, Q
from .reward_normaliser import RewardNormalizer
from .base_runner import BaseRunner
from .her import HERCfg
from .vec_env import VecEnv, Logger

def _recurse_obs(item, fn):
    new_dict = {}
    if isinstance(item, dict):
        for k, v in item.items():
            new_dict[k] = _recurse_obs(v, fn)
        return new_dict
    else:
        return fn(item)

class SACRunner(BaseRunner):
    def __init__(
            self,
            env: VecEnv,
            cfg: SACRunnerCfg,
            log_dir: str,
            writer: Logger,
    ):
        super().__init__(log_dir)
        self._env: VecEnv = env
        self.cfg = cfg
        self.alpha = cfg.alpha_init
        self.device = self._env.device
        self.num_envs = self._env.num_envs
        self.env_arange = torch.arange(self.num_envs, device=self.device)
        self.max_ep_len = self._env.max_episode_length
        self.action_shape = self._env.action_space.shape[0]
        self.target_entropy = -self.action_shape if self.cfg.target_entropy == "auto" else float(self.cfg.target_entropy)
        print(f"[SAC]: Target entropy set to {self.target_entropy}")
        if self.cfg.her_cfg:
            self.her = cfg.her_cfg
        else:
            self.her = None
        self._init_obs()
        self._init_networks()
        self.writer = writer
        if self.cfg.reward_scaling:
            self.rew_norm = RewardNormalizer(
                gamma=cfg.gamma,
                G_max=cfg.reward_G_max,
                device=self._env.device,
            )
        self._print_UTD_ratio()

    def _print_UTD_ratio(self):
        updates = self.cfg.num_train_updates
        trans_added = self.cfg.steps_per_iter * self.num_envs
        utd = updates / trans_added
        print(f"[INFO]: UTD ratio is currently: {utd:.3f}")

    def _init_obs(self):
        self.policy_obs_dim = self._env.observation_space["policy"].shape[0]
        goal_obs = self._env.observation_space.get("goal")
        goal_obs_dim = sum(goal_obs[k].shape[0] for k in goal_obs) if goal_obs else 0
        self.obs_shape = self.policy_obs_dim + goal_obs_dim
        self.obs_norm = RunningMeanStd(insize=(self.obs_shape,)).to(self._env.device)

    def _set_eval(self):
        self.q1.network.eval()
        self.q2.network.eval()
        self.policy.network.eval()
        self.obs_norm.eval()
    def _set_train(self):
        self.q1.network.train()
        self.q2.network.train()
        self.policy.network.train()
        self.obs_norm.train()

    def _get_checkpoint(self) -> dict:
        return {
            "policy": self.policy.network.state_dict(),
            "obs_norm": self.obs_norm.state_dict(),
        }

    def _load_checkpoint(self, state: dict):
        self.policy.network.load_state_dict(state["policy"])
        self.obs_norm.load_state_dict(state["obs_norm"])

    def get_deterministic_action(self, obs):
        self._set_eval()
        obs = self.add_goal_obs(obs)
        obs_n = self.obs_norm(obs)
        mu, _ = self.policy.network(obs_n).chunk(2, dim=-1)
        return torch.tanh(mu)

    def rollout_and_add_to_buffer(self, obs, ep_infos):
        self._set_eval()
        for step in range(self.cfg.steps_per_iter):
            obs_t = self.add_goal_obs(obs)
            if self.warmup:
                action = (torch.rand(size=(obs_t.shape[0], self.action_shape), device=self.device) * 2) - 1
            else:
                action = self.policy.dist(self.obs_norm(obs_t)).sample()

            next_obs, ex_rew, term, timeout, extras = self._env.step(action)
            if self.her:
                # make sure all her obs are shape (N, obs_dim)
                obs = _recurse_obs(obs, fn=lambda v: v.unsqueeze(-1) if v.dim() == 1 else v)
                next_obs = _recurse_obs(next_obs, fn=lambda v: v.unsqueeze(-1) if v.dim() == 1 else v)

            next_obs_t = self.add_goal_obs(next_obs)

            # for resets, use the terminal obs rather than next_obs
            # so we bootstrap off the correct obs, rather than an
            # uncorrelated random reset state
            resetted = term | timeout
            terminal_t = self.add_goal_obs(extras["terminal_obs"])  # (N, obs_dim), nan for non-reset
            next_obs_for_buffer = torch.where(resetted.unsqueeze(-1), terminal_t, next_obs_t)

            self.global_step += self.num_envs

            self.writer.add_scalar("rewards/extrinsic_mean", ex_rew.mean(), self.global_step)
            if self.cfg.reward_scaling:
                self.rew_norm.update_reward_stats(ex_rew, term, timeout)

            done = term.unsqueeze(1)  # ie DO bootstrap on timeout, DON'T on termination
            # buffer contains transitions which have the RAW observations
            # the goal observation has been added to it
            self.buffer.add(obs_t, action, ex_rew.unsqueeze(1), next_obs_for_buffer, done)

            if self.her:
                self.get_her_and_add_to_buffer(
                    obs=obs,
                    obs_t=obs_t,
                    next_obs=next_obs,
                    next_obs_for_buffer=next_obs_for_buffer,
                    action=action,
                    term=term,
                    resetted=resetted,
                    extras=extras,
                )


            obs = next_obs

        replay_buffer_state = self.buffer.capacity if self.buffer.full else self.buffer.idx
        self.writer.add_scalar("misc/replay_buffer_state", replay_buffer_state, self.global_step)
        ep_infos.append(extras.get("log", {}))
        return obs

    def get_her_and_add_to_buffer(
            self,
            obs,
            obs_t,
            next_obs,
            next_obs_for_buffer,
            action,
            term,
            resetted,
            extras,
    ):
        if self.her.her_obs_buf is None:
            self._init_her_obs_buf(obs["her"])

        ptr = self.her.ep_ptr
        self.her.trajectories["obs"][ptr, self.env_arange] = obs_t
        self.her.trajectories["action"][ptr, self.env_arange] = action
        self.her.trajectories["next_obs"][ptr, self.env_arange] = next_obs_for_buffer
        self.her.trajectories["term"][ptr, self.env_arange] = term.unsqueeze(1)
        for key, val in obs["her"].items():
            self.her.her_obs_buf[key][ptr, self.env_arange] = val

        # fill next_her_obs: use terminal_obs for reset envs, next_obs for others
        for key, val in next_obs["her"].items():
            term_her = extras["terminal_obs"]["her"][key]  # (N, ...), nan for non-reset
            next_her = val.clone()
            if resetted.any():
                next_her[resetted] = term_her[resetted]
            self.her.her_obs_buf[f"next_{key}"][ptr, self.env_arange] = next_her

        if resetted.any():
            term_envs = resetted.nonzero(as_tuple=False).squeeze(-1)
            lengths = self.her.ep_ptr[term_envs] + 1  # (num_term,) +1: ep_ptr is 0-indexed
            max_len = lengths.max().item()

            sliced = {
                "obs": self.her.trajectories["obs"][:max_len, term_envs],
                "action": self.her.trajectories["action"][:max_len, term_envs],
                "next_obs": self.her.trajectories["next_obs"][:max_len, term_envs],
                "term": self.her.trajectories["term"][:max_len, term_envs],
                "her_obs": {k: self.her.her_obs_buf[k][:max_len, term_envs] for k in self.her.her_obs_buf},
                "lengths": lengths,
            }

            t_idx = torch.arange(max_len, device=self.device).unsqueeze(1)  # (max_len, 1)
            sliced["valid"] = (t_idx < lengths.unsqueeze(0)).reshape(-1)    # (max_len * num_term,)

            new_transitions = self.her.get_hindsight_transitions(sliced, extras={"policy_obs_dim": self.policy_obs_dim})
            self.buffer.add(
                new_transitions["obs"],
                new_transitions["action"],
                new_transitions["reward"],
                new_transitions["next_obs"],
                new_transitions["done"],
            )
            if new_transitions["reward"].shape[0] != 0:
                self.writer.add_histogram("her/distances", new_transitions["distances"], self.global_step)
                self.writer.add_histogram("her/rewards", new_transitions["reward"], self.global_step)
                self.writer.add_scalar("her/her_ave_rew", new_transitions["reward"].mean(), self.global_step)

        self.her.ep_ptr = torch.where(
            resetted,
            torch.zeros_like(self.her.ep_ptr),
            self.her.ep_ptr + 1,
        )

    def train_batch(self, update_step):
        b_obs, b_act, b_extrns_rew, b_next_obs, b_dones = self.buffer.sample(self.cfg.batch_size)
        b_obs_n = self.obs_norm(b_obs)
        b_next_obs_n = self.obs_norm(b_next_obs)
        if update_step == 0:
            for i in range(0, b_act.shape[-1], 1):
                self.writer.add_histogram(f"act/dim_{i}", b_act[:, i], self.global_step)

        # train Q
        b_extrns_rew_scaled = self.rew_norm.normalize_rewards(b_extrns_rew) if self.cfg.reward_scaling else b_extrns_rew
        b_extrns_rew_scaled = torch.clamp(b_extrns_rew_scaled, -self.cfg.reward_clip, self.cfg.reward_clip) if self.cfg.reward_clip > 0 else b_extrns_rew_scaled
        q_targ = self._compute_q_targ(b_extrns_rew_scaled, b_next_obs_n, b_dones)

        # bookeeping
        self.writer.add_scalar("q_target/extrinsic_rew", b_extrns_rew.mean(), self.global_step)
        self.writer.add_scalar("q_target/extrinsic_rew_std", b_extrns_rew.std(), self.global_step)
        self.writer.add_scalar("q_target/extrinsic_rew_scaled", b_extrns_rew_scaled.mean(), self.global_step)
        self.writer.add_scalar("q_target/extrinsic_rew_scaled_std", b_extrns_rew_scaled.std(), self.global_step)
        self.writer.add_scalar("critic/q_target", q_targ.mean(), self.global_step)

        q_input = self._q_input(b_obs_n, b_act)
        q1_loss = self.q1.train_one_step(q_input, q_targ)
        q2_loss = self.q2.train_one_step(q_input, q_targ)
        self.writer.add_scalar("critic/q1_loss", q1_loss, self.global_step)
        self.writer.add_scalar("critic/q2_loss", q2_loss, self.global_step)

        # train policy
        self._train_onestep_policy(b_obs_n)

        # update critic target networks
        self.q1.update_target()
        self.q2.update_target()
        q1_targ_diff = self.q1.target_network_diff(q_input)
        q2_targ_diff = self.q2.target_network_diff(q_input)
        self.writer.add_scalar("critic/q1_targ_diff", q1_targ_diff, self.global_step)
        self.writer.add_scalar("critic/q2_targ_diff", q2_targ_diff, self.global_step)

    def learn(self):
        obs, extras = self._env.reset()
        self.warmup = True
        ep_infos = []
        for train_step in range(self.cfg.max_steps // self.cfg.steps_per_iter):
            obs = self.rollout_and_add_to_buffer(
                obs=obs,
                ep_infos=ep_infos,
            )

            self.warmup = self.global_step <= self.cfg.warmup_transitions
            if not self.warmup:
                all_keys = set(k for d in ep_infos for k in d)
                for key in all_keys:
                    vals = [d[key] for d in ep_infos if key in d]
                    mean_val = sum(float(v) for v in vals) / len(vals)
                    self.writer.add_scalar(key, mean_val, self.global_step)
                ep_infos = []
                self._set_train()
                for update_step in range(self.cfg.num_train_updates):
                    self.train_batch(update_step)


            if self.warmup and train_step % 10 == 0:
                print(f"[SAC] Warming up, {self.global_step} out of {self.cfg.warmup_transitions} ({100*self.global_step / self.cfg.warmup_transitions:.1f}%)")
            if self.global_step >= self.cfg.warmup_transitions and self.global_step <= self.cfg.warmup_transitions + self.num_envs*self.cfg.steps_per_iter:
                print("[SAC] Warmup over, starting policy rollout")

            if not self.warmup and train_step % self.cfg.save_interval == 0:
                self.save_checkpoint()

    def _critic_require_grad(self, t: bool):
        if type(t) is not bool:
            raise ValueError
        for p in self.q1.network.parameters():
            p.requires_grad = t
        for p in self.q2.network.parameters():
            p.requires_grad = t

    def _train_onestep_policy(self, obs_n):
        dist = self.policy.dist(obs_n)
        act = dist.rsample()
        log_prob = dist.log_prob(act).sum(dim=-1, keepdim=True)  # sum over action dimensions (model as independent)
        self._critic_require_grad(False)
        min_q = self._min_q(obs_n, act, use_target=False)
        self._critic_require_grad(True)
        self.alpha, policy_loss, alpha_loss = self.policy.train_one_step(min_q, log_prob)
        self.writer.add_scalar("alpha/alpha", self.alpha, self.global_step)
        self.writer.add_scalar("alpha/alpha_loss", alpha_loss, self.global_step)
        self.writer.add_scalar("policy/policy_loss", policy_loss, self.global_step)
        self.writer.add_scalar("policy/entropy_term", -log_prob.mean(), self.global_step)
        self.writer.add_scalar("policy/q_term", min_q.mean(), self.global_step)

    def _min_q(self, obs, act, use_target):
        q_input = self._q_input(obs, act)
        if use_target:
            q1 = self.q1.target(q_input)
            q2 = self.q2.target(q_input)
        else:
            q1 = self.q1.network(q_input)
            q2 = self.q2.network(q_input)
        q = torch.where(q1 <= q2, q1, q2)
        return q

    def _compute_q_targ(
            self,
            rew_scaled,
            next_obs_n,
            dones,
    ):
        with torch.no_grad():
            dist = self.policy.dist(next_obs_n)
            next_act = dist.sample()
            log_prob = dist.log_prob(next_act).sum(-1, keepdim=True)  # sum over action dimensions (model as independent)
            q = self._min_q(next_obs_n, next_act, use_target=True)

            self.writer.add_scalar("q_target/log_prob", log_prob.mean(), self.global_step)
            self.writer.add_scalar("q_target/q", q.mean(), self.global_step)


        return rew_scaled + self.cfg.gamma * (1 - dones.int()) * (q - self.alpha * log_prob)

    def _init_networks(self):
        self.policy = self._init_policy()
        self.q1 = self._init_q()
        self.q2 = self._init_q()
        self.buffer = self._init_buffer()
        if self.her:
            self.her.trajectories, self.her.ep_ptr = self._init_her()
            self.her.her_obs_buf = None

    def _init_her(self):
        her_trajectories = {
            "obs": torch.zeros(self.max_ep_len, self.num_envs, self.obs_shape, device=self.device),
            "action": torch.zeros(self.max_ep_len, self.num_envs, self.action_shape, device=self.device),
            "next_obs": torch.zeros(self.max_ep_len, self.num_envs, self.obs_shape, device=self.device),
            "term": torch.zeros(self.max_ep_len, self.num_envs, 1, dtype=torch.bool, device=self.device),
        }
        ep_ptr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        return her_trajectories, ep_ptr

    def _init_her_obs_buf(self, her_obs):
        self.her.her_obs_buf = {}
        for key, val in her_obs.items():
            shape = val.shape[1:]
            self.her.her_obs_buf[key] = torch.zeros(
                self.max_ep_len, self.num_envs, *shape, device=self.device, dtype=val.dtype
            )
            self.her.her_obs_buf[f"next_{key}"] = torch.zeros(
                self.max_ep_len, self.num_envs, *shape, device=self.device, dtype=val.dtype
            )

    def _init_q(self):
        in_size = self.action_shape + self.obs_shape
        return Q(
            in_size=in_size,
            hidden_dims=self.cfg.q_hidden_dims,
            out_size=1,
            lr=self.cfg.q_learning_rate,
            tau=self.cfg.q_tau,
            device=self._env.device,
            grad_clip_norm=self.cfg.q_grad_clip_norm,
        )

    def _init_buffer(self):
        return VectorizedReplayBuffer(
                obs_shape=[self.obs_shape],
                action_shape=[self.action_shape],
                capacity=self.cfg.replay_buffer_size,
                device=self._env.device,
            )

    def _init_policy(self):
        return Policy(
            logstd_min=self.cfg.logstd_min,
            logstd_max=self.cfg.logstd_max,
            obs_size=self.obs_shape,
            hidden_dims=self.cfg.policy_hidden_dims,
            action_shape=self.action_shape,
            lr=self.cfg.policy_learning_rate,
            device=self._env.device,
            alpha_init=self.alpha,
            alpha_lr=self.cfg.alpha_lr,
            target_entropy=self.target_entropy,
        )

    @staticmethod
    def _q_input(obs, act):
        return torch.cat([obs, act], dim=-1)


@dataclass
class SACRunnerCfg:
    # MDP
    gamma: float = MISSING
    alpha_init: float = MISSING
    alpha_lr: float = MISSING
    target_entropy: str = MISSING  # float value or "auto" = -dim(A)

    # buffer
    replay_buffer_size: int = MISSING
    warmup_transitions: int = MISSING  # total number of transitions over all envs

    # critic
    q_hidden_dims: list[int] = MISSING
    q_learning_rate: float = MISSING
    q_tau: float = MISSING
    q_grad_clip_norm: float = MISSING

    # actor
    policy_hidden_dims: list[int] = MISSING
    logstd_min: float = MISSING
    logstd_max: float = MISSING
    policy_learning_rate: float = MISSING

    reward_scaling: bool = MISSING
    reward_G_max: float = MISSING  # ignored when reward_scaling=False
    reward_clip: float = MISSING  # 0.0 = disabled

    # training
    max_steps: int = MISSING
    steps_per_iter: int = MISSING
    num_train_updates: int = MISSING
    batch_size: int = MISSING

    # logging
    experiment_name: str = MISSING

    save_interval: int = MISSING

    collect_states: bool = MISSING

    her_cfg: HERCfg | None = None
    algo_name: str = "sac"
