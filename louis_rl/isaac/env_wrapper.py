from __future__ import annotations
import torch
from gymnasium.spaces import Dict as DictSpace

from louis_rl.vec_env import SpaceInfo
from .terminal_obs_env import ReturnTerminalManagerBasedRLEnv


def _convert_space(space):
    """Convert Isaac obs space to VecEnv format, fixing (num_envs, obs_dim) → (obs_dim,)."""
    if isinstance(space, DictSpace):
        return {k: _convert_space(v) for k, v in space.spaces.items()}
    return SpaceInfo(shape=space.shape[1:])

def _fill_nan(item, mask):
    if isinstance(item, dict):
        return {k: _fill_nan(v, mask) for k, v in item.items()}
    t = item.clone().float()
    t[mask] = float("nan")
    return t

class IsaacEnvWrapper:
    """Adapts a ReturnTerminalManagerBasedRLEnv to the VecEnv protocol.

    Isaac obs spaces report shape (num_envs, obs_dim); this wrapper corrects
    observation_space to (obs_dim,) per the VecEnv convention.

    Episode-phase randomisation is the caller's responsibility before learn():
        env.unwrapped.episode_length_buf = torch.randint_like(
            env.unwrapped.episode_length_buf,
            high=int(env.unwrapped.max_episode_length),
        )
    """

    def __init__(self, env: ReturnTerminalManagerBasedRLEnv, add_terminal_obs=True):
        self._env = env
        self.add_terminal_obs = add_terminal_obs
        # Live visualisers (e.g. value-vs-distance); each gets maybe_update() per step.
        self._visualisers = []

    @property
    def device(self) -> torch.device:
        return self._env.unwrapped.device

    @property
    def unwrapped(self) -> ReturnTerminalManagerBasedRLEnv:
        return self._env.unwrapped

    @property
    def num_envs(self) -> int:
        return self._env.unwrapped.num_envs

    @property
    def max_episode_length(self) -> int:
        return self._env.unwrapped.max_episode_length

    @property
    def action_space(self) -> SpaceInfo:
        return SpaceInfo(shape=self._env.unwrapped.single_action_space.shape)

    @property
    def observation_space(self) -> dict[str, SpaceInfo]:
        raw = self._env.unwrapped.observation_space
        return {k: _convert_space(v) for k, v in raw.items()}

    def step(self, action: torch.Tensor):
        if self.add_terminal_obs:
            result = self._step_terminal_obs(action)
        else:
            result = self._env.step(action)
        for visualiser in self._visualisers:
            visualiser.maybe_update()
        return result

    def reset(self):
        return self._env.reset()

    def close(self):
        return self._env.close()
    
    def randomise_ep_counters(self):
        self._env.unwrapped.episode_length_buf = torch.randint_like(
                self._env.unwrapped.episode_length_buf, high=int(self._env.unwrapped.max_episode_length)
            )

    def _step_terminal_obs(self, action: torch.Tensor):
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        env = self._env.unwrapped
        # process actions
        env.action_manager.process_action(action.to(self.device))

        env.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(env.cfg.decimation):
            env._sim_step_counter += 1
            # set actions into buffers
            env.action_manager.apply_action()
            # set actions into simulator
            env.scene.write_data_to_sim()
            # simulate
            env.sim.step(render=False)
            env.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
                env.sim.render()
            # update buffers at sim dt
            env.scene.update(dt=env.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        env.episode_length_buf += 1  # step in current episode (per env)
        env.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        env.reset_buf = env.termination_manager.compute()
        env.reset_terminated = env.termination_manager.terminated
        env.reset_time_outs = env.termination_manager.time_outs
        # -- reward computation
        env.reward_buf = env.reward_manager.compute(dt=env.step_dt)

        if len(env.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            env.obs_buf = env.observation_manager.compute()
            env.recorder_manager.record_post_step()

        # -- compute pre-reset obs for ALL envs, nan out non-reset rows
        reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        all_pre_reset = env.observation_manager.compute()
        non_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        if len(reset_env_ids) > 0:
            non_reset[reset_env_ids] = False
        env.extras["terminal_obs"] = _fill_nan(all_pre_reset, non_reset)

        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            env.recorder_manager.record_pre_reset(reset_env_ids)

            env._reset_idx(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if env.sim.has_rtx_sensors() and env.cfg.num_rerenders_on_reset > 0:
                for _ in range(env.cfg.num_rerenders_on_reset):
                    env.sim.render()

            # trigger recorder terms for post-reset calls
            env.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        env.command_manager.compute(dt=env.step_dt)
        # -- step interval events
        if "interval" in env.event_manager.available_modes:
            env.event_manager.apply(mode="interval", dt=env.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        env.obs_buf = env.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return env.obs_buf, env.reward_buf, env.reset_terminated, env.reset_time_outs, env.extras
