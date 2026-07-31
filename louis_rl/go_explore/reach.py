"""Reaching goals that an explorer already proved are achievable.

Phase 2 of go-explore, expressed entirely as a task rather than as an algorithm. The
explorer archives one state per grid cell it ever reached; the reacher samples those
states as goals and trains a goal-conditioned policy to return to them. Nothing here
is a new RL algorithm, so nothing here subclasses a runner - ``make_reach_runner``
assembles a goal-conditioned MDP and hands it to SAC or PPO unchanged.
"""

from __future__ import annotations
from dataclasses import dataclass

import torch

from louis_rl.algos.ppo import PPORunner, PPORunnerCfg
from louis_rl.algos.sac import SACRunner, SACRunnerCfg
from louis_rl.implementations.goal_env import GoalSource, GoalVecEnv
from louis_rl.implementations.goal_spec import KeyedGoalSpec, _get
from louis_rl.utils.grid import CellGrid
from louis_rl.vec_env import VecEnv


class ArchiveGoalSpec(KeyedGoalSpec):
    """Reach a position, and stay there"""

    def __init__(
            self,
            context_dim: int,
            goal_dim: int,
            goal_radius: float,
            context_key: tuple[str, ...] = ("policy",),
            desired_key: tuple[str, ...] = ("goal",),
            achieved_key: tuple[str, ...] = ("explore",),
    ):
        super().__init__(context_dim, goal_dim, context_key, desired_key, achieved_key)
        self.goal_radius = goal_radius

    def achieved(self, obs: dict) -> torch.Tensor:
        # keep self.goal_dim leading dims of achieved obs
        return _get(obs, self.achieved_key)[..., : self.goal_dim]

    def reward(self, achieved: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # sparse reward here
        return self.success(achieved, goal).float()

    def success(self, achieved: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.distance(achieved, goal) < self.goal_radius


class ArchiveGoalSource(GoalSource):
    """Uniform over archived states"""

    def __init__(self, feats: torch.Tensor):
        self.feats = feats

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.randint(self.feats.shape[0], (n,), device=self.feats.device)
        return self.feats[idx]


def load_archive(path: str, device) -> tuple[torch.Tensor, CellGrid]:
        """Read an ExplorerRunner checkpoint into its states and the grid they were binned on."""
        state = torch.load(path, map_location=device, weights_only=True)
        feats = state["feats"].to(device)
        grid = CellGrid(state["limits"], state["resolutions"], device)
        print(f"Loaded {feats.shape[0]} goals from {path}")
        return feats, grid


def _space_dim(space, path: tuple[str, ...]) -> int:
    """Width of an observation group, matching how goal_spec._get flattens it."""
    for key in path:
        space = space[key]
    if isinstance(space, dict):
        return sum(s.shape[0] for s in space.values())
    return space.shape[0]


def make_reach_runner(env: VecEnv, cfg, log_dir: str, inference_only: bool = False):
    """Assemble the reach task and hand it to the algorithm named by ``cfg.base_algo``."""
    feats, grid = load_archive(cfg.archive_path, env.device)
    spec = ArchiveGoalSpec(
        context_dim=_space_dim(env.observation_space, ("policy",)),
        goal_dim=feats.shape[-1],
        goal_radius=cfg.goal_radius,
    )

    env = GoalVecEnv(
        env,
        spec,
        ArchiveGoalSource(feats),
        log_dir=None if inference_only else log_dir,  # don't litter play runs with events
        grid=grid,
    )
    cfg.goal_spec = spec

    runners = {"sac": SACRunner, "ppo": PPORunner}
    base = cfg.base_algo.lower()
    if base not in runners:
        raise ValueError(f"Unknown base_algo {cfg.base_algo}. Expected one of {sorted(runners)}.")
    return runners[base](env, cfg, log_dir, inference_only=inference_only)


@dataclass
class ReachSACCfg(SACRunnerCfg):
    archive_path: str = ""  # ExplorerRunner checkpoint holding the goals
    goal_radius: float = 0.025  # metres; one archive cell at the default resolution
    base_algo: str = "sac"
    algo_name: str = "reach"


@dataclass
class ReachPPOCfg(PPORunnerCfg):
    archive_path: str = ""
    goal_radius: float = 0.025
    base_algo: str = "ppo"
    algo_name: str = "reach"
