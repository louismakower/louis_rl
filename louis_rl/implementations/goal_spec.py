from __future__ import annotations
from abc import ABC, abstractmethod
import torch


def _get(obs: dict, path: tuple[str, ...]) -> torch.Tensor:
    for key in path:
        obs = obs[key]
    if isinstance(obs, dict):
        return torch.cat(list(obs.values()), dim=-1)
    return obs


class GoalSpec(ABC):
    """Everything the algorithm needs to know about a goal-conditioned task.

    One object owns the whole contract, so its parts cannot drift apart:

      - which part of an observation is goal-independent      (context)
      - which goal is currently being pursued                 (desired)
      - which goal a state actually attains                   (achieved)
      - what attaining one goal while pursuing another earns  (reward)
      - how context and goal become a network input           (encode)

    ``encode`` is the only place the network's input layout is defined, so hindsight
    relabelling is ``encode(context, other_goal)`` rather than splicing a flat
    observation at a remembered offset. Likewise ``reward`` is defined once and called
    by both the rollout and the relabeller, so a real transition and a relabelled one
    can never disagree about what a reward is.
    """

    goal_dim: int
    context_dim: int
    terminate_on_success: bool = True

    @abstractmethod
    def context(self, obs: dict) -> torch.Tensor:
        """(N, C) the part of the observation that does not depend on the goal."""

    @abstractmethod
    def desired(self, obs: dict) -> torch.Tensor:
        """(N, G) the goal being pursued in this observation."""

    @abstractmethod
    def achieved(self, obs: dict) -> torch.Tensor:
        """(N, G) the goal this state actually attains."""

    @abstractmethod
    def reward(self, achieved: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """(N,) reward for attaining ``achieved`` while pursuing ``goal``.

        Called with real goals during rollout and hindsight goals during relabelling,
        so it must be a pure function of its two arguments.
        """

    @abstractmethod
    def success(self, achieved: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """(N,) bool - whether ``achieved`` counts as reaching ``goal``."""

    def distance(self, achieved: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """(N,) how far ``achieved`` is from ``goal``. Logging only."""
        return torch.norm(achieved - goal, dim=-1)

    def encode(self, context: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """(..., C + G) network input. The single definition of the observation layout."""
        return torch.cat([context, goal], dim=-1)

    def encode_obs(self, obs: dict) -> torch.Tensor:
        """Encode an observation against the goal it is already pursuing."""
        return self.encode(self.context(obs), self.desired(obs))

    @property
    def obs_dim(self) -> int:
        return self.context_dim + self.goal_dim


class KeyedGoalSpec(GoalSpec):
    """GoalSpec for envs that publish goal information as named observation groups.

    Key paths are given explicitly, so adding an observation group can't silently
    change the network input. Subclasses supply ``reward`` and ``success``.
    """

    def __init__(
        self,
        context_dim: int,
        goal_dim: int,
        context_key: tuple[str, ...] = ("policy",),
        desired_key: tuple[str, ...] = ("goal",),
        achieved_key: tuple[str, ...] = ("her", "position"),
    ):
        self.context_dim = context_dim
        self.goal_dim = goal_dim
        self.context_key = context_key
        self.desired_key = desired_key
        self.achieved_key = achieved_key

    def context(self, obs: dict) -> torch.Tensor:
        return _get(obs, self.context_key)

    def desired(self, obs: dict) -> torch.Tensor:
        return _get(obs, self.desired_key)

    def achieved(self, obs: dict) -> torch.Tensor:
        return _get(obs, self.achieved_key)


class PlainSpec(GoalSpec):
    """The degenerate, non-goal-conditioned case: the observation is the whole input.

    Lets an un-goal-conditioned task travel the same code path with goal_dim == 0,
    rather than the runner branching on whether goals exist.
    """

    goal_dim = 0
    terminate_on_success = False

    def __init__(self, observation_space: dict):
        self.context_keys = [k for k in observation_space if "policy" in k]
        self.context_dim = sum(observation_space[k].shape[0] for k in self.context_keys)

    def context(self, obs: dict) -> torch.Tensor:
        return torch.cat([obs[k] for k in self.context_keys], dim=-1)

    def desired(self, obs: dict) -> torch.Tensor:
        ctx = self.context(obs)
        return ctx.new_zeros((*ctx.shape[:-1], 0))

    def achieved(self, obs: dict) -> torch.Tensor:
        raise NotImplementedError("PlainSpec has no goals; HER cannot be used with it")

    def reward(self, achieved, goal):
        raise NotImplementedError("PlainSpec has no goals; the env's reward is used")

    def success(self, achieved, goal):
        raise NotImplementedError("PlainSpec has no goals")