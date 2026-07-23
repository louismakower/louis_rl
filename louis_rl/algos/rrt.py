from __future__ import annotations
from .base_runner import BaseRunner
from louis_rl.vec_env import SnapshotVecEnv
from dataclasses import dataclass, MISSING
import torch

# init tree with a root node 
# requires a stability function
# also requires a distance function - 
    # this might just have to be euclidean distance for now (but should move to temporal distance later)

# while N < N_max:
    # x_sample = random point in state space
    # x_node = closest point to x_sample currently in the tree
 
    # actions = sample random actions
    # env.step(actions)
    # stable_envs = env.check_stable()
    # distances = env.distance_to(x_sample)
    # distances = torch.where(stable_envs, distances, infinity)
    # best_env = argmin(distances)
    # best_action = actions[best_env]
    # best_state = env.get_state[best_env]

    # add x_new to tree with x_node as the parent and best_action as the step
    # N = N+1

class Graph:
    def __init__(self, root, max_nodes, state_dim, dist_fn, device):
        self.device = device
        self.states = torch.zeros(max_nodes, state_dim, device=device)
        self.parents = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
        self.dist_fn = dist_fn
        self.states[0] = root
        self.size = 1
        self.max_nodes = max_nodes

    @property
    def full(self):
        return self.size >= self.max_nodes

    def add_node(self, state, parent_idx):
        if self.full:
            print("Reached capacity of graph, skipping add")
            return
        
        i = self.size
        self.states[i] = state
        self.parents[i] = parent_idx
        self.size += 1
        return i

    def nearest(self, x_sample):
        dists = self.dist_fn(self.states[:self.size], x_sample)
        idx = dists.argmin()
        return idx.item(), self.states[idx:idx+1]


class RRTRunner(BaseRunner):
    def __init__(self, env: SnapshotVecEnv, cfg: RRTRunnerCfg, log_dir):
        super().__init__(log_dir=log_dir)
        self._env = env
        self.cfg = cfg
        self.device = self._env.device
        self.num_envs = self._env.num_envs
        self.act_dim = self._env.action_space.shape[0]

        self.lims = torch.tensor(self.cfg.env_lims, dtype=torch.float32, device=self.device)

    def learn(self):
        obs, extras = self._env.reset()
        input("press enter")
        self.graph = Graph(
            root=self.obs_to_node(obs)[0], # first env is where we init the tree
            max_nodes=self.cfg.max_nodes,
            state_dim=self.cfg.rrt_dim,
            dist_fn=self.cfg.distance_fn,
            device=self.device,
        )

        lows, highs = self.lims[:, 0], self.lims[:, 1]
        for n in range(self.cfg.N_max):
            # uniform sample within env_lims (one dim per limit row)
            x_sample = lows + torch.rand(1, self.lims.shape[0], device=self.device) * (highs - lows)
            self.x_sample = x_sample  # exposed for the tree visualiser
            idx, x_node = self.graph.nearest(x_sample)
            self._env.restore_state(slice(None), x_node)

            actions = torch.rand(size=(self.num_envs, self.act_dim), device=self.device) * 2 - 1
            next_obs, rew, term, timeout, extras = self._env.step(actions)
            next_nodes = self.obs_to_node(next_obs)
            distances = self.cfg.distance_fn(x_sample, next_nodes)
            stable = self.cfg.is_stable(next_obs)
            resetted = term | timeout
            distances[~stable | resetted] = float('inf')

            min_dist, best_env = distances.min(dim=0)
            if min_dist < float('inf'):
                self.graph.add_node(
                    state=next_nodes[best_env],
                    parent_idx=idx,
                )

    def obs_to_node(self, obs):
        return obs["rnd"]

    def _get_checkpoint(self):
        pass

    def _load_checkpoint(self, state):
        pass

    def get_deterministic_action(self, obs):
        return torch.rand(size=(self.num_envs, self.act_dim), device=self.device)
        

@dataclass
class RRTRunnerCfg:
    experiment_name: str = MISSING
    rrt_dim: int = MISSING
    env_lims: tuple[tuple] = MISSING
    N_max: int = MISSING
    max_nodes: int = MISSING
    is_stable: callable = MISSING
    distance_fn: callable = MISSING
    algo_name: str = "rrt"