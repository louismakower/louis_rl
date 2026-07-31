from .algos.ppo import PPORunner
from .algos.sac import SACRunner
from .algos.rrt import RRTRunner
from .go_explore.explore import ExplorerRunner
from .go_explore.reach import make_reach_runner

class RLRunner:
    def __init__(self, env, cfg, log_dir, inference_only=False):
        if cfg.algo_name.lower() == "ppo":
            self.runner = PPORunner(env, cfg, log_dir, inference_only=inference_only)
        elif cfg.algo_name.lower() == "sac":
            self.runner = SACRunner(env, cfg, log_dir, inference_only=inference_only)
        elif cfg.algo_name.lower() == "rrt":
            self.runner = RRTRunner(env, cfg, log_dir)
        elif cfg.algo_name.lower() == "explore":
            self.runner = ExplorerRunner(env, cfg, log_dir)
        elif cfg.algo_name.lower() == "reach":
            self.runner = make_reach_runner(env, cfg, log_dir, inference_only=inference_only)
        else:
            raise ValueError(f"Unknown algo_name {cfg.algo_name!r}")

    def learn(self):
        self.runner.learn()

    def load_checkpoint(self, path: str):
        self.runner.load_checkpoint(path)

    def get_deterministic_action(self, obs):
        return self.runner.get_deterministic_action(obs)
    
    def encode_obs(self, obs):
        return self.runner.spec.encode_obs(obs)
