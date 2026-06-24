from .algos.ppo import PPORunner
from .algos.sac import SACRunner

class RLRunner:
    def __init__(self, env, cfg, log_dir, inference_only=False):
        if cfg.algo_name.lower() == "ppo":
            self.runner = PPORunner(env, cfg, log_dir, inference_only=inference_only)
        elif cfg.algo_name.lower() == "sac":
            self.runner = SACRunner(env, cfg, log_dir, inference_only=inference_only)

    def learn(self):
        self.runner.learn()

    def load_checkpoint(self, path: str):
        self.runner.load_checkpoint(path)

    def get_deterministic_action(self, obs):
        return self.runner.get_deterministic_action(obs)
    
    def add_goal_obs(self, obs):
        return self.runner.add_goal_obs(obs)
