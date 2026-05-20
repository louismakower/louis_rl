from __future__ import annotations
import torch


class HERCfg():
    def get_hindsight_transitions(self, trajectories: dict[str: torch.Tensor]) -> dict[str: torch.Tensor]:
        """
        1. Use other trajectories to get hindsight goal states
        2. Compute rewards with these goal states and return so they can
           be added to the buffer

        Args:
            trajectories - dictionary containing a rollout of transitions across all envs
                           where each element of the dictionary is obs, goal, etc
                           and each tensor has shape (num_steps, num_envs, element_size)
                           where element size is determied by the type, eg obs_size, goal_size etc
        
        Returns:
            dict         - returns a dictionary of each item to be added to the buffer
                         - each element in the dictionary is of shape (num_envs * num_goals_sampled_per_env, element_size)
        """
        raise NotImplementedError()