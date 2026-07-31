from abc import ABC, abstractmethod
import os
import torch


class BaseRunner(ABC):
    def __init__(self, log_dir: str):
        self.checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.global_step = 0

    @abstractmethod
    def learn(self): ...

    @abstractmethod
    def get_deterministic_action(self, obs): ...

    @abstractmethod
    def _get_checkpoint(self) -> dict: ...

    @abstractmethod
    def _load_checkpoint(self, state: dict): ...

    def save_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, f"model_{self.global_step}.pth")
        torch.save(self._get_checkpoint(), path)
        print(f"[INFO]: Saved checkpoint at {path}")

    def load_checkpoint(self, path: str):
        state = torch.load(path, weights_only=True)
        self._load_checkpoint(state)