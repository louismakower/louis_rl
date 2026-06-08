# based off RL Games
# same logic, just generalises to abtirary items in the buffer
# used to store different sets of observations, eg HER or RND observations

import torch

class VectorizedReplayBuffer:
    def __init__(self, tensor_desc, capacity, device):
        """Create Vectorized Replay buffer.
        Parameters
        ----------
        tensors: dict[str, tuple[int, dtype]]
            description of tensors to store in the buffer
            eg:
              {
                "obs": (17, torch.float32),
                "next_obs": (17, torch.float32),
                "rew": (1, torch.float32),
                "dones": (1, torch.bool),
            }
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        See Also
        --------
        ReplayBuffer.__init__
        """

        self.device = device
        self.num_tensors = len(tensor_desc)
        self.tensor_names = set(tensor_desc.keys())
        assert self.num_tensors == len(self.tensor_names), "Duplicate tensor names were provided"

        for name, (shape, dtype) in tensor_desc.items():
            assert isinstance(name, str)
            assert isinstance(shape, int)
            assert isinstance(dtype, torch.dtype)
            setattr(self, name, torch.empty((capacity, shape), dtype=dtype, device=self.device))

        self.capacity = capacity
        self.idx = 0
        self.full = False
        

    def add(self, tensors: dict[str, torch.Tensor]):
        assert len(tensors) == self.num_tensors, f"Buffer contains {self.num_tensors} but {len(tensors)} added"
        assert set(tensors.keys()) == self.tensor_names, f"Buffer tensor names doesn't match tensors added to the buffer"

        num_observations = next(iter(tensors.values())).shape[0]
        remaining_capacity = min(self.capacity - self.idx, num_observations)
        overflow = num_observations - remaining_capacity
        if remaining_capacity < num_observations:
            for t in self.tensor_names:
                buf_t = getattr(self, t)
                add_t = tensors[t]
                buf_t[0: overflow] = add_t[-overflow:]
            self.full = True
        for t in self.tensor_names:
            buf_t = getattr(self, t)
            add_t = tensors[t]
            buf_t[self.idx: self.idx + remaining_capacity] = add_t[:remaining_capacity]

        self.idx = (self.idx + num_observations) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        sampled:
            dict[str, torch.Tensor]
        """

        idxs = torch.randint(0,
                            self.capacity if self.full else self.idx, 
                            (batch_size,), device=self.device)
        sampled = {t: getattr(self, t)[idxs] for t in self.tensor_names}

        return sampled
