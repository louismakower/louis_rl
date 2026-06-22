from __future__ import annotations
import torch


class CellGrid:
    """Discretises continuous features into a fixed grid of cells."""

    def __init__(
        self,
        limits: list[tuple[float, float]],
        resolutions: float | list[float],
        device,
    ):
        self.device = device
        dists = [hi - lo for lo, hi in limits]
        assert all(d > 0 for d in dists), "limits must be (min, max)"
        eps = 1e-9
        if isinstance(resolutions, (int, float)):
            resolutions = [float(resolutions)] * len(limits)
        self.resolutions = resolutions
        self.mins = [lo for lo, _ in limits]

        self.shape = tuple(int(d // (r - eps)) for d, r in zip(dists, resolutions))
        # right-hand cell boundaries per dim, used by torch.bucketize
        self.upper_lims = [
            lo + torch.arange(1, n + 1, device=device, dtype=torch.float32) * r
            for lo, r, n in zip(self.mins, resolutions, self.shape)
        ]
        self.n_cells = 1
        for n in self.shape:
            self.n_cells *= n
        print(f"CellGrid shape: {self.shape} ({self.n_cells} cells)")

    def to_index(self, feats: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """(N, n_dims) -> tuple of per-dim (N,) indices, for N-D tensor indexing."""
        indices = []
        for d, boundaries in enumerate(self.upper_lims):
            idx = torch.bucketize(feats[:, d].contiguous(), boundaries)
            idx = idx.clamp(0, self.shape[d] - 1)
            indices.append(idx)
        return tuple(indices)

    def to_flat(self, feats: torch.Tensor) -> torch.Tensor:
        """(N, n_dims) -> (N,) row-major flat cell index in [0, n_cells)."""
        flat = torch.zeros(feats.shape[0], dtype=torch.long, device=self.device)
        for d, idx in enumerate(self.to_index(feats)):
            flat = flat * self.shape[d] + idx
        return flat