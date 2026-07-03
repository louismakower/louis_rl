from __future__ import annotations
import torch
from typing import Callable

from louis_rl.utils.grid import CellGrid


def norm_threshold_stability(dims: list[int], threshold: float) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    Default stability: norm(features[:, dims]) < threshold.
    Returns a callable suitable for the StableArchive. Higher score is better.
    """
    dims_t = torch.tensor(dims, dtype=torch.long)

    def fn(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        metric = features[:, dims_t.to(features.device)].norm(dim=-1)
        return -metric, metric < threshold

    return fn


class StableArchive:
    """
    Two decoupled per-cell stores, based on the same CellGrid:

     - stable_snapshots / best_score: best snapshot per cell by stability_fn
       score
     - reset_snapshots / visited: most recent snapshot per cell, from any visit,
       shat select() draws from. this is indep. of stability

     - features is used for grid binning and stability
     - snapshot is used for simulator restores - env.restore_state()
    """

    def __init__(
        self,
        limits: list[tuple[float, float]],
        resolutions: float | list[float],
        snapshot_dim: int,
        device,
        stability_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        count_exponent: float = 1.0,  # higher -> more aggressively favour rarely-visited cells
    ):
        self.device = device
        self.snapshot_dim = snapshot_dim
        self.stability_fn = stability_fn
        self.count_exponent = count_exponent
        self.grid = CellGrid(limits, resolutions, device)

        n = self.grid.n_cells
        self.stable_counts = torch.ones(n, dtype=torch.int32, device=device)  # stable visits, used for intrinsic reward
        self.visit_counts = torch.ones(n, dtype=torch.int32, device=device)  # every visit, used for reward's dense term + reset weighting
        self.best_score = torch.full((n,), -float("inf"), device=device)  # higher is better
        self.stable_snapshots = torch.zeros(n, snapshot_dim, device=device)
        self.visited = torch.zeros(n, dtype=torch.bool, device=device)
        self.reset_snapshots = torch.zeros(n, snapshot_dim, device=device)

    @torch.no_grad()
    def observe(self, features: torch.Tensor, snapshot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Insert / refresh cells from a batch of visited states.

        features: (N, F) - used for binning
        snapshot: (N, snapshot_dim) - used for env restore

        Every visit updates visit_counts / reset_snapshots.
        Stable visits (per stability_fn) also update stable_counts / stable_snapshots.

        Returns (cells, stable)
        """
        cells = self.grid.to_flat(features[:, : self.grid.n_bin])
        self.visit_counts.scatter_add_(0, cells, torch.ones_like(cells, dtype=self.visit_counts.dtype))
        self._update_reset_snapshots(cells, snapshot)

        score, stable = self.stability_fn(features)
        if bool(stable.any()):
            s_cells, s_score, s_snapshot = cells[stable], score[stable], snapshot[stable]
            self.stable_counts.scatter_add_(0, s_cells, torch.ones_like(s_cells, dtype=self.stable_counts.dtype))
            self._archive(s_cells, s_score, s_snapshot)
        return cells, stable

    def _update_reset_snapshots(self, cells: torch.Tensor, snapshot: torch.Tensor):
        # any visit counts - keep the most recent, deduped by highest row index
        n, num = self.grid.n_cells, snapshot.shape[0]
        rows = torch.arange(num, device=self.device)
        win_row = torch.full((n,), -1, dtype=torch.long, device=self.device)
        win_row.scatter_reduce_(0, cells, rows, reduce="amax", include_self=True)
        present = (win_row >= 0).nonzero(as_tuple=True)[0]
        self.reset_snapshots[present] = snapshot[win_row[present]]
        self.visited[present] = True

    def _archive(self, cells: torch.Tensor, score: torch.Tensor, snapshot: torch.Tensor):
        # collapse duplicate cells within this batch deterministically: for each
        # cell pick the highest-scoring row (break ties by smallest row index),
        # then overwrite the stored best only if this batch's winner beats it.
        n, num = self.grid.n_cells, snapshot.shape[0]
        batch_best = torch.full((n,), -float("inf"), device=self.device)
        batch_best.scatter_reduce_(0, cells, score, reduce="amax", include_self=True)
        is_winner = score == batch_best[cells]
        rows = torch.arange(num, device=self.device)
        tie_break = torch.where(is_winner, rows, torch.full_like(rows, num))
        win_row = torch.full((n,), num, dtype=torch.long, device=self.device)
        win_row.scatter_reduce_(0, cells, tie_break, reduce="amin", include_self=True)

        present = (win_row < num).nonzero(as_tuple=True)[0]  # unique cells seen this batch
        win_row = win_row[present]

        # upd_cells is unique, so these writes are order-independent
        better = score[win_row] >= self.best_score[present]
        upd_cells, upd_rows = present[better], win_row[better]
        self.best_score[upd_cells] = score[upd_rows]
        self.stable_snapshots[upd_cells] = snapshot[upd_rows]

    @torch.no_grad()
    def reward(self, cells: torch.Tensor, stable: torch.Tensor, ungated_weight: float = 0.0) -> torch.Tensor:
        """
        Reward for reaching a stable and novel state.
        Optional small term for reaching a novel but unstable state
        Returns (N, 1)
        """
        rew = stable.float() / torch.sqrt(self.stable_counts[cells].float())
        if ungated_weight > 0:
            rew = rew + ungated_weight / torch.sqrt(self.visit_counts[cells].float())
        return rew.unsqueeze(-1)

    @torch.no_grad()
    def select(self, n: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample n cells to restart from (with replacement), weighted by inverse
        visit count. Draws from any visited cell, independent of stability.

        Returns (snapshots, cells), shapes (n, snapshot_dim), (n,) or None
        if the archive is empty.
        """
        visited = self.visited.nonzero(as_tuple=True)[0]
        if visited.numel() == 0:
            return None

        counts = self.visit_counts[visited].float()
        weights = 1.0 / counts ** self.count_exponent
        probs = weights / weights.sum()

        chosen = visited[torch.multinomial(probs, n, replacement=True)]
        return self.reset_snapshots[chosen], chosen

    @property
    def occupied(self) -> torch.Tensor:
        return self.best_score > -float("inf")

    @property
    def n_stable(self) -> int:
        return int(self.occupied.sum())

    @property
    def coverage(self) -> float:
        return self.n_stable / self.grid.n_cells  # denominator includes unreachable cells

    @property
    def n_visited(self) -> int:
        return int(self.visited.sum())

    @property
    def visited_coverage(self) -> float:
        return self.n_visited / self.grid.n_cells  # denominator includes unreachable cells
