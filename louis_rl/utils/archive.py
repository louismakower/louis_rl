from __future__ import annotations
import torch

from louis_rl.utils.grid import CellGrid


class GoExploreArchive:
    def __init__(
        self,
        limits: list[tuple[float, float]],
        resolutions: float | list[float],
        snapshot_dim: int,
        device,
        count_exponent: float = 1.0,  # higher means more aggressively prefer rarely chosen cells when selecting
    ):
        self.device = device
        self.snapshot_dim = snapshot_dim
        
        self.count_exponent = count_exponent
        self.grid = CellGrid(limits, resolutions, device)

        # dense, flat-indexed per-cell storage
        n = self.grid.n_cells
        self.visited = torch.zeros(n, dtype=torch.bool, device=device)
        self.visit_counts = torch.zeros(n, dtype=torch.long, device=device)
        self.snapshots = torch.zeros(n, snapshot_dim, device=device)
        self.best_scores = torch.full((n,), -float("inf"), device=device)
        self.parents = -torch.ones((n,), dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, feats: torch.Tensor, scores: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
        """Insert / refresh cells from a batch of visited states.

        feats: (N, snapshot_dim)  states to discretise into cells 
        scores: (N,)  higher is better; the best-scoring visit to a cell keeps its snapshot and parent
        parents: (N,)  flat cell index of each state's predecessor

        Returns the (N,) flat cell index each state mapped to.
        """
        n = self.grid.n_cells
        num = scores.shape[0]
        cells = self.grid.to_flat(feats)

        # collapse duplicate cells within this batch deterministically
        # for each cell pick the highest-scoring row (break ties by smallest row index)
        batch_best = torch.full((n,), -float("inf"), device=self.device)
        batch_best.scatter_reduce_(0, cells, scores, reduce="amax", include_self=True)
        is_winner = scores == batch_best[cells]
        rows = torch.arange(num, device=self.device)
        tie_break = torch.where(is_winner, rows, torch.full_like(rows, num))
        win_row = torch.full((n,), num, dtype=torch.long, device=self.device)
        win_row.scatter_reduce_(0, cells, tie_break, reduce="amin", include_self=True)

        present = (win_row < num).nonzero(as_tuple=True)[0]  # unique cells seen this batch
        win_row = win_row[present]  # the winning row per cell

        # only overwrite a cell when this batch's winner beats what's stored.
        # upd_cells is unique, so these writes are order-independent.
        better = scores[win_row] >= self.best_scores[present]
        upd_cells = present[better]
        upd_rows = win_row[better]
        self.best_scores[upd_cells] = scores[upd_rows]
        self.snapshots[upd_cells] = feats[upd_rows]
        self.parents[upd_cells] = parents[upd_rows]
        self.visited[present] = True

        # count every visit, including the duplicates collapsed above
        self.visit_counts.scatter_add_(0, cells, torch.ones_like(cells))
        return cells

    @torch.no_grad()
    def select(self, n: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample ``n`` cells to restart from (with replacement).

        Returns ``(snapshots, cells)`` — the restore snapshots, shape
        (n, snapshot_dim), and their flat cell indices, shape (n,) — or ``None``
        if the archive is empty.
        """
        occupied = self.visited.nonzero(as_tuple=True)[0]
        if occupied.numel() == 0:
            return None

        counts = self.visit_counts[occupied].float()
        weights = 1.0 / (1.0 + counts) ** self.count_exponent
        probs = weights / weights.sum()

        chosen = occupied[torch.multinomial(probs, n, replacement=True)]
        # TODO: maybe want to increment visit counts here too?
        return self.snapshots[chosen], chosen

    @property
    def n_visited(self) -> int:
        return int(self.visited.sum())

    @property
    def coverage(self) -> float:
        return self.n_visited / self.grid.n_cells