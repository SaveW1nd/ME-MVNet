from __future__ import annotations

import torch

from src.models.pit_perm import align_true_by_perm, best_perm_from_pairwise


def test_best_perm_from_pairwise() -> None:
    # Batch size = 2, shape (B,3,3)
    # sample0 best perm is (0,1,2), sample1 best perm is (2,0,1)
    cost = torch.tensor(
        [
            [[0.1, 2.0, 3.0], [2.0, 0.1, 3.0], [2.0, 3.0, 0.2]],
            [[2.0, 3.0, 0.1], [0.1, 2.0, 3.0], [3.0, 0.1, 2.0]],
        ],
        dtype=torch.float32,
    )
    perm, best = best_perm_from_pairwise(cost)
    assert perm.shape == (2, 3)
    assert best.shape == (2,)
    assert torch.equal(perm[0], torch.tensor([0, 1, 2]))
    assert torch.equal(perm[1], torch.tensor([2, 0, 1]))


def test_align_true_by_perm() -> None:
    true = torch.tensor(
        [
            [[10, 11], [20, 21], [30, 31]],
            [[40, 41], [50, 51], [60, 61]],
        ],
        dtype=torch.float32,
    )  # (2,3,2)
    perm = torch.tensor([[2, 0, 1], [1, 2, 0]], dtype=torch.long)
    aligned = align_true_by_perm(true, perm)
    assert aligned.shape == true.shape
    assert torch.equal(aligned[0, 0], true[0, 2])
    assert torch.equal(aligned[0, 1], true[0, 0])
    assert torch.equal(aligned[1, 2], true[1, 0])
