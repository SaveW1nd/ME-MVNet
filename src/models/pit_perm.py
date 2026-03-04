"""Permutation helpers for 3-source PIT matching."""

from __future__ import annotations

import itertools

import torch

from .sisdr import si_sdr


PERMUTATIONS_3 = list(itertools.permutations([0, 1, 2]))


def pairwise_sep_cost(
    j_hat: torch.Tensor,
    j_true: torch.Tensor,
    nf_true: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise source matching costs.

    Args:
        j_hat: Predicted jammer sources, shape (B,3,2,N).
        j_true: Ground-truth jammer sources, shape (B,3,2,N).
        nf_true: Ground-truth NF labels, shape (B,3), values in {0,1,2,3}.

    Returns:
        Pairwise cost matrix, shape (B,3,3), where [b,k,j] is cost
        for matching predicted slot k to true slot j.
    """
    if j_hat.shape != j_true.shape:
        raise ValueError(f"Shape mismatch: {tuple(j_hat.shape)} vs {tuple(j_true.shape)}")
    if j_hat.ndim != 4 or j_hat.shape[1] != 3 or j_hat.shape[2] != 2:
        raise ValueError(f"Expected (B,3,2,N), got {tuple(j_hat.shape)}")
    if nf_true.shape != (j_hat.shape[0], 3):
        raise ValueError(f"Expected nf_true shape (B,3), got {tuple(nf_true.shape)}")

    bsz = j_hat.shape[0]
    costs = torch.zeros((bsz, 3, 3), device=j_hat.device, dtype=j_hat.dtype)
    for k in range(3):
        for j in range(3):
            est = j_hat[:, k, :, :]
            ref = j_true[:, j, :, :]
            active = (nf_true[:, j] > 0).to(j_hat.dtype)

            active_cost = -si_sdr(est, ref)  # better -> smaller cost
            silence_cost = torch.mean(est * est, dim=(1, 2))
            costs[:, k, j] = active * active_cost + (1.0 - active) * silence_cost
    return costs


def best_perm_from_pairwise(pair_cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the best permutation from pairwise costs.

    Args:
        pair_cost: Tensor (B,3,3), cost[k,j] for pred-k vs true-j.

    Returns:
        best_perm: Long tensor (B,3), mapping pred slot -> true slot.
        best_cost: Tensor (B,), total minimum matching cost.
    """
    if pair_cost.ndim != 3 or pair_cost.shape[1:] != (3, 3):
        raise ValueError(f"Expected pair_cost shape (B,3,3), got {tuple(pair_cost.shape)}")

    bsz = pair_cost.shape[0]
    perm_costs = []
    for perm in PERMUTATIONS_3:
        # sum_k cost[k, perm[k]]
        cost = pair_cost[:, 0, perm[0]] + pair_cost[:, 1, perm[1]] + pair_cost[:, 2, perm[2]]
        perm_costs.append(cost.unsqueeze(1))
    all_costs = torch.cat(perm_costs, dim=1)  # (B,6)
    best_idx = torch.argmin(all_costs, dim=1)  # (B,)
    best_cost = all_costs[torch.arange(bsz, device=pair_cost.device), best_idx]

    perm_table = torch.tensor(PERMUTATIONS_3, device=pair_cost.device, dtype=torch.long)  # (6,3)
    best_perm = perm_table[best_idx]  # (B,3)
    return best_perm, best_cost


def align_true_by_perm(true_tensor: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Align true tensor by best permutation.

    Args:
        true_tensor: Tensor with shape (B,3,...) to align.
        perm: Long tensor (B,3), pred slot -> true slot.

    Returns:
        Aligned tensor with shape (B,3,...), so index-1 aligns with pred slot.
    """
    if true_tensor.ndim < 2 or true_tensor.shape[1] != 3:
        raise ValueError(f"Expected true_tensor shape (B,3,...), got {tuple(true_tensor.shape)}")
    if perm.shape != (true_tensor.shape[0], 3):
        raise ValueError(f"Expected perm shape (B,3), got {tuple(perm.shape)}")

    gather_idx = perm
    for _ in range(true_tensor.ndim - 2):
        gather_idx = gather_idx.unsqueeze(-1)
    gather_idx = gather_idx.expand(*true_tensor.shape[:2], *true_tensor.shape[2:])
    return torch.gather(true_tensor, dim=1, index=gather_idx)
