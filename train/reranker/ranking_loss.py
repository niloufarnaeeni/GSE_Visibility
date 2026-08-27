"""The four ranking objectives reported in the final paper."""

from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F


def ranknet(logits: torch.Tensor, labels: torch.Tensor, group_size: int) -> torch.Tensor:
    """RankNet loss (unchanged from the original implementation)."""
    grouped_logits = logits.view(-1, group_size)
    grouped_labels = labels.view(-1, group_size)
    candidates = list(product(range(grouped_labels.shape[1]), repeat=2))
    pairs_true = grouped_labels[:, candidates]
    selected_pred = grouped_logits[:, candidates]
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]
    mask = (true_diffs > 0) & (~torch.isinf(true_diffs))
    return nn.BCEWithLogitsLoss(reduction="mean", weight=torch.abs(true_diffs)[mask])(
        pred_diffs[mask], (true_diffs > 0).type(torch.float32)[mask]
    )


def _safe_abs_pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return absolute Pearson correlation, safely handling degenerate inputs."""
    zero = x.new_zeros(())
    if x.numel() < 2 or y.numel() < 2:
        return zero
    finite_mask = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[finite_mask], y[finite_mask]
    if x.numel() < 2 or y.numel() < 2:
        return zero
    x, y = x.float(), y.float()
    x_centered, y_centered = x - x.mean(), y - y.mean()
    x_std = torch.sqrt(torch.mean(x_centered * x_centered))
    y_std = torch.sqrt(torch.mean(y_centered * y_centered))
    if (not torch.isfinite(x_std)) or (not torch.isfinite(y_std)) or x_std <= eps or y_std <= eps:
        return zero
    return torch.nan_to_num(
        torch.mean(x_centered * y_centered) / (x_std * y_std).clamp_min(eps),
        nan=0.0, posinf=0.0, neginf=0.0,
    ).abs()


def _maybe_scale_prior_attention(prior_attention, scores, eps=1e-8, max_abs_threshold=200.0, ratio_threshold=15.0):
    """Apply the original adaptive prior-attention scaling rule."""
    attention_abs_max = prior_attention.abs().max()
    score_scale = scores.abs().mean(dim=1, keepdim=True).clamp_min(eps)
    attention_scale = prior_attention.abs().mean(dim=1, keepdim=True)
    if not ((attention_abs_max > max_abs_threshold) or ((attention_scale / score_scale).median() > ratio_threshold)):
        return prior_attention
    compressed = torch.log1p(torch.clamp(prior_attention, min=0.0))
    return compressed / compressed.mean(dim=1, keepdim=True).clamp_min(eps)


def _group_ranks_from_labels(labels: torch.Tensor) -> torch.Tensor:
    batch_size, group_size = labels.shape
    order = torch.argsort(labels, dim=1, descending=True)
    ranks = torch.empty((batch_size, group_size), device=labels.device, dtype=torch.float32)
    positions = torch.arange(1, group_size + 1, device=labels.device, dtype=torch.float32)
    ranks.scatter_(1, order, positions.unsqueeze(0).expand(batch_size, -1))
    return ranks


def _ear(logits, labels, prior_attention, group_size, mode, lambda_prior_attention=0.3):
    """Shared, unchanged EAR implementation for the negative and symmetric variants."""
    group_size = int(group_size)
    assert logits.numel() % group_size == 0, "logits length must be divisible by group_size"
    batch_size = logits.numel() // group_size
    scores = logits.view(batch_size, group_size).float()
    relevance = labels.view(batch_size, group_size).float()
    attention = _maybe_scale_prior_attention(prior_attention.view(batch_size, group_size).float(), scores)
    ranks = _group_ranks_from_labels(relevance)
    score_i, score_j = scores.unsqueeze(2), scores.unsqueeze(1)
    label_i, label_j = relevance.unsqueeze(2), relevance.unsqueeze(1)
    mask = label_i > label_j
    weights = (ranks.unsqueeze(1) - ranks.unsqueeze(2)).abs()
    attention_i, attention_j = attention.unsqueeze(2), attention.unsqueeze(1)
    if mode == "neg":
        adjusted_difference = (score_i - score_j) - lambda_prior_attention * attention_j
    elif mode == "both":
        adjusted_difference = (score_i - score_j) - lambda_prior_attention * attention_i - lambda_prior_attention * attention_j
    else:
        raise ValueError(f"Unknown EAR mode: {mode}")
    per_pair = weights * F.softplus(-adjusted_difference) * mask.to(scores.dtype)
    return per_pair.sum() / mask.sum().clamp_min(1).to(per_pair.dtype)


def ear(logits, labels, prior_attention, group_size, lambda_prior_attention: float = 0.3):
    """EAR: the original negative-side prior-attention objective."""
    return _ear(logits, labels, prior_attention, group_size, "neg", lambda_prior_attention)


def ear_sym(logits, labels, prior_attention, group_size, lambda_prior_attention: float = 0.3):
    """EAR-Sym: the original symmetric prior-attention objective."""
    return _ear(logits, labels, prior_attention, group_size, "both", lambda_prior_attention)


def pairwise_reg(logits, labels, prior_attention, group_size, lambda_corr: float = 0.05):
    """Pairwise Reg, mathematically identical to the prior correlation objective."""
    base_loss = ranknet(logits, labels, group_size)
    group_size = int(group_size)
    assert logits.numel() % group_size == 0, "logits length must be divisible by group_size"
    scores = logits.view(-1, group_size).float()
    relevance = labels.view(-1, group_size).float()
    attention_log = torch.log1p(torch.clamp(prior_attention.view(-1, group_size).float(), min=0.0))
    score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    label_diff = relevance.unsqueeze(2) - relevance.unsqueeze(1)
    attention_diff = attention_log.unsqueeze(2) - attention_log.unsqueeze(1)
    valid = torch.isfinite(relevance)
    upper = torch.triu(torch.ones((group_size, group_size), device=scores.device, dtype=torch.bool), diagonal=1).unsqueeze(0)
    pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1) & upper & (label_diff != 0) & torch.isfinite(label_diff)
    if not pair_mask.any():
        return base_loss
    return base_loss + float(lambda_corr) * _safe_abs_pearson_corr(
        (score_diff * label_diff)[pair_mask], (attention_diff * label_diff)[pair_mask]
    )
