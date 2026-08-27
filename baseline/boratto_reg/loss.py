import torch
import torch.nn.functional as F


def _safe_abs_pearson_corr(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return a safe absolute Pearson correlation."""

    zero = x.new_zeros(())

    if x.numel() < 2 or y.numel() < 2:
        return zero

    finite_mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if x.numel() < 2 or y.numel() < 2:
        return zero

    x = x.float()
    y = y.float()

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    x_std = torch.sqrt(torch.mean(x_centered * x_centered))
    y_std = torch.sqrt(torch.mean(y_centered * y_centered))

    if (not torch.isfinite(x_std)) or (not torch.isfinite(y_std)):
        return zero

    if x_std <= eps or y_std <= eps:
        return zero

    corr = (
        torch.mean(x_centered * y_centered)
        / (x_std * y_std).clamp_min(eps)
    )

    corr = torch.nan_to_num(
        corr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return corr.abs()


def pairwise_boratto_reg_prior_attention(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prior_attention: torch.Tensor,
    group_size: int,
    lambda_corr: float = 0.2,
) -> torch.Tensor:
    """
    Boratto-reg objective recovered from the previous repository version.

    The objective combines the mean RankNet pair loss with an absolute
    Pearson correlation penalty between each valid pair loss and the
    preferred creator's raw prior attention.

    Only pairs satisfying label_i > label_j are used.

    Final loss:

        (1 - lambda_corr) * rank_loss
        + lambda_corr * corr_loss
    """

    lambda_corr = float(lambda_corr)

    if not 0.0 <= lambda_corr <= 1.0:
        raise ValueError(
            f"lambda_corr must be within [0, 1], got {lambda_corr}"
        )

    group_size = int(group_size)

    if group_size <= 0:
        raise ValueError(
            f"group_size must be positive, got {group_size}"
        )

    if logits.numel() % group_size != 0:
        raise ValueError(
            f"logits length {logits.numel()} is not divisible by "
            f"group_size {group_size}"
        )

    batch_groups = logits.numel() // group_size

    scores = logits.view(batch_groups, group_size).float()
    relevance = labels.view(batch_groups, group_size).float()

    raw_prior_attention = prior_attention.view(
        batch_groups,
        group_size,
    ).float()

    score_i = scores.unsqueeze(2)
    score_j = scores.unsqueeze(1)

    relevance_i = relevance.unsqueeze(2)
    relevance_j = relevance.unsqueeze(1)

    preferred_prior_attention = (
        raw_prior_attention
        .unsqueeze(2)
        .expand(-1, -1, group_size)
    )

    valid = torch.isfinite(score_i) & torch.isfinite(score_j)
    valid = (
        valid
        & torch.isfinite(relevance_i)
        & torch.isfinite(relevance_j)
    )
    valid = valid & torch.isfinite(preferred_prior_attention)

    pair_mask = valid & (relevance_i > relevance_j)

    pair_loss = F.softplus(
        -(score_i - score_j)
    )[pair_mask]

    preferred_attention = preferred_prior_attention[pair_mask]

    if pair_loss.numel() == 0:
        return logits.float().sum() * 0.0

    rank_loss = pair_loss.mean()
    zero_corr = pair_loss.sum() * 0.0

    if (
        pair_loss.numel() < 2
        or preferred_attention.numel() < 2
    ):
        corr_loss = zero_corr

    else:
        pair_loss_float = pair_loss.float()
        attention_float = preferred_attention.float()

        pair_centered = (
            pair_loss_float
            - pair_loss_float.mean()
        )

        attention_centered = (
            attention_float
            - attention_float.mean()
        )

        pair_std = torch.sqrt(
            torch.mean(pair_centered * pair_centered)
        )

        attention_std = torch.sqrt(
            torch.mean(
                attention_centered
                * attention_centered
            )
        )

        if (
            (not torch.isfinite(pair_std))
            or (not torch.isfinite(attention_std))
            or pair_std <= 1e-8
            or attention_std <= 1e-8
        ):
            corr_loss = zero_corr

        else:
            corr_loss = _safe_abs_pearson_corr(
                pair_loss,
                preferred_attention,
            )

    return (
        (1.0 - lambda_corr) * rank_loss
        + lambda_corr * corr_loss
    )