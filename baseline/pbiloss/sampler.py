from typing import Dict, List, Tuple

import torch


def sample_popneg_ft_pairs(
    labels: torch.Tensor,
    prior_attention: torch.Tensor,
    group_size: int,
    baseline_config: Dict,
) -> List[Tuple[int, int]]:
    """
    Uniformly sample one (relevant low-prior-attention, irrelevant high-prior-attention) pair per group.
    """
    threshold = float(baseline_config["resolved"]["prior_attention_threshold"])

    grouped_labels = labels.view(-1, group_size).float()
    grouped_prior_attention = prior_attention.view(-1, group_size).float()

    pairs: List[Tuple[int, int]] = []
    for group_idx in range(grouped_labels.shape[0]):
        group_labels = grouped_labels[group_idx]
        group_prior_attention = grouped_prior_attention[group_idx]
        finite = torch.isfinite(group_labels) & torch.isfinite(group_prior_attention)

        relevant_low = torch.nonzero(
            finite & (group_labels > 0) & (group_prior_attention < threshold),
            as_tuple=False,
        ).view(-1)
        irrelevant_high = torch.nonzero(
            finite & (group_labels == 0) & (group_prior_attention >= threshold),
            as_tuple=False,
        ).view(-1)

        if relevant_low.numel() == 0 or irrelevant_high.numel() == 0:
            continue

        rel_offset = torch.randint(
            relevant_low.numel(),
            (1,),
            device=grouped_labels.device,
        ).item()
        neg_offset = torch.randint(
            irrelevant_high.numel(),
            (1,),
            device=grouped_labels.device,
        ).item()

        base = group_idx * group_size
        pairs.append(
            (
                base + int(relevant_low[rel_offset].item()),
                base + int(irrelevant_high[neg_offset].item()),
            )
        )

    return pairs
