"""Isolated PBiLoss objective."""

import torch
import torch.nn.functional as F

from rag_retrieval.baseline.pbiloss.sampler import sample_popneg_ft_pairs
from rag_retrieval.train.reranker.ranking_loss import ranknet


def pbiloss_popneg_ft(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prior_attention: torch.Tensor,
    group_size: int,
    baseline_config: dict,
) -> torch.Tensor:
    ranknet_loss = ranknet(logits, labels, group_size)
    pairs = sample_popneg_ft_pairs(
        labels=labels,
        prior_attention=prior_attention,
        group_size=group_size,
        baseline_config=baseline_config,
    )
    if not pairs:
        return ranknet_loss

    relevant_low = torch.tensor(
        [pair[0] for pair in pairs],
        device=logits.device,
        dtype=torch.long,
    )
    popular_irrelevant = torch.tensor(
        [pair[1] for pair in pairs],
        device=logits.device,
        dtype=torch.long,
    )
    pbi_loss = F.softplus(
        logits[popular_irrelevant].float() - logits[relevant_low].float()
    ).mean()
    return ranknet_loss + float(baseline_config["lambda"]) * pbi_loss
