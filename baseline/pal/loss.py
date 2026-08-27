import torch
import torch.nn.functional as F


def pairwise_pal_prior_attention(logits, labels, prior_attention, group_size, baseline_config):
    G = int(group_size)
    assert logits.numel() % G == 0, (
        f"logits length {logits.numel()} not divisible by group_size {G}"
    )

    s = logits.view(-1, G).float()
    y = labels.view(-1, G).float()
    raw_prior_attention = prior_attention.view(-1, G).float()

    resolved = baseline_config["resolved"]
    train_min = float(resolved["train_log_prior_attention_min"])
    train_max = float(resolved["train_log_prior_attention_max"])
    alpha = float(baseline_config.get("alpha", 1.0))

    s_i = s.unsqueeze(2)
    s_j = s.unsqueeze(1)
    y_i = y.unsqueeze(2)
    y_j = y.unsqueeze(1)
    prior_attention_i = raw_prior_attention.unsqueeze(2).expand(-1, -1, G)

    valid = torch.isfinite(s_i) & torch.isfinite(s_j)
    valid = valid & torch.isfinite(y_i) & torch.isfinite(y_j)
    valid = valid & torch.isfinite(prior_attention_i) & (prior_attention_i >= 0.0)
    pair_mask = valid & (y_i > y_j)

    if not pair_mask.any():
        return logits.float().sum() * 0.0

    label_weight = torch.abs(y_i - y_j)
    pair_loss = F.softplus(-(s_i - s_j))

    log_prior_attention_i = torch.log1p(prior_attention_i)
    denom = train_max - train_min
    if denom == 0.0:
        normalized = torch.zeros_like(log_prior_attention_i)
    else:
        normalized = (log_prior_attention_i - train_min) / denom
        normalized = torch.clamp(normalized, 0.0, 1.0)

    pal_weight = torch.exp(-alpha * normalized)
    weighted_pair_loss = label_weight * pal_weight * pair_loss
    return weighted_pair_loss[pair_mask].mean()
