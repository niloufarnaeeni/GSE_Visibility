import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


CREATOR_ID_KEYS = ("creator_id", "creator_code", "creator")
PRIOR_ATTENTION_KEYS = ("prior_attention",)
CREATOR_CONTENT_RE = re.compile(r"\bCreator\s+(C\d+)\b", re.IGNORECASE)


def _first_present(record: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Any]:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _finite_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _creator_id_from_hit(hit: Dict[str, Any]) -> Optional[str]:
    creator_id = _first_present(hit, CREATOR_ID_KEYS)
    if creator_id is not None:
        return str(creator_id).strip()

    content = hit.get("content")
    if isinstance(content, str):
        match = CREATOR_CONTENT_RE.search(content)
        if match:
            return match.group(1)

    return None


def resolve_training_prior_attention_threshold(
    train_jsonl: str | Path,
    popular_fraction: float = 0.20,
) -> Dict[str, Any]:
    """Resolve one global raw-prior-attention threshold from training JSONL only."""
    popular_fraction = float(popular_fraction)
    if not 0.0 < popular_fraction < 1.0:
        raise ValueError(
            f"popular_fraction must be within (0, 1), got {popular_fraction}"
        )

    path = Path(train_jsonl)
    values_by_creator: Dict[str, float] = {}
    total_hits = 0
    missing_prior_attention_hits = 0
    nonfinite_prior_attention_hits = 0
    missing_creator_id_hits = 0

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            hits = row.get("hits")
            if not isinstance(hits, list):
                continue

            for hit_idx, hit in enumerate(hits):
                if not isinstance(hit, dict):
                    continue
                total_hits += 1

                raw_prior_attention = _first_present(hit, PRIOR_ATTENTION_KEYS)
                if raw_prior_attention is None:
                    missing_prior_attention_hits += 1
                    continue

                prior_attention = _finite_float(raw_prior_attention)
                if prior_attention is None:
                    nonfinite_prior_attention_hits += 1
                    continue

                creator_id = _creator_id_from_hit(hit)
                if creator_id is None:
                    missing_creator_id_hits += 1
                    continue

                previous_prior_attention = values_by_creator.get(creator_id)
                if previous_prior_attention is None:
                    values_by_creator[creator_id] = prior_attention
                elif not math.isclose(previous_prior_attention, prior_attention, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(
                        "Conflicting finite prior-attention values for creator "
                        f"{creator_id!r}: {previous_prior_attention} vs {prior_attention} "
                        f"in training JSONL {path}"
                    )

    prior_attention_values = list(values_by_creator.values())
    if not prior_attention_values:
        raise ValueError(f"No finite prior-attention values found in training JSONL: {path}")

    sorted_prior_attention = sorted(prior_attention_values, reverse=True)
    requested_popular_count = max(1, math.ceil(len(sorted_prior_attention) * popular_fraction))
    threshold = sorted_prior_attention[requested_popular_count - 1]
    actual_popular_count = sum(1 for value in sorted_prior_attention if value >= threshold)

    return {
        "prior_attention_threshold": threshold,
        "popular_fraction": popular_fraction,
        "requested_popular_count": requested_popular_count,
        "actual_popular_count": actual_popular_count,
        "num_valid_prior_attention_creators": len(sorted_prior_attention),
        "num_unique_creators": len(values_by_creator),
        "total_hits": total_hits,
        "missing_prior_attention_hits": missing_prior_attention_hits,
        "nonfinite_prior_attention_hits": nonfinite_prior_attention_hits,
        "missing_creator_id_hits": missing_creator_id_hits,
        "train_jsonl": str(path),
    }
