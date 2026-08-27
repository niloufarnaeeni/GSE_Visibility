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


def _finite_nonnegative_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0.0:
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


def resolve_train_log_prior_attention_minmax(train_jsonl: str | Path) -> Dict[str, Any]:
    path = Path(train_jsonl)
    prior_attention_by_creator: Dict[str, float] = {}
    total_hits = 0
    skipped_hits = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            hits = row.get("hits")
            if not isinstance(hits, list):
                continue

            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                total_hits += 1

                creator_id = _creator_id_from_hit(hit)
                raw_prior_attention = _first_present(hit, PRIOR_ATTENTION_KEYS)
                prior_attention = _finite_nonnegative_float(raw_prior_attention)
                if creator_id is None or prior_attention is None:
                    skipped_hits += 1
                    continue

                previous_prior_attention = prior_attention_by_creator.get(creator_id)
                if previous_prior_attention is None:
                    prior_attention_by_creator[creator_id] = prior_attention
                elif not math.isclose(previous_prior_attention, prior_attention, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(
                        "Conflicting finite nonnegative prior-attention values for creator "
                        f"{creator_id!r}: {previous_prior_attention} vs {prior_attention} "
                        f"in training JSONL {path}"
                    )

    if not prior_attention_by_creator:
        raise ValueError(f"No finite nonnegative creator prior-attention values found in {path}")

    log_prior_attention = [math.log1p(value) for value in prior_attention_by_creator.values()]
    return {
        "normalization": "log1p_minmax",
        "train_log_prior_attention_min": min(log_prior_attention),
        "train_log_prior_attention_max": max(log_prior_attention),
        "num_unique_creators": len(prior_attention_by_creator),
        "num_valid_prior_attention_creators": len(log_prior_attention),
        "total_hits": total_hits,
        "skipped_hits": skipped_hits,
        "train_jsonl": str(path),
    }
