import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from rag_retrieval.generative_search.study_a.selection_analysis import PRIOR_ATTENTION_GROUPS


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def nullable_mean(values: Iterable[float]):
    values = list(values)
    return sum(values) / len(values) if values else None


def group_rank_summary(ranked_rows: List[dict], prior_attention_groups: Dict[str, str], input_k: int = 10, top_k: int = 5) -> dict:
    positions = defaultdict(list)
    top_counts = defaultdict(int)
    available = defaultdict(int)
    for row in ranked_rows:
        for pos, candidate in enumerate((row.get("ranked_candidates") or [])[:input_k], start=1):
            group = prior_attention_groups.get(str(candidate.get("creator_id")))
            if group not in PRIOR_ATTENTION_GROUPS:
                raise ValueError(f"Missing global prior attention group for creator {candidate.get('creator_id')}")
            available[group] += 1
            positions[group].append(float(pos))
            if pos <= top_k:
                top_counts[group] += 1
    out = {}
    for group in PRIOR_ATTENTION_GROUPS:
        out[f"mean_rank_{group}"] = nullable_mean(positions[group])
        out[f"top5_rate_{group}"] = top_counts[group] / available[group] if available[group] else None
    return out


def generated_selection_summary(generation_rows: List[dict], prior_attention_groups: Dict[str, str], input_k: int = 10) -> dict:
    available = defaultdict(int)
    selected = defaultdict(int)
    for row in generation_rows:
        if not row.get("generation_success"):
            continue
        ranked = (row.get("ranked_candidates") or [])[:input_k]
        selected_ids = {str(rec.get("creator_id")) for rec in row.get("valid_recommendations") or []}
        for candidate in ranked:
            creator_id = str(candidate.get("creator_id"))
            group = prior_attention_groups.get(creator_id)
            if group not in PRIOR_ATTENTION_GROUPS:
                raise ValueError(f"Missing global prior attention group for creator {creator_id}")
            available[group] += 1
            if creator_id in selected_ids:
                selected[group] += 1
    return {
        f"generated_selection_rate_{group}": selected[group] / available[group] if available[group] else None
        for group in PRIOR_ATTENTION_GROUPS
    }


def position_selection_rows(generation_rows: List[dict], input_k: int = 10) -> List[dict]:
    successful = [row for row in generation_rows if row.get("generation_success")]
    rows = []
    for pos in range(1, input_k + 1):
        selected_positions = []
        for row in successful:
            ranked = (row.get("ranked_candidates") or [])[:input_k]
            if pos > len(ranked):
                continue
            creator_id = str(ranked[pos - 1].get("creator_id"))
            generated = [str(rec.get("creator_id")) for rec in row.get("valid_recommendations") or []]
            if creator_id in generated:
                selected_positions.extend(idx + 1 for idx, cid in enumerate(generated) if cid == creator_id)
        rows.append({
            "input_position": pos,
            "eligible_successful_query_count": len(successful),
            "selected_count": len(selected_positions),
            "selection_rate": len(selected_positions) / len(successful) if successful else 0.0,
            "mean_generated_position_when_selected": mean(selected_positions) if selected_positions else None,
        })
    return rows


def aggregate_metric_rows(metric_rows: List[dict], excluded: Optional[set] = None) -> dict:
    excluded = excluded or {"query_id"}
    keys = sorted({
        key for row in metric_rows for key, value in row.items()
        if key not in excluded
        and value is not None
        and isinstance(value, (int, float))
        and not (isinstance(value, float) and math.isnan(value))
    })
    return {
        key: mean(float(row[key]) for row in metric_rows if row.get(key) is not None and isinstance(row.get(key), (int, float)))
        for key in keys
    }
