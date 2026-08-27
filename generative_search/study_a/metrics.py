import math
from typing import Dict, List, Tuple

from rag_retrieval.infer.eval.eval_metrics import compute_exposure_metrics, evaluate_binary_and_graded


def ideal_dcg(qrels: Dict[str, float], k: int) -> float:
    gains = sorted((float(v) for v in qrels.values()), reverse=True)[:k]
    return sum(gain / math.log2(idx + 2) for idx, gain in enumerate(gains))


def validate_idcg_unchanged(query_id: str, full_qrels: Dict[str, float], sampled_qrels: Dict[str, float], ks: Tuple[int, ...] = (5, 10)) -> None:
    for k in tuple(dict.fromkeys(int(value) for value in ks)):
        full = ideal_dcg(full_qrels, k)
        sampled = ideal_dcg(sampled_qrels, k)
        if not math.isclose(full, sampled, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"IDCG mismatch for query_id={query_id} at @{k}: full={full}, sampled={sampled}")


def run_from_order(query_id: str, ordered_creator_ids: List[str]) -> Dict[str, Dict[str, float]]:
    n = len(ordered_creator_ids)
    return {query_id: {creator_id: float(n - idx) for idx, creator_id in enumerate(ordered_creator_ids)}}


def graded_ndcg(query_id: str, qrels: Dict[str, float], ordered_creator_ids: List[str], ks: Tuple[int, ...]) -> Dict[int, float]:
    graded_qrels = {cid: int(value) for cid, value in qrels.items()}
    _, per_query = evaluate_binary_and_graded(
        qrels_binary={query_id: {cid: 1 for cid in graded_qrels}},
        qrels_graded={query_id: graded_qrels}, run=run_from_order(query_id, ordered_creator_ids), ks=ks,
    )
    result = {}
    for k in ks:
        key = f"graded_ndcg_cut_{k}"
        if query_id not in per_query or key not in per_query[query_id]:
            raise KeyError(f"Canonical graded nDCG did not return required key {key!r} for {query_id}")
        result[k] = float(per_query[query_id][key])
    return result


def exposure_metrics(query_id: str, ordered_creator_ids: List[str], creator_prior_attentions: Dict[str, float], ks: Tuple[int, ...]) -> Dict[str, float]:
    missing = [cid for cid in ordered_creator_ids if cid not in creator_prior_attentions]
    if missing:
        raise KeyError(f"Missing prior attention values for required creator IDs: {missing}")
    avg, disc = compute_exposure_metrics(run_from_order(query_id, ordered_creator_ids), creator_prior_attentions, ks)
    result = {**avg, **disc}
    required = [name for k in ks for name in (f"Exp@{k}", f"DExp@{k}")]
    missing_keys = [key for key in required if key not in result]
    if missing_keys:
        raise KeyError(f"Canonical compute_exposure_metrics missing required keys: {missing_keys}; returned {sorted(result)}")
    return result
