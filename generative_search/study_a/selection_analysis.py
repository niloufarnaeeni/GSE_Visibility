import argparse
import csv
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional

from .skill_analysis import SkillCoverageContext, load_skill_context, skill_split_metadata


ANALYSIS_FILES = (
    "generator_selection_candidates.csv",
    "generator_selection_analysis.csv",
    "prior_attention_group_selection.csv",
    "prior_attention_group_stage_analysis.csv",
    "creator_coverage.csv",
    "generator_textual_visibility.csv",
    "prior_attention_group_textual_visibility.csv",
    "prior_attention_group_thresholds.json",
)
PRIOR_ATTENTION_GROUPS = ("low", "mid", "high")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[.'_-][A-Za-z0-9]+)*", re.UNICODE)


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def finite_float(value, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} is not finite: {value!r}")
    return numeric


def profile_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(str(text)))


def reason_word_count(text: str) -> int:
    return len(WORD_RE.findall(str(text)))


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_creator_prior_attention_values(source: str | Path) -> Dict[str, float]:
    """Load the repository's canonical historical creator prior attention map."""
    from rag_retrieval.infer.eval.build_run_qrels import load_creator_prior_attention

    prior_attention = load_creator_prior_attention(Path(source))
    clean = {}
    for creator_id, prior_attention in prior_attention.items():
        value = finite_float(prior_attention, f"historical prior attention for {creator_id}")
        if value < 0:
            raise ValueError(f"historical prior attention for {creator_id} is negative: {value}")
        clean[str(creator_id)] = value
    if not clean:
        raise ValueError(f"No valid historical creator prior attention values found in {source}")
    return clean


def build_prior_attention_groups(creator_prior_attentions: Dict[str, float]) -> tuple[Dict[str, str], dict]:
    items = sorted(((cid, finite_float(prior_attention, f"prior attention for {cid}")) for cid, prior_attention in creator_prior_attentions.items()), key=lambda item: (item[1], item[0]))
    n = len(items)
    if n == 0:
        raise ValueError("Cannot build prior attention groups from an empty creator-prior attention map")
    low_count = max(1, math.ceil(0.20 * n))
    high_count = max(1, math.ceil(0.20 * n))
    if low_count + high_count > n:
        high_count = max(0, n - low_count)

    groups = {}
    for idx, (creator_id, _) in enumerate(items):
        if idx < low_count:
            groups[creator_id] = "low"
        elif idx >= n - high_count:
            groups[creator_id] = "high"
        else:
            groups[creator_id] = "mid"

    counts = {group: sum(1 for value in groups.values() if value == group) for group in PRIOR_ATTENTION_GROUPS}
    metadata = {
        "prior_attention_source": "canonical load_creator_prior_attention(raw_dir)",
        "num_unique_creators": n,
        "p20_prior_attention_threshold": items[low_count - 1][1],
        "p80_prior_attention_threshold": items[n - high_count][1] if high_count else None,
        "low_creator_count": counts["low"],
        "mid_creator_count": counts["mid"],
        "high_creator_count": counts["high"],
        "tie_rule": "Creators are sorted by (prior attention ascending, creator_id ascending); exact percentile ties are assigned deterministically by creator_id to hit bottom-20/top-20 counts.",
    }
    return groups, metadata


def _successful_generation_rows(generations: Iterable[dict]) -> Dict[str, dict]:
    rows = {}
    for row in generations:
        if row.get("generation_success") and row.get("query_id"):
            rows[row["query_id"]] = row
    return rows


def build_candidate_rows(
    ranked_rows: List[dict],
    generation_rows: List[dict],
    prior_attention_groups: Dict[str, str],
    creator_prior_attentions: Dict[str, float],
    skill_context: Optional[SkillCoverageContext] = None,
) -> List[dict]:
    generations_by_qid = _successful_generation_rows(generation_rows)
    ranked_qids = {row.get("query_id") for row in ranked_rows if row.get("query_id")}
    missing_ranked = sorted(query_id for query_id in generations_by_qid if query_id not in ranked_qids)
    if missing_ranked:
        raise ValueError(f"Missing ranked_inputs rows for successful generation query_ids={missing_ranked}")
    rows = []
    output_k_by_query = {}
    for ranked in ranked_rows:
        query_id = ranked.get("query_id")
        generation = generations_by_qid.get(query_id)
        if not generation:
            continue
        input_k = int(generation.get("input_k") or ranked.get("input_k") or len(ranked.get("top_k_creator_ids", [])))
        output_k = int(generation.get("output_k") or len(generation.get("valid_recommendations") or []))
        ranked_candidates = ranked.get("ranked_candidates") or []
        top_candidates = ranked_candidates[:input_k]
        valid_recs = generation.get("valid_recommendations") or []
        if len(top_candidates) != input_k:
            raise ValueError(f"query_id {query_id} has {len(top_candidates)} top candidates, expected input_k={input_k}")
        if len(valid_recs) != output_k:
            raise ValueError(f"query_id {query_id} has {len(valid_recs)} valid recommendations, expected output_k={output_k}")
        output_k_by_query[query_id] = output_k
        selected_rank_by_creator = {}
        for rec in valid_recs:
            creator_id = str(rec.get("creator_id", "")).strip()
            rank = rec.get("rank")
            if creator_id and creator_id not in selected_rank_by_creator:
                selected_rank_by_creator[creator_id] = rank
        top_creator_ids = {str(candidate.get("creator_id", "")).strip() for candidate in top_candidates}
        unknown_selected = sorted(set(selected_rank_by_creator) - top_creator_ids)
        if unknown_selected:
            raise ValueError(f"query_id {query_id} selected creators absent from top-k candidates: {unknown_selected}")
        for position, candidate in enumerate(top_candidates, start=1):
            creator_id = str(candidate.get("creator_id", "")).strip()
            if not creator_id:
                continue
            relevance = finite_float(candidate.get("label"), f"relevance label for {query_id}/{creator_id}")
            prior_attention = finite_float(candidate.get("prior_attention"), f"prior attention for {query_id}/{creator_id}")
            if prior_attention < 0:
                raise ValueError(f"prior attention for {query_id}/{creator_id} is negative: {prior_attention}")
            historical_prior_attention = creator_prior_attentions.get(creator_id)
            if historical_prior_attention is None:
                raise ValueError(f"Missing canonical historical prior attention for creator {creator_id}")
            if not math.isclose(prior_attention, historical_prior_attention, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(
                    f"prior attention mismatch for query_id={query_id} creator_id={creator_id}: "
                    f"ranked candidate prior_attention={prior_attention}, canonical historical prior attention={historical_prior_attention}"
                )
            score = finite_float(candidate.get("reranker_score"), f"reranker score for {query_id}/{creator_id}")
            token_count = profile_token_count(candidate.get("content", ""))
            generated_rank = selected_rank_by_creator.get(creator_id)
            trial_id, fold_id, split_name = skill_split_metadata({**ranked, **generation})
            individual_skill = (
                skill_context.individual_skill_coverage(
                    query_id,
                    ranked.get("query", generation.get("query", "")),
                    creator_id,
                    trial_id,
                    fold_id,
                    split_name,
                )
                if skill_context is not None
                else None
            )
            rows.append({
                "query_id": query_id,
                "loss_type": generation.get("loss_type") or ranked.get("loss_type"),
                "method": generation.get("loss_type") or ranked.get("loss_type"),
                "creator_id": creator_id,
                "input_position": position,
                "relevance_label": relevance,
                "prior_attention": prior_attention,
                "log1p_prior_attention": math.log1p(prior_attention),
                "reranker_score": score,
                "profile_token_count": token_count,
                "log1p_profile_tokens": math.log1p(token_count),
                "individual_skill_coverage": individual_skill,
                "selected_by_generator": 1 if generated_rank is not None else 0,
                "generated_rank": generated_rank,
                "prior_attention_group": prior_attention_groups.get(creator_id),
            })
    validate_candidate_rows(rows, output_k_by_query)
    return rows


def validate_candidate_rows(rows: List[dict], output_k_by_query: Optional[Dict[str, int]] = None) -> None:
    by_query = defaultdict(list)
    for row in rows:
        if row["prior_attention_group"] not in PRIOR_ATTENTION_GROUPS:
            raise ValueError(f"Missing global prior attention group for creator {row['creator_id']}")
        by_query[row["query_id"]].append(row)
    for query_id, query_rows in by_query.items():
        input_k = len(query_rows)
        positions = sorted(int(row["input_position"]) for row in query_rows)
        if positions != list(range(1, input_k + 1)):
            raise ValueError(f"input_position is not 1..input_k for {query_id}: {positions}")
        selected = [row for row in query_rows if int(row["selected_by_generator"]) == 1]
        generated_ranks = sorted(int(row["generated_rank"]) for row in selected)
        if generated_ranks != list(range(1, len(selected) + 1)):
            raise ValueError(f"generated_rank is not consecutive for selected creators in {query_id}: {generated_ranks}")
        if output_k_by_query and query_id in output_k_by_query and len(selected) != int(output_k_by_query[query_id]):
            raise ValueError(f"query_id {query_id} has {len(selected)} selected_by_generator rows, expected output_k={output_k_by_query[query_id]}")


def _group_by_loss(rows: List[dict]) -> Dict[str, List[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("loss_type") or "unknown")].append(row)
    return grouped


def group_selection_rows(candidate_rows: List[dict]) -> List[dict]:
    output = []
    for loss_type, rows in _group_by_loss(candidate_rows).items():
        num_queries = len({row["query_id"] for row in rows})
        for group in PRIOR_ATTENTION_GROUPS:
            group_rows = [row for row in rows if row["prior_attention_group"] == group]
            selected = sum(int(row["selected_by_generator"]) for row in group_rows)
            relevant_rows = [row for row in group_rows if float(row["relevance_label"]) > 0.0]
            relevant_selected = sum(int(row["selected_by_generator"]) for row in relevant_rows)
            if relevant_selected > len(relevant_rows):
                raise ValueError(f"relevant_selected_count exceeds relevant_available_count for {loss_type}/{group}")
            output.append({
                "loss_type": loss_type,
                "method": loss_type,
                "prior_attention_group": group,
                "available_candidate_count": len(group_rows),
                "selected_candidate_count": selected,
                "selection_rate": selected / len(group_rows) if group_rows else None,
                "relevant_available_count": len(relevant_rows),
                "relevant_selected_count": relevant_selected,
                "relevant_selection_rate": relevant_selected / len(relevant_rows) if relevant_rows else None,
                "num_queries": num_queries,
            })
    return output


def textual_visibility_rows(candidate_rows: List[dict], generation_rows: List[dict]) -> List[dict]:
    candidate_by_query_creator = {
        (row["query_id"], row["creator_id"]): row
        for row in candidate_rows
    }
    rows = []
    for generation in generation_rows:
        if not generation.get("generation_success") or not generation.get("query_id"):
            continue
        query_id = generation["query_id"]
        output_k = int(generation.get("output_k") or len(generation.get("valid_recommendations") or []))
        valid_recs = generation.get("valid_recommendations") or []
        if len(valid_recs) != output_k:
            raise ValueError(f"query_id {query_id} has {len(valid_recs)} valid recommendations, expected output_k={output_k}")
        ranks = sorted(int(rec.get("rank")) for rec in valid_recs)
        if ranks != list(range(1, output_k + 1)):
            raise ValueError(f"query_id {query_id} generated ranks are not 1..output_k: {ranks}")

        query_items = []
        for rec in sorted(valid_recs, key=lambda item: int(item.get("rank"))):
            creator_id = str(rec.get("creator_id", "")).strip()
            rank = int(rec.get("rank"))
            candidate = candidate_by_query_creator.get((query_id, creator_id))
            if candidate is None:
                raise ValueError(f"query_id {query_id} selected creator {creator_id} is missing from post-hoc candidate rows")
            reason_count = reason_word_count(rec.get("reason", ""))
            if reason_count <= 0:
                raise ValueError(f"query_id {query_id} creator_id {creator_id} has empty generated reason")
            position_weight = math.exp(-rank / output_k)
            query_items.append((rec, candidate, reason_count, position_weight))

        total_reason_words = sum(item[2] for item in query_items)
        weighted_total = sum(reason_count * position_weight for _, _, reason_count, position_weight in query_items)
        if total_reason_words <= 0 or weighted_total <= 0:
            raise ValueError(f"query_id {query_id} has nonpositive textual visibility denominator")

        query_rows = []
        for rec, candidate, reason_count, position_weight in query_items:
            query_rows.append({
                "query_id": query_id,
                "loss_type": generation.get("loss_type") or candidate.get("loss_type"),
                "method": generation.get("loss_type") or candidate.get("loss_type"),
                "creator_id": candidate["creator_id"],
                "generated_rank": int(rec.get("rank")),
                "prior_attention": candidate["prior_attention"],
                "prior_attention_group": candidate["prior_attention_group"],
                "relevance_label": candidate["relevance_label"],
                "reason_word_count": reason_count,
                "reason_word_share": reason_count / total_reason_words,
                "position_weight": position_weight,
                "position_adjusted_reason_visibility": (reason_count * position_weight) / weighted_total,
            })
        validate_textual_visibility_query_rows(query_id, output_k, query_rows)
        rows.extend(query_rows)
    return rows


def validate_textual_visibility_query_rows(query_id: str, output_k: int, rows: List[dict]) -> None:
    if len(rows) != output_k:
        raise ValueError(f"query_id {query_id} has {len(rows)} textual visibility rows, expected output_k={output_k}")
    ranks = sorted(int(row["generated_rank"]) for row in rows)
    if ranks != list(range(1, output_k + 1)):
        raise ValueError(f"query_id {query_id} textual visibility ranks are not 1..output_k: {ranks}")
    if any(int(row["reason_word_count"]) <= 0 for row in rows):
        raise ValueError(f"query_id {query_id} has nonpositive reason_word_count")
    reason_share_sum = sum(float(row["reason_word_share"]) for row in rows)
    if not math.isclose(reason_share_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"reason_word_share does not sum to 1 for {query_id}: {reason_share_sum}")
    visibility_sum = sum(float(row["position_adjusted_reason_visibility"]) for row in rows)
    if not math.isclose(visibility_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"position_adjusted_reason_visibility does not sum to 1 for {query_id}: {visibility_sum}")
    if any(row["prior_attention_group"] not in PRIOR_ATTENTION_GROUPS for row in rows):
        raise ValueError(f"query_id {query_id} has a selected creator without a global prior attention group")


def group_textual_visibility_rows(visibility_rows: List[dict]) -> List[dict]:
    output = []
    for loss_type, rows in _group_by_loss(visibility_rows).items():
        num_queries = len({row["query_id"] for row in rows})
        for group in PRIOR_ATTENTION_GROUPS:
            group_rows = [row for row in rows if row["prior_attention_group"] == group]
            count = len(group_rows)
            total_share = sum(float(row["reason_word_share"]) for row in group_rows)
            total_adjusted = sum(float(row["position_adjusted_reason_visibility"]) for row in group_rows)
            output.append({
                "loss_type": loss_type,
                "method": loss_type,
                "prior_attention_group": group,
                "num_queries": num_queries,
                "selected_creator_count": count,
                "mean_reason_word_count": mean([float(row["reason_word_count"]) for row in group_rows]) if group_rows else None,
                "total_reason_word_share": total_share,
                "mean_reason_word_share": total_share / count if count else None,
                "mean_query_reason_word_share": total_share / num_queries if num_queries else None,
                "total_position_adjusted_visibility": total_adjusted,
                "mean_position_adjusted_visibility": total_adjusted / count if count else None,
                "mean_query_position_adjusted_visibility": total_adjusted / num_queries if num_queries else None,
            })
    return output


def _full_candidate_creators_by_query(ranked_rows: List[dict], generation_rows: List[dict]) -> Dict[str, dict]:
    ranked_by_qid = {row.get("query_id"): row for row in ranked_rows if row.get("query_id")}
    output = {}
    for generation in generation_rows:
        if not generation.get("generation_success") or not generation.get("query_id"):
            continue
        query_id = generation["query_id"]
        creators = generation.get("final_candidate_creator_ids")
        if not creators:
            creators = (generation.get("candidate_details") or {}).get("candidate_creator_ids")
        if not creators:
            ranked = ranked_by_qid.get(query_id) or {}
            creators = (ranked.get("candidate_details") or {}).get("candidate_creator_ids")
        if not creators:
            ranked = ranked_by_qid.get(query_id) or {}
            creators = [candidate.get("creator_id") for candidate in ranked.get("ranked_candidates") or []]
        clean = [str(creator_id).strip() for creator_id in creators if str(creator_id).strip()]
        if not clean:
            raise ValueError(f"Missing full eligible candidate creator IDs for successful query_id={query_id}")
        output[query_id] = {
            "loss_type": generation.get("loss_type") or (ranked_by_qid.get(query_id) or {}).get("loss_type") or "unknown",
            "creator_ids": clean,
        }
    return output


def creator_coverage_rows(candidate_rows: List[dict], ranked_rows: List[dict], generation_rows: List[dict]) -> List[dict]:
    output = []
    stage_specs = (
        ("reranker_top10", lambda row: int(row["input_position"]) <= 10),
        ("reranker_top5", lambda row: int(row["input_position"]) <= 5),
        ("generator_top5", lambda row: int(row["selected_by_generator"]) == 1),
    )
    full_by_query = _full_candidate_creators_by_query(ranked_rows, generation_rows)
    for loss_type, rows in _group_by_loss(candidate_rows).items():
        successful_query_ids = {row["query_id"] for row in rows}
        missing = sorted(query_id for query_id in successful_query_ids if query_id not in full_by_query)
        if missing:
            raise ValueError(f"Missing full eligible candidate creator IDs for successful query_ids={missing}")
        eligible_creators = {
            creator_id
            for query_id in successful_query_ids
            for creator_id in full_by_query[query_id]["creator_ids"]
        }
        eligible_count = len(eligible_creators)
        num_queries = len(successful_query_ids)
        for stage, predicate in stage_specs:
            stage_creators = {row["creator_id"] for row in rows if predicate(row)}
            coverage = len(stage_creators) / eligible_count if eligible_count else None
            if coverage is not None and not 0.0 <= coverage <= 1.0:
                raise ValueError(f"creator_coverage outside [0, 1] for {loss_type}/{stage}: {coverage}")
            output.append({
                "loss_type": loss_type,
                "method": loss_type,
                "stage": stage,
                "unique_creator_count": len(stage_creators),
                "eligible_creator_count": eligible_count,
                "creator_coverage": coverage,
                "num_queries": num_queries,
            })
    return output


def stage_analysis_rows(candidate_rows: List[dict]) -> List[dict]:
    output = []
    stage_specs = (
        ("reranker_top10", lambda row: int(row["input_position"]) <= 10, None),
        ("reranker_top5", lambda row: int(row["input_position"]) <= 5, "input_position"),
        ("generator_top5", lambda row: int(row["selected_by_generator"]) == 1, "generated_rank"),
    )
    for loss_type, rows in _group_by_loss(candidate_rows).items():
        num_queries = len({row["query_id"] for row in rows})
        for stage, predicate, rank_field in stage_specs:
            stage_rows = [row for row in rows if predicate(row)]
            total_count = len(stage_rows)
            exposure_by_group = defaultdict(float)
            if rank_field:
                for row in stage_rows:
                    rank = int(row[rank_field])
                    exposure_by_group[row["prior_attention_group"]] += 1.0 / math.log2(1 + rank)
            total_exposure = sum(exposure_by_group.values())
            for group in PRIOR_ATTENTION_GROUPS:
                count = sum(1 for row in stage_rows if row["prior_attention_group"] == group)
                exposure = exposure_by_group[group]
                output.append({
                    "loss_type": loss_type,
                    "method": loss_type,
                    "stage": stage,
                    "prior_attention_group": group,
                    "creator_count": count,
                    "creator_share": count / total_count if total_count else None,
                    "total_group_exposure": exposure if rank_field else None,
                    "group_exposure_share": exposure / total_exposure if total_exposure else None,
                    "num_queries": num_queries,
                })
    return output


def _standardize(values: List[float]) -> tuple[List[float], bool]:
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    if scale <= 0 or not math.isfinite(scale):
        return [0.0 for _ in values], True
    return [(value - mean_value) / scale for value in values], False


def _normal_pvalue(coef: float, se: float) -> float:
    if se <= 0 or not math.isfinite(se):
        return None
    z = abs(coef / se)
    return 2.0 * (1.0 - NormalDist().cdf(z))


def _safe_exp(value: float) -> Optional[float]:
    try:
        result = math.exp(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _fit_logit_clustered(rows: List[dict], predictors: List[str], analysis_type: str) -> tuple[List[dict], str, str]:
    try:
        import pandas as pd
        import statsmodels.api as sm
    except Exception as exc:
        return [], "logistic_regression_query_clustered_se", f"fit_failed: missing statsmodels/pandas: {exc}"

    if len({row["query_id"] for row in rows}) < 2:
        return [], "logistic_regression_query_clustered_se", "fit_failed: too few successful queries for clustered standard errors"
    if len({int(row["selected_by_generator"]) for row in rows}) < 2:
        return [], "logistic_regression_query_clustered_se", "fit_failed: constant selected_by_generator"

    df = pd.DataFrame(rows)
    y = df["selected_by_generator"].astype(float)
    x = df[predictors].astype(float)
    nonconstant = [col for col in predictors if x[col].nunique(dropna=True) > 1]
    dropped = [col for col in predictors if col not in nonconstant]
    if not nonconstant:
        return [], "logistic_regression_query_clustered_se", "fit_failed: all predictors are constant"
    x = sm.add_constant(x[nonconstant], has_constant="add")
    try:
        fitted = sm.Logit(y, x).fit(
            disp=False,
            maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": df["query_id"]},
        )
        converged = getattr(fitted, "mle_retvals", {}).get("converged")
        if converged is None:
            converged = getattr(fitted, "converged", None)
        if converged is False:
            return [], "logistic_regression_query_clustered_se", "fit_failed: optimizer did not converge"
        fit_status = "ok" if not dropped else f"ok: dropped constant predictors {','.join(dropped)}"
    except Exception as exc:
        return [], "logistic_regression_query_clustered_se", f"fit_failed: {type(exc).__name__}: {exc}"

    output = []
    for predictor in nonconstant:
        coef = float(fitted.params[predictor])
        se = float(fitted.bse[predictor])
        if not math.isfinite(coef) or not math.isfinite(se):
            return [], "logistic_regression_query_clustered_se", f"fit_failed: nonfinite coefficient or standard error for {predictor}"
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        output.append({
            "analysis_type": analysis_type,
            "predictor": predictor,
            "coefficient": coef,
            "odds_ratio": _safe_exp(coef),
            "ci_95_lower": _safe_exp(ci_low),
            "ci_95_upper": _safe_exp(ci_high),
            "p_value": _normal_pvalue(coef, se),
            "model_type": "logistic_regression_query_clustered_se",
            "fit_status": fit_status,
        })
    for predictor in dropped:
        output.append({
            "analysis_type": analysis_type,
            "predictor": predictor,
            "coefficient": None,
            "odds_ratio": None,
            "ci_95_lower": None,
            "ci_95_upper": None,
            "p_value": None,
            "model_type": "logistic_regression_query_clustered_se",
            "fit_status": "not_fit: constant predictor",
        })
    return output, "logistic_regression_query_clustered_se", fit_status


def model_rows(candidate_rows: List[dict]) -> List[dict]:
    output = []
    for loss_type, rows in _group_by_loss(candidate_rows).items():
        num_queries = len({row["query_id"] for row in rows})
        relevance_z, relevance_constant = _standardize([float(row["relevance_label"]) for row in rows])
        log_prior_attention_z, log_prior_attention_constant = _standardize([float(row["log1p_prior_attention"]) for row in rows])
        log_profile_z, log_profile_constant = _standardize([float(row["log1p_profile_tokens"]) for row in rows])
        modeled = []
        for idx, row in enumerate(rows):
            input_k = max(int(item["input_position"]) for item in rows if item["query_id"] == row["query_id"])
            modeled.append({
                **row,
                "standardized_relevance": relevance_z[idx],
                "standardized_log1p_prior_attention": log_prior_attention_z[idx],
                "standardized_log1p_profile_tokens": log_profile_z[idx],
                "earlier_position": input_k + 1 - int(row["input_position"]),
                "mid_vs_low": 1 if row["prior_attention_group"] == "mid" else 0,
                "high_vs_low": 1 if row["prior_attention_group"] == "high" else 0,
            })

        specs = (
            ("continuous_prior_attention", ["standardized_relevance", "earlier_position", "standardized_log1p_prior_attention"]),
            ("prior_attention_group", ["standardized_relevance", "earlier_position", "mid_vs_low", "high_vs_low"]),
            ("continuous_prior_attention_profile_length", ["standardized_relevance", "earlier_position", "standardized_log1p_prior_attention", "standardized_log1p_profile_tokens"]),
        )
        for analysis_type, predictors in specs:
            fitted_rows, model_type, fit_status = _fit_logit_clustered(modeled, predictors, analysis_type)
            if not fitted_rows:
                fitted_rows = [{"analysis_type": analysis_type, "predictor": predictor, "coefficient": None, "odds_ratio": None, "ci_95_lower": None, "ci_95_upper": None, "p_value": None, "model_type": model_type, "fit_status": fit_status} for predictor in predictors]
            for result in fitted_rows:
                result.update({
                    "loss_type": loss_type,
                    "method": loss_type,
                    "num_candidate_rows": len(rows),
                    "num_queries": num_queries,
                })
                output.append(result)
        if relevance_constant or log_prior_attention_constant or log_profile_constant:
            logging.warning("Constant standardized predictor detected for loss_type=%s", loss_type)
        skill_rows = [row for row in rows if row.get("individual_skill_coverage") is not None]
        skill_values = [float(row["individual_skill_coverage"]) for row in skill_rows]
        if skill_rows:
            skill_relevance_z, _ = _standardize([float(row["relevance_label"]) for row in skill_rows])
            skill_log_prior_attention_z, _ = _standardize([float(row["log1p_prior_attention"]) for row in skill_rows])
            skill_z, skill_constant = _standardize(skill_values)
            skill_modeled = []
            for idx, row in enumerate(skill_rows):
                input_k = max(int(item["input_position"]) for item in skill_rows if item["query_id"] == row["query_id"])
                skill_modeled.append({
                    **row,
                    "standardized_relevance": skill_relevance_z[idx],
                    "standardized_log1p_prior_attention": skill_log_prior_attention_z[idx],
                    "earlier_position": input_k + 1 - int(row["input_position"]),
                    "standardized_individual_skill_coverage": skill_z[idx],
                })
            predictors = ["standardized_relevance", "earlier_position", "standardized_log1p_prior_attention", "standardized_individual_skill_coverage"]
            fitted_rows, model_type, fit_status = _fit_logit_clustered(skill_modeled, predictors, "continuous_prior_attention_skill_control")
            if not fitted_rows:
                fitted_rows = [{"analysis_type": "continuous_prior_attention_skill_control", "predictor": predictor, "coefficient": None, "odds_ratio": None, "ci_95_lower": None, "ci_95_upper": None, "p_value": None, "model_type": model_type, "fit_status": fit_status} for predictor in predictors]
            for result in fitted_rows:
                result.update({
                    "loss_type": loss_type,
                    "method": loss_type,
                    "num_candidate_rows": len(skill_rows),
                    "num_queries": len({row["query_id"] for row in skill_rows}),
                })
                output.append(result)
            if skill_constant:
                logging.warning("Constant individual_skill_coverage predictor detected for loss_type=%s", loss_type)
        else:
            for predictor in ("standardized_relevance", "earlier_position", "standardized_log1p_prior_attention", "standardized_individual_skill_coverage"):
                output.append({
                    "analysis_type": "continuous_prior_attention_skill_control",
                    "predictor": predictor,
                    "coefficient": None,
                    "odds_ratio": None,
                    "ci_95_lower": None,
                    "ci_95_upper": None,
                    "p_value": None,
                    "model_type": "logistic_regression_query_clustered_se",
                    "fit_status": "not_fit: no rows with valid individual_skill_coverage",
                    "loss_type": loss_type,
                    "method": loss_type,
                    "num_candidate_rows": 0,
                    "num_queries": 0,
                })
    return output


def run_level_posthoc_summary(
    regression_rows: List[dict],
    selection_rows: List[dict],
    stage_rows: List[dict],
    coverage_rows_: List[dict],
    group_textual_visibility_rows_: List[dict],
) -> dict:
    summary = {}
    predictor_map = {
        "standardized_relevance": "relevance",
        "earlier_position": "position",
        "standardized_log1p_prior_attention": "prior_attention",
    }
    continuous_rows = {
        row.get("predictor"): row
        for row in regression_rows
        if row.get("analysis_type") == "continuous_prior_attention"
    }
    for predictor, label in predictor_map.items():
        row = continuous_rows.get(predictor, {})
        summary[f"selection_OR_{label}"] = row.get("odds_ratio")
        summary[f"selection_OR_{label}_ci_low"] = row.get("ci_95_lower")
        summary[f"selection_OR_{label}_ci_high"] = row.get("ci_95_upper")
        summary[f"selection_p_{label}"] = row.get("p_value")

    selection_by_group = {row.get("prior_attention_group"): row for row in selection_rows}
    for group in PRIOR_ATTENTION_GROUPS:
        row = selection_by_group.get(group, {})
        summary[f"{group}_prior_attention_selection_rate"] = row.get("selection_rate")
        summary[f"{group}_prior_attention_relevant_selection_rate"] = row.get("relevant_selection_rate")

    generator_stage_by_group = {
        row.get("prior_attention_group"): row
        for row in stage_rows
        if row.get("stage") == "generator_top5"
    }
    for group in PRIOR_ATTENTION_GROUPS:
        row = generator_stage_by_group.get(group, {})
        summary[f"generator_{group}_prior_attention_share"] = row.get("creator_share")
        summary[f"generator_{group}_prior_attention_exposure_share"] = row.get("group_exposure_share")

    coverage_by_stage = {row.get("stage"): row for row in coverage_rows_}
    for stage in ("reranker_top10", "reranker_top5", "generator_top5"):
        row = coverage_by_stage.get(stage, {})
        summary[f"creator_coverage_{stage}"] = row.get("creator_coverage")
    generator_coverage = summary.get("creator_coverage_generator_top5")
    reranker5_coverage = summary.get("creator_coverage_reranker_top5")
    summary["delta_creator_coverage_generator_vs_reranker5"] = (
        float(generator_coverage) - float(reranker5_coverage)
        if generator_coverage is not None and reranker5_coverage is not None
        else None
    )

    textual_by_group = {row.get("prior_attention_group"): row for row in group_textual_visibility_rows_}
    for group in PRIOR_ATTENTION_GROUPS:
        row = textual_by_group.get(group, {})
        summary[f"{group}_prior_attention_textual_visibility"] = row.get("mean_query_position_adjusted_visibility")
    return summary


def validate_stage_rows(rows: List[dict]) -> None:
    for loss_type in {row["loss_type"] for row in rows}:
        for stage in {row["stage"] for row in rows if row["loss_type"] == loss_type}:
            stage_rows = [row for row in rows if row["loss_type"] == loss_type and row["stage"] == stage]
            share_sum = sum(float(row["creator_share"]) for row in stage_rows if row["creator_share"] is not None)
            if stage_rows and not math.isclose(share_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"creator_share does not sum to 1 for {loss_type}/{stage}: {share_sum}")
            exposure_values = [row["group_exposure_share"] for row in stage_rows if row["group_exposure_share"] is not None]
            if exposure_values:
                exposure_sum = sum(float(value) for value in exposure_values)
                if not math.isclose(exposure_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"group_exposure_share does not sum to 1 for {loss_type}/{stage}: {exposure_sum}")


def run_analysis(output_dir: str | Path, raw_data_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    creator_prior_attentions = load_creator_prior_attention_values(raw_data_dir)
    prior_attention_groups, threshold_metadata = build_prior_attention_groups(creator_prior_attentions)
    threshold_metadata["raw_data_dir_path"] = str(raw_data_dir)
    ranked_rows = read_jsonl(output_dir / "ranked_inputs.jsonl")
    generation_rows = read_jsonl(output_dir / "generations.jsonl")
    skill_context = load_skill_context(raw_data_dir, output_dir)
    candidate_rows = build_candidate_rows(ranked_rows, generation_rows, prior_attention_groups, creator_prior_attentions, skill_context)
    selection_rows = group_selection_rows(candidate_rows)
    stages = stage_analysis_rows(candidate_rows)
    coverage = creator_coverage_rows(candidate_rows, ranked_rows, generation_rows)
    textual_visibility = textual_visibility_rows(candidate_rows, generation_rows)
    group_textual_visibility = group_textual_visibility_rows(textual_visibility)
    validate_stage_rows(stages)
    regressions = model_rows(candidate_rows)

    write_csv(output_dir / "generator_selection_candidates.csv", candidate_rows)
    write_csv(output_dir / "generator_selection_analysis.csv", regressions)
    write_csv(output_dir / "prior_attention_group_selection.csv", selection_rows)
    write_csv(output_dir / "prior_attention_group_stage_analysis.csv", stages)
    write_csv(output_dir / "creator_coverage.csv", coverage)
    write_csv(output_dir / "generator_textual_visibility.csv", textual_visibility)
    write_csv(output_dir / "prior_attention_group_textual_visibility.csv", group_textual_visibility)
    write_json(output_dir / "prior_attention_group_thresholds.json", threshold_metadata)
    return {
        "candidate_rows": len(candidate_rows),
        "successful_queries": len({row["query_id"] for row in candidate_rows}),
        "thresholds": threshold_metadata,
        "analysis_files": list(ANALYSIS_FILES),
        "posthoc_summary": run_level_posthoc_summary(regressions, selection_rows, stages, coverage, group_textual_visibility),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc Study A generator-selection and prior attention-group analyses")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--raw_data_dir", required=True, help="Raw data directory containing canonical prior_attention_scores.csv and creator_details.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(args.output_dir, args.raw_data_dir)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.info("Study A selection analysis complete: %s", summary)


if __name__ == "__main__":
    main()
