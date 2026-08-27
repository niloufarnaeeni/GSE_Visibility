import argparse
import csv
import json
import logging
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from rag_retrieval.generative_search.study_a.checkpoint import detect_checkpoint
from rag_retrieval.generative_search.study_a.data import (
    Hit,
    MissingRequiredData,
    QueryRecord,
    read_jsonl as read_input_jsonl,
    stable_query_id,
    test_jsonl_identity,
)
from rag_retrieval.generative_search.study_a.generator import (
    DEFAULT_PROMPT_PATH,
    build_prompt,
    call_ollama,
    parse_json_output,
    validate_generation,
)
from rag_retrieval.generative_search.study_a.metrics import graded_ndcg, exposure_metrics
from rag_retrieval.generative_search.study_a.run import (
    PROJECT_FIELDS,
    apply_document_id_strategy,
    build_ranker,
    checkpoint_fingerprint,
    clear_output_files,
    count_records_by_split,
    file_content_fingerprint,
    json_dumps_strict,
    make_echo_generation,
    metadata_for,
    normalize_allowed_splits,
    rank_candidates,
    read_json,
    read_jsonl,
    rebuild_failure_file,
    resolve_eval_max_length,
    setup_logging,
    split_is_allowed,
    summary_provenance,
    validate_resume_config,
    validate_unique_document_ids,
    write_csv,
    write_json,
)
from rag_retrieval.generative_search.study_a.selection_analysis import (
    build_candidate_rows,
    build_prior_attention_groups,
    load_creator_prior_attention_values,
    model_rows,
)
from rag_retrieval.generative_search.study_a.skill_analysis import load_skill_context, per_query_skill_metrics, skill_metric_summary
from rag_retrieval.infer.eval.build_run_qrels import convert_hits_to_graded_qrels

from .analysis import aggregate_metric_rows, generated_selection_summary, group_rank_summary, position_selection_rows
from .fixed_candidates import CANDIDATE_SOURCE, FIXED_CANDIDATE_COUNT, FIXED_CANDIDATE_RULE, assert_same_creator_set, fixed_from_prepared


DEFAULT_TEST_JSONL = Path(__file__).resolve().parents[3] / "data" / "kaito" / "large_data_creator_profile" / "study_b" / "test&valid.jsonl"
RUN_FILES = (
    "ranked_inputs.jsonl", "generations.jsonl", "generation_failures.jsonl",
    "per_query_metrics.csv", "summary_metrics.csv", "position_selection.csv",
    "generator_selection_candidates.csv", "generator_selection_analysis.csv",
)
RERANK_CACHE_FILES = ("fixed_candidates.jsonl", "ranked_inputs.jsonl", "rerank_config.json", "eligibility_summary.json")
MASTER_IDENTITY_COLUMNS = (
    "loss_type",
    "reranker_checkpoint_fingerprint",
    "candidate_source",
    "fixed_candidate_rule",
    "generator_backend",
    "generator_model",
    "generator_temperature",
    "generator_seed",
    "generator_prompt_fingerprint",
    "eval_max_length",
    "input_k",
    "output_k",
    "allowed_splits",
    "test_data_identity",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Study B fixed-candidate generative-search experiment")
    parser.add_argument("--test_jsonl", default=str(DEFAULT_TEST_JSONL))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--stage", default="all", choices=["all", "rerank", "generate"])
    parser.add_argument("--rerank_cache_dir", default=None)
    parser.add_argument("--allowed_splits", default="test,valid")
    parser.add_argument("--generator_backend", default="ollama", choices=["ollama", "echo"])
    parser.add_argument("--generator_model", default="mistral-nemo:12b")
    parser.add_argument("--ollama_base_url", default="http://localhost:11434")
    parser.add_argument("--generator_prompt_path", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--raw_data_dir", required=True)
    parser.add_argument("--input_k", type=int, default=10)
    parser.add_argument("--output_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--eval_max_length", type=int, default=None)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--model_type", default=None)
    parser.add_argument("--loss_type", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock_reranker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_cli_values(args) -> None:
    if int(args.input_k) != FIXED_CANDIDATE_COUNT:
        raise ValueError("Study B requires --input_k 10")
    if int(args.output_k) != 5:
        raise ValueError("Study B requires --output_k 5")
    if args.temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    normalize_allowed_splits(args.allowed_splits)


def clear_stage_output_files(stage: str, output_dir: Path, cache_dir: Path) -> None:
    if stage == "rerank":
        clear_output_files(cache_dir, RERANK_CACHE_FILES)
    elif stage == "generate":
        clear_output_files(output_dir, RUN_FILES)
    else:
        clear_output_files(cache_dir, RERANK_CACHE_FILES)
        clear_output_files(output_dir, RUN_FILES)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json_dumps_strict(record, ensure_ascii=False) + "\n")


def load_prepared_records(path: str, return_skips: bool = False):
    records = []
    seen_query_ids = {}
    for idx, rec in enumerate(read_input_jsonl(Path(path)), start=1):
        query_id = str(rec.get("query_id") or stable_query_id(rec, idx))
        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate stable query_id {query_id!r} for input rows {seen_query_ids[query_id]} and {idx}")
        seen_query_ids[query_id] = idx
        hits = []
        creator_ids = set()
        for hit in rec.get("hits", []):
            creator_id = str(hit.get("creator_id", "")).strip()
            content = str(hit.get("content", "")).strip()
            if not creator_id or not content:
                raise MissingRequiredData(f"query_id={query_id} empty creator_id/content at input row {idx}")
            if creator_id in creator_ids:
                raise MissingRequiredData(f"query_id={query_id} duplicate creator_id {creator_id} at input row {idx}")
            creator_ids.add(creator_id)
            label = float(hit["label"])
            if not math.isfinite(label) or label < 0:
                raise MissingRequiredData(f"query_id={query_id} label must be finite and nonnegative for creator {creator_id} at input row {idx}")
            if hit.get("prior_attention") is None or "prior_attention" not in hit:
                raise MissingRequiredData(f"query_id={query_id} missing/null prior_attention for creator {creator_id} at input row {idx}")
            prior_attention = float(hit["prior_attention"])
            if not math.isfinite(prior_attention) or prior_attention < 0:
                raise MissingRequiredData(f"query_id={query_id} prior_attention must be finite and nonnegative for creator {creator_id} at input row {idx}")
            document_id = str(hit.get("document_id") or hit.get("doc_id") or "").strip()
            hits.append(Hit(creator_id=creator_id, content=content, label=label, prior_attention=prior_attention, document_id=document_id, document_id_source="input" if document_id else "fallback"))
        records.append(QueryRecord(
            idx=idx,
            query_id=query_id,
            query=str(rec["query"]),
            hits=hits,
            metadata={key: value for key, value in rec.items() if key not in {"query", "hits"}},
        ))
    return (records, []) if return_skips else records


def validate_prepared_candidate_prior_attention(records: List[QueryRecord], creator_prior_attentions: Dict[str, float]) -> None:
    for record in records:
        for hit in record.hits:
            historical_prior_attention = creator_prior_attentions.get(hit.creator_id)
            if historical_prior_attention is None:
                raise ValueError(f"Missing canonical historical prior attention for creator {hit.creator_id} in query_id={record.query_id}")
            if not math.isclose(hit.prior_attention, historical_prior_attention, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(
                    f"prior attention mismatch for query_id={record.query_id} creator_id={hit.creator_id}: "
                    f"prepared prior_attention={hit.prior_attention}, canonical historical prior attention={historical_prior_attention}"
                )


def unique_by_qid(rows: List[dict]) -> List[dict]:
    return list({row.get("query_id"): row for row in rows if row.get("query_id")}.values())


def compact_jsonl(path: Path) -> List[dict]:
    rows = unique_by_qid(read_jsonl(path))
    if rows:
        path.write_text("".join(json_dumps_strict(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return rows


def master_results_path(output_root: Path) -> Path:
    return Path(output_root) / "study_b_all_results.csv"


def _csv_cell(value) -> str:
    return "" if value is None else str(value)


def _identity_key(row: dict) -> tuple:
    return tuple(_csv_cell(row.get(column, "")) for column in MASTER_IDENTITY_COLUMNS)


def upsert_master_results_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows = [existing for existing in rows if _identity_key(existing) != _identity_key(row)]
    rows.append(dict(row))
    fields = sorted({key for item in rows for key in item})
    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=str(path.parent), delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def rerank_cache_signature(args, checkpoint: dict, input_identity: str) -> dict:
    return {
        "study": "B",
        "candidate_source": CANDIDATE_SOURCE,
        "fixed_candidate_rule": FIXED_CANDIDATE_RULE,
        "test_data_identity": input_identity,
        "checkpoint": str(Path(args.model_path)),
        "checkpoint_fingerprint": checkpoint_fingerprint(args.model_path),
        "model_type": checkpoint["model_type"],
        "loss_type": checkpoint["loss_type"],
        "allowed_splits": normalize_allowed_splits(args.allowed_splits),
        "eval_max_length": int(args.eval_max_length) if args.eval_max_length is not None else None,
        "input_k": int(args.input_k),
        "limit": int(args.limit) if args.limit is not None else None,
        "mock_reranker": bool(getattr(args, "mock_reranker", False)),
    }


def run_signature(args, checkpoint: dict, input_identity: str) -> dict:
    signature = rerank_cache_signature(args, checkpoint, input_identity)
    signature.update({
        "generator_model": args.generator_model,
        "generator_backend": args.generator_backend,
        "generator_prompt_path": str(Path(args.generator_prompt_path)),
        "generator_prompt_fingerprint": file_content_fingerprint(args.generator_prompt_path),
        "temperature": float(args.temperature),
        "seed": int(args.seed),
        "output_k": int(args.output_k),
    })
    return signature


def validate_rerank_cache_config(path: Path, current_signature: dict) -> None:
    if not path.exists():
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved_signature = saved.get("rerank_cache_signature")
    if saved_signature is None:
        raise ValueError(f"Existing Study B rerank config has no rerank_cache_signature: {path}")
    mismatches = {key: (saved_signature.get(key), value) for key, value in current_signature.items() if saved_signature.get(key) != value}
    if mismatches:
        raise ValueError(f"Existing Study B rerank cache is incompatible: {mismatches}")


def prepare_context(args):
    records, skips = load_prepared_records(args.test_jsonl, return_skips=True)
    split_counts = count_records_by_split(records)
    document_id_audit = apply_document_id_strategy(records)
    creator_prior_attentions = load_creator_prior_attention_values(args.raw_data_dir)
    validate_prepared_candidate_prior_attention(records, creator_prior_attentions)
    checkpoint = detect_checkpoint(args.model_path, args.model_type, args.loss_type)
    cfg = checkpoint["run_config"]
    args.eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else int(cfg.get("eval_batch_size", 4))
    args.eval_max_length, eval_length_source = resolve_eval_max_length(args.eval_max_length, cfg)
    return records, skips, split_counts, document_id_audit, checkpoint, test_jsonl_identity(Path(args.test_jsonl)), eval_length_source


def collect_fixed_candidates(args, records, skips, split_counts, document_id_audit):
    totals = defaultdict(int)
    totals["total_records"] = len(records) + len(skips)
    totals["skipped_missing_data"] = len(skips)
    skip_rows = list(skips)
    eligible = []
    for record in records:
        split = str(record.metadata.get("split", "test")).strip().lower() or "test"
        if not split_is_allowed(split, args.allowed_splits):
            totals["skipped_disallowed_split"] += 1
            skip_rows.append({"query_id": record.query_id, "reason": "disallowed_split", "split": split})
            continue
        fixed = fixed_from_prepared(record, args.input_k)
        validate_unique_document_ids(record.query_id, fixed.hits, "fixed Study B candidate set")
        eligible.append((record, fixed))
    if args.limit is not None:
        eligible = eligible[:args.limit]
    totals["eligible_query_count"] = len(eligible)
    totals["fixed_candidate_count_per_query"] = args.input_k
    totals["candidate_source"] = CANDIDATE_SOURCE
    totals["fixed_candidate_rule"] = FIXED_CANDIDATE_RULE
    return eligible, {
        **dict(totals),
        "allowed_splits": normalize_allowed_splits(args.allowed_splits),
        "query_count_by_split": {split: split_counts.get(split, 0) for split in normalize_allowed_splits(args.allowed_splits)},
        "document_id_audit": document_id_audit,
        "skip_rows": skip_rows,
    }


def qrels_for_record(record) -> Dict[str, float]:
    return {str(k): float(v) for k, v in convert_hits_to_graded_qrels([
        {"creator_id": hit.creator_id, "content": hit.content, "label": hit.label}
        for hit in record.hits
    ]).items()}


def reranker_metrics(query_id: str, qrels: Dict[str, float], ordered_ids: List[str], prior_attention: Dict[str, float]) -> dict:
    ndcg = graded_ndcg(query_id, qrels, ordered_ids, (5, 10))
    exposure_at = exposure_metrics(query_id, ordered_ids, prior_attention, (5, 10))
    return {
        "graded_nDCG_reranker@5": ndcg[5],
        "graded_nDCG_reranker@10": ndcg[10],
        "Exp_reranker@5": exposure_at["Exp@5"],
        "Exp_reranker@10": exposure_at["Exp@10"],
        "DExp_reranker@5": exposure_at["DExp@5"],
        "DExp_reranker@10": exposure_at["DExp@10"],
    }


def generated_metrics(query_id: str, qrels: Dict[str, float], position_preserving_ids: List[str], valid_ids: List[str], prior_attention: Dict[str, float], output_k: int, complete: bool) -> dict:
    ndcg = graded_ndcg(query_id, qrels, position_preserving_ids, (output_k,))
    generated_exposure = exposure_metrics(query_id, valid_ids, prior_attention, (output_k,)) if complete else {}
    return {
        f"graded_nDCG_generated@{output_k}": ndcg[output_k],
        f"Exp_generated@{output_k}": generated_exposure.get(f"Exp@{output_k}") if complete else None,
        f"DExp_generated@{output_k}": generated_exposure.get(f"DExp@{output_k}") if complete else None,
        "generated_metric_coverage": len(valid_ids) / output_k if output_k else 0.0,
    }


def build_rerank_cache_row(record, fixed, ranked: List[dict], checkpoint: dict, args) -> dict:
    fixed_ids = [hit.creator_id for hit in fixed.hits]
    top10 = ranked[:args.input_k]
    for idx, item in enumerate(top10, start=1):
        item["rank"] = idx
    ranked_ids = [item["creator_id"] for item in top10]
    assert_same_creator_set(record.query_id, fixed_ids, ranked_ids)
    qrels = qrels_for_record(record)
    prior_attention = {hit.creator_id: hit.prior_attention for hit in record.hits}
    metrics = reranker_metrics(record.query_id, qrels, ranked_ids, prior_attention)
    return {
        "query_id": record.query_id,
        "query": record.query,
        **metadata_for(record),
        "candidate_source": CANDIDATE_SOURCE,
        "fixed_candidate_rule": FIXED_CANDIDATE_RULE,
        "candidate_count": args.input_k,
        "loss_type": checkpoint["loss_type"],
        "fixed_candidate_details": fixed.details,
        "fixed_candidate_creator_ids": fixed_ids,
        "fixed_candidate_document_ids": [hit.document_id for hit in fixed.hits],
        "input_k": args.input_k,
        "output_k": args.output_k,
        "top_k_creator_ids": ranked_ids,
        "top_k_document_ids": [item["document_id"] for item in top10],
        "ranked_candidates": top10,
        "full_qrels": qrels,
        "candidate_prior_attention": prior_attention,
        "reranker_metrics": metrics,
    }


def validate_cached_ranked_rows(rows: List[dict], input_k: int) -> None:
    for row in rows:
        query_id = row.get("query_id", "unknown")
        fixed = [str(cid) for cid in row.get("fixed_candidate_creator_ids") or []]
        ranked = row.get("ranked_candidates") or []
        ranked_ids = [str(item.get("creator_id")) for item in ranked]
        assert_same_creator_set(query_id, fixed, ranked_ids)
        if len(ranked) != input_k:
            raise ValueError(f"query_id={query_id} has {len(ranked)} ranked candidates; expected {input_k}")
        if [int(item.get("rank")) for item in ranked] != list(range(1, input_k + 1)):
            raise ValueError(f"query_id={query_id} ranks must be 1..{input_k}")
        scores = [float(item["reranker_score"]) for item in ranked]
        if scores != sorted(scores, reverse=True):
            raise ValueError(f"query_id={query_id} ranked candidates are not sorted by reranker_score descending")


def run_rerank_stage(args, output_dir: Path, records, skips, split_counts, document_id_audit, checkpoint, input_identity, eval_length_source):
    signature = rerank_cache_signature(args, checkpoint, input_identity)
    if not args.overwrite:
        validate_rerank_cache_config(output_dir / "rerank_config.json", signature)
    config = vars(args).copy()
    config["detected_checkpoint"] = checkpoint
    config["rerank_cache_signature"] = signature
    config["resolved_eval_settings"] = {"eval_batch_size": args.eval_batch_size, "eval_max_length": args.eval_max_length, "eval_max_length_source": eval_length_source}
    write_json(output_dir / "rerank_config.json", config)
    eligible, eligibility = collect_fixed_candidates(args, records, skips, split_counts, document_id_audit)
    fixed_rows = [
        {
            "query_id": record.query_id,
            "query": record.query,
            **metadata_for(record),
            **fixed.details,
        }
        for record, fixed in eligible
    ]
    (output_dir / "fixed_candidates.jsonl").write_text(
        "".join(json_dumps_strict(row, ensure_ascii=False) + "\n" for row in fixed_rows),
        encoding="utf-8",
    )
    cached = {row["query_id"]: row for row in unique_by_qid(read_jsonl(output_dir / "ranked_inputs.jsonl"))}
    missing = [(record, fixed) for record, fixed in eligible if record.query_id not in cached]
    if missing:
        ranker = build_ranker(checkpoint, args)
        for record, fixed in missing:
            ranked = rank_candidates(ranker, record, fixed.hits, args)
            row = build_rerank_cache_row(record, fixed, ranked, checkpoint, args)
            append_jsonl(output_dir / "ranked_inputs.jsonl", row)
    target_qids = {record.query_id for record, _ in eligible}
    rows = [row for row in compact_jsonl(output_dir / "ranked_inputs.jsonl") if row.get("query_id") in target_qids]
    validate_cached_ranked_rows(rows, args.input_k)
    eligibility["rerank_cache_complete"] = len(rows) == len(target_qids)
    eligibility["reranked_query_count"] = len(rows)
    write_json(output_dir / "eligibility_summary.json", eligibility)
    return rows


def load_validated_rerank_cache(cache_dir: Path, args, checkpoint, input_identity) -> List[dict]:
    validate_rerank_cache_config(cache_dir / "rerank_config.json", rerank_cache_signature(args, checkpoint, input_identity))
    rows = unique_by_qid(read_jsonl(cache_dir / "ranked_inputs.jsonl"))
    if not rows:
        raise ValueError(f"Study B rerank cache is empty: {cache_dir / 'ranked_inputs.jsonl'}")
    validate_cached_ranked_rows(rows, args.input_k)
    summary = read_json(cache_dir / "eligibility_summary.json")
    expected = int(summary.get("eligible_query_count", len(rows)))
    if len(rows) != expected:
        raise ValueError(f"Study B rerank cache has {len(rows)} rows; eligibility_summary expects {expected}")
    return rows


def load_completed_valid(path: Path) -> set:
    return {row["query_id"] for row in unique_by_qid(read_jsonl(path)) if row.get("generation_success")}


def run_generate_stage(args, output_dir: Path, cache_dir: Path, checkpoint, input_identity, eval_length_source):
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_rows = load_validated_rerank_cache(cache_dir, args, checkpoint, input_identity)
    signature = run_signature(args, checkpoint, input_identity)
    if not args.overwrite:
        validate_resume_config(output_dir / "config.json", signature)
    config = vars(args).copy()
    config["detected_checkpoint"] = checkpoint
    config["run_signature"] = signature
    config["rerank_cache_dir"] = str(cache_dir)
    config["resolved_eval_settings"] = {"eval_batch_size": args.eval_batch_size, "eval_max_length": args.eval_max_length, "eval_max_length_source": eval_length_source}
    write_json(output_dir / "config.json", config)
    (output_dir / "ranked_inputs.jsonl").write_text("".join(json_dumps_strict(row, ensure_ascii=False) + "\n" for row in cache_rows), encoding="utf-8")
    completed = set() if args.overwrite else load_completed_valid(output_dir / "generations.jsonl")
    totals = defaultdict(int)
    for row in cache_rows:
        if row["query_id"] in completed:
            totals["resumed_completed"] += 1
            continue
        profiles = [{"creator_id": item["creator_id"], "document_id": item["document_id"], "content": item["content"]} for item in row["ranked_candidates"][:args.input_k]]
        ids = [item["creator_id"] for item in profiles]
        doc_ids = [item["document_id"] for item in profiles]
        prompt = build_prompt(row["query"], profiles, args.output_k, args.generator_prompt_path)
        raw = ""
        parsed = None
        errors = []
        validation = {"recommendations": [], "valid_recommendations": []}
        ok = False
        try:
            raw = make_echo_generation(profiles, args.output_k) if args.generator_backend == "echo" else call_ollama(args.ollama_base_url, args.generator_model, prompt, args.temperature, args.output_k, args.seed, ids, doc_ids)
            parsed = parse_json_output(raw)
            ok, errors, validation = validate_generation(parsed, profiles, args.output_k, row["query_id"])
        except Exception as exc:
            errors = [str(exc)]
        valid_ids = [rec["creator_id"] for rec in validation.get("valid_recommendations", [])]
        position_preserving = validation.get("position_preserving_creator_ids", [f"__INVALID_{row['query_id']}_{idx}" for idx in range(1, args.output_k + 1)])
        metrics = dict(row["reranker_metrics"])
        complete = bool(ok and len(valid_ids) == args.output_k)
        metrics.update(generated_metrics(row["query_id"], row["full_qrels"], position_preserving, valid_ids, row["candidate_prior_attention"], args.output_k, complete))
        metrics.update({
            "query_id": row["query_id"],
            **{key: row.get(key) for key in PROJECT_FIELDS},
            "loss_type": checkpoint["loss_type"],
            "parse_success": float(parsed is not None),
            "generation_success": float(complete),
            "citation_id_validity_rate": validation.get("citation_id_validity_rate", 0.0),
            "valid_citation_rate": validation.get("valid_citation_rate", 0.0),
            "creator_document_match_rate": validation.get("creator_document_match_rate", 0.0),
            "hallucinated_creator_rate": validation.get("hallucinated_creator_rate", 0.0),
            "duplicate_creator_rate": validation.get("duplicate_creator_rate", 0.0),
            "output_completeness": validation.get("output_completeness", 0.0),
            "exact_output_validity": validation.get("exact_output_validity", 0.0),
        })
        generation = {
            "query_id": row["query_id"],
            "query": row["query"],
            **{key: row.get(key) for key in PROJECT_FIELDS},
            "loss_type": checkpoint["loss_type"],
            "input_k": args.input_k,
            "output_k": args.output_k,
            "fixed_candidate_creator_ids": row["fixed_candidate_creator_ids"],
            "candidate_source": row.get("candidate_source", CANDIDATE_SOURCE),
            "fixed_candidate_rule": row.get("fixed_candidate_rule", FIXED_CANDIDATE_RULE),
            "top_k_creator_ids": ids,
            "top_k_document_ids": doc_ids,
            "ranked_candidates": row["ranked_candidates"],
            "generated_recommendations": validation.get("recommendations", []),
            "valid_recommendations": validation.get("valid_recommendations", []),
            "position_preserving_creator_ids": position_preserving,
            "raw_output": raw,
            "parsed_output": parsed,
            "prompt": prompt,
            "errors": errors,
            "generation_success": complete,
            "metrics": metrics,
        }
        append_jsonl(output_dir / "generations.jsonl", generation)
        if not complete:
            append_jsonl(output_dir / "generation_failures.jsonl", generation)
        totals["attempted_generations"] += 1
        totals["successful_generations"] += int(complete)
        totals["failed_generations"] += int(not complete)
    generations = compact_jsonl(output_dir / "generations.jsonl")
    rebuild_failure_file(output_dir, generations)
    eligibility = read_json(cache_dir / "eligibility_summary.json")
    eligibility.update({"rerank_cache_dir": str(cache_dir)})
    write_json(output_dir / "eligibility_summary.json", eligibility)
    rebuild_reports(output_dir, checkpoint, args)
    logging.info("Study B generate stage complete: %s", dict(totals))


def provenance(output_dir: Path, checkpoint: dict, args) -> dict:
    signature = read_json(output_dir / "config.json").get("run_signature", {})
    row = summary_provenance(output_dir, checkpoint)
    row.update({
        "candidate_source": CANDIDATE_SOURCE,
        "fixed_candidate_rule": FIXED_CANDIDATE_RULE,
        "reranker_checkpoint_fingerprint": signature.get("checkpoint_fingerprint") or checkpoint_fingerprint(args.model_path),
        "generator_temperature": float(args.temperature),
        "generator_seed": int(args.seed),
        "generator_prompt_fingerprint": signature.get("generator_prompt_fingerprint"),
        "eval_max_length": args.eval_max_length,
        "input_k": args.input_k,
        "output_k": args.output_k,
        "allowed_splits": ",".join(normalize_allowed_splits(args.allowed_splits)),
        "test_data_identity": signature.get("test_data_identity"),
        "study_output_dir": str(output_dir),
    })
    return row


def rebuild_reports(output_dir: Path, checkpoint: dict, args) -> None:
    generations = unique_by_qid(read_jsonl(output_dir / "generations.jsonl"))
    ranked_rows = unique_by_qid(read_jsonl(output_dir / "ranked_inputs.jsonl"))
    skill_by_qid = per_query_skill_metrics(
        ranked_rows=ranked_rows,
        generation_rows=generations,
        source=args.raw_data_dir,
        cache_dir=output_dir,
        input_k=args.input_k,
        output_k=args.output_k,
    )
    for generation in generations:
        metrics = generation.get("metrics")
        if isinstance(metrics, dict):
            metrics.update(skill_by_qid.get(generation.get("query_id"), {
                "skill_coverage_before@10": None,
                "skill_coverage_before@5": None,
                "skill_coverage_after@5": None,
                "delta_skill_coverage@5": None,
            }))
    metric_rows = [row["metrics"] for row in generations if row.get("metrics")]
    write_csv(output_dir / "per_query_metrics.csv", metric_rows)
    creator_prior_attentions = load_creator_prior_attention_values(args.raw_data_dir)
    prior_attention_groups, thresholds = build_prior_attention_groups(creator_prior_attentions)
    thresholds["raw_data_dir_path"] = str(args.raw_data_dir)
    write_json(output_dir / "prior_attention_group_thresholds.json", thresholds)
    skill_context = load_skill_context(args.raw_data_dir, output_dir)
    candidate_rows = build_candidate_rows(ranked_rows, generations, prior_attention_groups, creator_prior_attentions, skill_context)
    regression_rows = model_rows(candidate_rows)
    write_csv(output_dir / "generator_selection_candidates.csv", candidate_rows)
    write_csv(output_dir / "generator_selection_analysis.csv", regression_rows)
    summary = {
        "num_queries": len(metric_rows),
        "successful_generation_count": sum(bool(row.get("generation_success")) for row in generations),
        "failed_generation_count": sum(not bool(row.get("generation_success")) for row in generations),
        **skill_metric_summary(metric_rows),
        **aggregate_metric_rows(metric_rows, excluded={"query_id", *PROJECT_FIELDS}),
        **group_rank_summary(ranked_rows, prior_attention_groups, args.input_k, args.output_k),
        **generated_selection_summary(generations, prior_attention_groups, args.input_k),
        **provenance(output_dir, checkpoint, args),
    }
    write_csv(output_dir / "position_selection.csv", position_selection_rows(generations, args.input_k))
    write_csv(output_dir / "summary_metrics.csv", [summary])
    eligibility = read_json(output_dir / "eligibility_summary.json")
    allow_master_upsert = (
        args.limit is None
        and not bool(getattr(args, "mock_reranker", False))
        and str(args.generator_backend).strip().lower() != "echo"
    )
    if allow_master_upsert:
        expected = int(eligibility.get("eligible_query_count", len(generations)))
        if summary["successful_generation_count"] != expected or summary["failed_generation_count"] != 0:
            raise RuntimeError(f"Study B generation incomplete; refusing master CSV upsert for {output_dir}")
        upsert_master_results_row(master_results_path(args.output_root or output_dir), summary)


def main():
    args = parse_args()
    validate_cli_values(args)
    args.allowed_splits = normalize_allowed_splits(args.allowed_splits)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.rerank_cache_dir) if args.rerank_cache_dir else output_dir
    setup_logging(output_dir)
    if args.stage in {"rerank", "all"}:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clear_stage_output_files(args.stage, output_dir, cache_dir)
    records, skips, split_counts, document_id_audit, checkpoint, input_identity, eval_length_source = prepare_context(args)
    logging.info("Detected model_type=%s loss_type=%s eval_batch_size=%s eval_max_length=%s source=%s", checkpoint["model_type"], checkpoint["loss_type"], args.eval_batch_size, args.eval_max_length, eval_length_source)
    if args.stage in {"rerank", "all"}:
        run_rerank_stage(args, cache_dir, records, skips, split_counts, document_id_audit, checkpoint, input_identity, eval_length_source)
    if args.stage in {"generate", "all"}:
        run_generate_stage(args, output_dir, cache_dir, checkpoint, input_identity, eval_length_source)
    logging.info("Study B %s stage complete", args.stage)


if __name__ == "__main__":
    main()
