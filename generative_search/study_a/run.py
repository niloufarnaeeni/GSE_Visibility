import argparse
import csv
import hashlib
import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import median
import tempfile
from typing import Dict, List, Optional

try:
    import fcntl
except ImportError:
    class _FcntlFallback:
        LOCK_EX = 0
        LOCK_UN = 0

        @staticmethod
        def flock(*_args, **_kwargs):
            return None

    fcntl = _FcntlFallback()

from .checkpoint import detect_checkpoint
from .candidates import SUPPORTED_CANDIDATE_SOURCES, build_candidate_source
from .data import Hit, load_records, stable_content_document_id, test_data_identity, test_jsonl_identity
from .generator import DEFAULT_PROMPT_PATH, build_prompt, call_ollama, parse_json_output, validate_generation
from .metrics import graded_ndcg, exposure_metrics
from .skill_analysis import per_query_skill_metrics, skill_metric_summary
from rag_retrieval.infer.eval.eval_metrics import _candidate_run_metadata_paths, _load_config_file

RUN_FILES = (
    "ranked_inputs.jsonl", "generations.jsonl", "generation_failures.jsonl",
    "per_query_metrics.csv", "summary_metrics.csv", "position_selection.csv",
    "generator_selection_candidates.csv", "generator_selection_analysis.csv",
    "prior_attention_group_selection.csv", "prior_attention_group_stage_analysis.csv", "creator_coverage.csv",
    "generator_textual_visibility.csv", "prior_attention_group_textual_visibility.csv",
    "prior_attention_group_thresholds.json",
)
RERANK_CACHE_FILES = ("ranked_inputs.jsonl", "rerank_config.json", "eligibility_summary.json")
PROJECT_FIELDS = ("project_name", "view", "project_variant_type", "source_project_file", "source_project_row", "project_variant_index", "trial_id", "fold_id", "split")
SUMMARY_AVERAGE_EXCLUDED_KEYS = {"query_id", *PROJECT_FIELDS}


def parse_args():
    p = argparse.ArgumentParser(description="Study A downstream generative-search effectiveness experiment")
    p.add_argument("--test_jsonl", required=True); p.add_argument("--model_path", default=None)
    p.add_argument("--stage", default="all", choices=["all", "rerank", "generate"])
    p.add_argument("--rerank_cache_dir", default=None, help="Directory containing checkpoint-level rerank cache for --stage generate")
    p.add_argument("--allowed_splits", default="test")
    p.add_argument("--generator_backend", default="ollama", choices=["ollama", "echo"])
    p.add_argument("--generator_model", default="mistral-nemo:12b"); p.add_argument("--ollama_base_url", default="http://localhost:11434")
    p.add_argument("--generator_prompt_path", default=str(DEFAULT_PROMPT_PATH))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--raw_data_dir", default=None, help="Raw data directory containing prior_attention_scores.csv, creator_details.csv, and skill analysis resources")
    p.add_argument("--candidate_source", default="full", choices=SUPPORTED_CANDIDATE_SOURCES)
    p.add_argument("--retrieval_k", type=int, default=None)
    p.add_argument("--retrieval_method", default=None, help="First-stage retrieval method used to prepare preselected candidates, e.g. bm25 or dense")
    p.add_argument("--embedding_model", default=None, help="Embedding model used by dense first-stage retrieval, if any")
    p.add_argument("--no_reranking", action="store_true", help="Use first-stage candidate order directly instead of reranking")
    p.add_argument("--input_k", type=int, default=10); p.add_argument("--output_k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--eval_batch_size", type=int, default=None); p.add_argument("--eval_max_length", type=int, default=None)
    p.add_argument("--device_map", default=None); p.add_argument("--model_type", default=None); p.add_argument("--loss_type", default=None)
    p.add_argument("--overwrite", action="store_true"); p.add_argument("--limit", type=int, default=None); p.add_argument("--mock_reranker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(output_dir / "run.log", mode="a", encoding="utf-8"), logging.StreamHandler()])


def to_jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def json_dumps_strict(value, **kwargs) -> str:
    return json.dumps(to_jsonable(value), allow_nan=False, **kwargs)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f: f.write(json_dumps_strict(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict) -> None: path.write_text(json_dumps_strict(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows: path.write_text("", encoding="utf-8"); return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def unique_by_qid(rows: List[dict]) -> List[dict]:
    return list({row.get("query_id"): row for row in rows if row.get("query_id")}.values())


def load_completed_valid(path: Path) -> set:
    return {row["query_id"] for row in unique_by_qid(read_jsonl(path)) if row.get("generation_success")}


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def candidate_count_stats(values: List[int]) -> dict:
    if not values:
        return {"min_candidate_count": 0, "mean_candidate_count": 0.0, "median_candidate_count": 0.0, "max_candidate_count": 0}
    return {
        "min_candidate_count": min(values),
        "mean_candidate_count": mean([float(value) for value in values]),
        "median_candidate_count": float(median(values)),
        "max_candidate_count": max(values),
    }


def audit_document_ids(records: List) -> dict:
    explicit_document_id_count = 0
    content_by_creator = defaultdict(set)
    for record in records:
        for hit in record.hits:
            if hit.document_id_source == "input":
                explicit_document_id_count += 1
            content_by_creator[hit.creator_id].add(hit.content)
    multi_profile_creators = sorted(cid for cid, contents in content_by_creator.items() if len(contents) > 1)
    return {
        "explicit_document_id_count": explicit_document_id_count,
        "fallback_document_id_strategy": "DOC_{creator_id}",
        "creator_count": len(content_by_creator),
        "multi_profile_creator_count": len(multi_profile_creators),
        "multi_profile_creator_examples": multi_profile_creators[:20],
        "one_stable_profile_per_creator": len(multi_profile_creators) == 0,
    }


def apply_document_id_strategy(records: List) -> dict:
    audit = audit_document_ids(records)
    if audit["one_stable_profile_per_creator"]:
        audit["effective_document_id_strategy"] = "DOC_{creator_id}"
        return audit

    content_by_creator = defaultdict(set)
    for record in records:
        for hit in record.hits:
            content_by_creator[hit.creator_id].add(hit.content)
    multi_profile_creators = {cid for cid, contents in content_by_creator.items() if len(contents) > 1}
    for record in records:
        for hit in record.hits:
            if hit.document_id_source == "fallback" and hit.creator_id in multi_profile_creators:
                hit.document_id = stable_content_document_id(hit.creator_id, hit.content)
                hit.document_id_source = "content_hash_fallback"
    audit = audit_document_ids(records)
    audit["effective_document_id_strategy"] = "DOC_{creator_id}_{sha256(creator_id,content)[:16]} for multi-profile fallback creators"
    return audit


def validate_cli_values(args) -> None:
    if args.temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive when supplied")
    if args.input_k <= 0 or args.output_k <= 0:
        raise ValueError("input_k and output_k must be positive")
    if args.output_k > args.input_k:
        raise ValueError("output_k must be less than or equal to input_k")
    if args.retrieval_k is not None and args.retrieval_k <= 0:
        raise ValueError("retrieval_k must be positive")
    normalize_allowed_splits(getattr(args, "allowed_splits", "test"))
    if not getattr(args, "no_reranking", False) and not args.model_path:
        raise ValueError("--model_path is required unless --no_reranking is set")
    if getattr(args, "no_reranking", False) and (args.model_type or args.loss_type):
        raise ValueError("--model_type and --loss_type are reranker options and cannot be used with --no_reranking")
    if getattr(args, "no_reranking", False) and args.candidate_source != "preselected":
        raise ValueError("--no_reranking requires --candidate_source preselected so first-stage candidate order is preserved")


def normalize_allowed_splits(value) -> List[str]:
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        raise ValueError("allowed_splits must be a comma-separated string or a sequence of split names")
    normalized = sorted({str(part).strip().lower() for part in parts if str(part).strip()})
    if not normalized:
        raise ValueError("allowed_splits must include at least one split")
    return normalized


def split_is_allowed(split_value, allowed_splits: List[str]) -> bool:
    split = str(split_value if split_value is not None else "test").strip().lower() or "test"
    return split in set(normalize_allowed_splits(allowed_splits))


def resolve_eval_max_length(cli_value, run_config: dict, repository_default: int = 256):
    if cli_value is not None:
        return int(cli_value), "cli_override"
    for key in ("eval_max_length", "max_length", "max_len"):
        if run_config.get(key) is not None:
            return int(run_config[key]), key
    return repository_default, "repository_default"


CHECKPOINT_FINGERPRINT_FILES = {
    "config.json", "run_config.json", "run_config.yaml", "run_config.yml",
    "adapter_config.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.txt", "merges.txt", "tokenizer.json",
    "sentencepiece.bpe.model", "spiece.model", "training_args.bin",
}
CHECKPOINT_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}
MAX_HASH_BYTES = 2 * 1024 * 1024
HASH_CHUNK_BYTES = 8 * 1024 * 1024
BLANK_PROVENANCE_STRINGS = {"", "none", "null", "unknown"}
METHOD_NAME_BY_LOSS_TYPE = {
    "ranknet": "RankNet",
    "ear": "EAR",
    "ear_sym": "EAR-Sym",
    "pairwise_reg": "Pairwise Reg",
    "boratto_reg": "Boratto-reg",
    "pal": "PAL",
    "pbiloss_popneg_ft": "PBiLoss",
    "no_reranking": "No Reranking",
}
MASTER_RESULTS_FILENAME = "study_a_all_results.csv"
MASTER_IDENTITY_COLUMNS = (
    "loss_type",
    "reranker_checkpoint_fingerprint",
    "generator_backend",
    "generator_model",
    "generator_temperature",
    "generator_seed",
    "generator_prompt_fingerprint",
    "eval_max_length",
    "input_k",
    "output_k",
    "candidate_source",
    "reranking_mode",
    "retrieval_method",
    "retrieval_k",
    "embedding_model",
    "allowed_splits",
    "test_data_identity",
)


def file_content_fingerprint(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _streaming_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    item = {"path": path.name, "size": stat.st_size}
    if path.suffix in CHECKPOINT_WEIGHT_SUFFIXES:
        item["sha256"] = _streaming_sha256(path)
    elif stat.st_size <= MAX_HASH_BYTES or path.name in CHECKPOINT_FINGERPRINT_FILES:
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return item


def checkpoint_fingerprint(model_path: str) -> str:
    root = Path(model_path)
    if root.is_file():
        payload = {"kind": "file", **_file_fingerprint(root)}
    elif root.is_dir():
        files = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            rel = path.relative_to(root).as_posix()
            if path.name in CHECKPOINT_FINGERPRINT_FILES or path.suffix in CHECKPOINT_WEIGHT_SUFFIXES:
                entry = _file_fingerprint(path)
                entry["path"] = rel
                files.append(entry)
        if not files:
            files = []
            for path in sorted(item for item in root.iterdir() if item.is_file())[:20]:
                entry = _file_fingerprint(path)
                entry["path"] = path.relative_to(root).as_posix()
                files.append(entry)
        payload = {"kind": "directory", "files": files}
    else:
        payload = {"kind": "missing", "path": str(root)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def assert_qrels_creator_ids(query_id: str, qrels: Dict[str, float], hits: List[Hit], label: str) -> None:
    expected_ids = {hit.creator_id for hit in hits if hit.label > 0}
    actual_ids = set(qrels)
    if actual_ids != expected_ids:
        raise ValueError(f"{label} qrels creator ID mismatch for query_id={query_id}: expected={sorted(expected_ids)}, actual={sorted(actual_ids)}")


def validate_unique_document_ids(query_id: str, hits: List[Hit], label: str) -> None:
    document_ids = [str(hit.document_id).strip() for hit in hits if str(hit.document_id).strip()]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError(f"{label} document_id values are not unique for query_id={query_id}")


def _checkpoint_path_for_signature(args):
    if getattr(args, "no_reranking", False):
        return None
    return str(Path(args.model_path)) if args.model_path else None


def _checkpoint_fingerprint_for_signature(args):
    if getattr(args, "no_reranking", False):
        return None
    return checkpoint_fingerprint(args.model_path) if args.model_path else None


def run_signature(args, checkpoint: dict, records, test_identity_value: Optional[str] = None) -> dict:
    signature = {
        "test_data_identity": test_identity_value or test_data_identity(records),
        "checkpoint": _checkpoint_path_for_signature(args),
        "checkpoint_fingerprint": _checkpoint_fingerprint_for_signature(args),
        "model_type": checkpoint["model_type"],
        "loss_type": checkpoint["loss_type"],
        "reranking_mode": "no_reranking" if getattr(args, "no_reranking", False) else "reranker",
        "candidate_source": getattr(args, "candidate_source", "full"),
        "retrieval_method": getattr(args, "retrieval_method", None),
        "allowed_splits": normalize_allowed_splits(getattr(args, "allowed_splits", "test")),
        "generator_model": args.generator_model,
        "generator_backend": args.generator_backend,
        "generator_prompt_path": str(Path(getattr(args, "generator_prompt_path", DEFAULT_PROMPT_PATH))),
        "generator_prompt_fingerprint": file_content_fingerprint(getattr(args, "generator_prompt_path", DEFAULT_PROMPT_PATH)),
        "temperature": float(args.temperature),
        "eval_max_length": int(args.eval_max_length) if args.eval_max_length is not None else None,
        "input_k": int(args.input_k),
        "output_k": int(args.output_k),
        "seed": int(args.seed),
        "retrieval_k": None,
        "embedding_model": getattr(args, "embedding_model", None),
    }
    if signature["candidate_source"] != "full":
        signature["retrieval_k"] = int(getattr(args, "retrieval_k", 50))
    return signature


def rerank_cache_signature(args, checkpoint: dict, records, test_identity_value: Optional[str] = None) -> dict:
    signature = {
        "test_data_identity": test_identity_value or test_data_identity(records),
        "checkpoint": _checkpoint_path_for_signature(args),
        "checkpoint_fingerprint": _checkpoint_fingerprint_for_signature(args),
        "model_type": checkpoint["model_type"],
        "loss_type": checkpoint["loss_type"],
        "reranking_mode": "no_reranking" if getattr(args, "no_reranking", False) else "reranker",
        "candidate_source": getattr(args, "candidate_source", "full"),
        "retrieval_method": getattr(args, "retrieval_method", None),
        "allowed_splits": normalize_allowed_splits(getattr(args, "allowed_splits", "test")),
        "eval_max_length": int(args.eval_max_length) if args.eval_max_length is not None else None,
        "input_k": int(args.input_k),
        "retrieval_k": None,
        "embedding_model": getattr(args, "embedding_model", None),
    }
    if signature["candidate_source"] != "full":
        signature["retrieval_k"] = int(getattr(args, "retrieval_k", 50))
    return signature


def validate_resume_config(path: Path, current_signature: dict) -> None:
    if not path.exists():
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved_signature = saved.get("run_signature")
    if saved_signature is None:
        raise ValueError(f"Existing config {path} has no run_signature; use --overwrite or a new output directory")
    mismatches = {key: (saved_signature.get(key), value) for key, value in current_signature.items() if saved_signature.get(key) != value}
    if mismatches:
        raise ValueError(f"Existing Study A run is incompatible; use --overwrite or a new output directory. Mismatches: {mismatches}")


def validate_rerank_cache_config(path: Path, current_signature: dict) -> None:
    if not path.exists():
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved_signature = saved.get("rerank_cache_signature")
    if saved_signature is None:
        raise ValueError(f"Existing rerank cache config {path} has no rerank_cache_signature; use --overwrite or a new cache directory")
    mismatches = {key: (saved_signature.get(key), value) for key, value in current_signature.items() if saved_signature.get(key) != value}
    if mismatches:
        raise ValueError(f"Existing rerank cache is incompatible with this checkpoint/data/input setup. Mismatches: {mismatches}")


def validate_ranked_output(ranked, candidate_hits, input_k: int) -> None:
    results = getattr(ranked, "results", None)
    candidate_count = len(candidate_hits)
    if not isinstance(results, list) or len(results) != candidate_count:
        raise ValueError(f"Malformed reranker output: expected {candidate_count} results, got {len(results) if isinstance(results, list) else type(results).__name__}")
    validate_unique_document_ids("ranked_candidates", candidate_hits, "candidate pool")
    valid_doc_ids = set(range(candidate_count))
    seen_doc_ids = set(); seen_creators = set(); seen_ranks = set()
    for result in results:
        try:
            doc_id = int(result.doc_id); rank = int(result.rank); creator_id = str(candidate_hits[doc_id].creator_id)
        except (AttributeError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"Malformed reranker result: invalid doc_id/rank: {exc}") from exc
        if doc_id not in valid_doc_ids or doc_id in seen_doc_ids:
            raise ValueError(f"Malformed reranker output: doc_id {doc_id} is invalid or duplicated")
        if rank in seen_ranks or rank < 1 or rank > candidate_count:
            raise ValueError(f"Malformed reranker output: rank {rank} is invalid or duplicated")
        if creator_id in seen_creators:
            raise ValueError(f"Malformed reranker output: creator_id {creator_id} is duplicated")
        seen_doc_ids.add(doc_id); seen_ranks.add(rank); seen_creators.add(creator_id)
    if seen_doc_ids != valid_doc_ids:
        raise ValueError(f"Malformed reranker output: every candidate must appear exactly once; got doc_ids={sorted(seen_doc_ids)}")
    if seen_ranks != set(range(1, candidate_count + 1)):
        raise ValueError(f"Malformed reranker output: ranks must be 1..{candidate_count}, got {sorted(seen_ranks)}")
    ordered_by_rank = sorted(results, key=lambda result: int(result.rank))
    ordered = sorted(results, key=lambda result: (-float(result.score), int(result.rank)))
    if [int(result.doc_id) for result in ordered_by_rank] != [int(result.doc_id) for result in ordered]:
        raise ValueError("Malformed reranker output: rank ordering does not agree with score ordering")
    top_doc_ids = [int(result.doc_id) for result in ordered[:input_k]]
    top_creators = [candidate_hits[doc_id].creator_id for doc_id in top_doc_ids]
    if len(top_creators) != input_k or len(set(top_creators)) != input_k:
        raise ValueError(f"Malformed reranker output: top_k must contain exactly {input_k} unique creators")


def clean_provenance_value(value):
    if isinstance(value, str) and value.strip().lower() in BLANK_PROVENANCE_STRINGS:
        return None
    return value


def clean_first_present(*values):
    for value in values:
        cleaned = clean_provenance_value(value)
        if cleaned is not None:
            return cleaned
    return None


def method_name_for_loss(loss_type: Optional[str]) -> Optional[str]:
    cleaned = clean_provenance_value(loss_type)
    if cleaned is None:
        return None
    text = str(cleaned)
    return METHOD_NAME_BY_LOSS_TYPE.get(text, text.replace("_", " ").strip().title())


def study_master_results_path(output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    return output_dir.parent.parent / MASTER_RESULTS_FILENAME


def _csv_cell_string(value) -> str:
    return "" if value is None else str(value)


def _csv_identity_key(row: dict) -> tuple:
    return tuple(_csv_cell_string(row.get(column, "")) for column in MASTER_IDENTITY_COLUMNS)


def _read_csv_rows(path: Path) -> List[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def upsert_master_results_row(master_csv_path: Path, row: dict) -> None:
    master_csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = master_csv_path.with_suffix(master_csv_path.suffix + ".lock")
    identity = _csv_identity_key(row)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing_rows = _read_csv_rows(master_csv_path)
        kept_rows = [existing for existing in existing_rows if _csv_identity_key(existing) != identity]
        kept_rows.append({key: value for key, value in row.items()})
        fieldnames = sorted({key for existing in kept_rows for key in existing})
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=str(master_csv_path.parent),
            delete=False,
            prefix=master_csv_path.name + ".",
            suffix=".tmp",
        ) as tmp:
            writer = csv.DictWriter(tmp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)
            temp_path = Path(tmp.name)
        os.replace(temp_path, master_csv_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def summary_provenance(output_dir: Path, checkpoint: dict) -> dict:
    saved_config = read_json(output_dir / "config.json")
    run_signature_cfg = saved_config.get("run_signature") if isinstance(saved_config.get("run_signature"), dict) else {}
    run_cfg = checkpoint.get("run_config") if isinstance(checkpoint.get("run_config"), dict) else {}
    raw_run_cfg = {}
    model_path = checkpoint.get("model_path")
    if model_path:
        for cfg_path in _candidate_run_metadata_paths(Path(model_path)):
            candidate = _load_config_file(cfg_path)
            if isinstance(candidate, dict):
                raw_run_cfg = candidate
                break
    provenance_cfg = raw_run_cfg or run_cfg
    reranking_mode = clean_first_present(run_signature_cfg.get("reranking_mode"), saved_config.get("reranking_mode"))
    loss_type = "no_reranking" if reranking_mode == "no_reranking" else clean_provenance_value(checkpoint.get("loss_type"))
    reranker_checkpoint_path = None if loss_type == "no_reranking" else clean_first_present(run_signature_cfg.get("checkpoint"), checkpoint.get("model_path"))
    reranker_checkpoint_fp = None if loss_type == "no_reranking" else clean_first_present(
        run_signature_cfg.get("checkpoint_fingerprint"),
        checkpoint_fingerprint(checkpoint["model_path"]) if checkpoint.get("model_path") else None,
    )

    return {
        "loss_type": loss_type,
        "method_name": method_name_for_loss(loss_type),
        "reranker_model_name": None if loss_type == "no_reranking" else clean_provenance_value(provenance_cfg.get("model_name_or_path")),
        "reranker_checkpoint_path": reranker_checkpoint_path,
        "reranker_checkpoint_fingerprint": reranker_checkpoint_fp,
        "reranker_run_dir": None if loss_type == "no_reranking" else clean_first_present(provenance_cfg.get("run_dir"), provenance_cfg.get("output_dir")),
        "reranker_run_time": None if loss_type == "no_reranking" else clean_provenance_value(provenance_cfg.get("run_time")),
        "training_lambda": clean_first_present(provenance_cfg.get("training_lambda"), provenance_cfg.get("lambda")),
        "training_lambda_prior_attention": clean_first_present(provenance_cfg.get("training_lambda_prior_attention"), provenance_cfg.get("lambda_prior_attention")),
        "training_lambda_exposure": clean_first_present(provenance_cfg.get("training_lambda_exposure"), provenance_cfg.get("lambda_exposure")),
        "lambda_ranknet": clean_provenance_value(provenance_cfg.get("lambda_ranknet")),
        "lambda_kl": clean_provenance_value(provenance_cfg.get("lambda_kl")),
        "lambda_corr": clean_first_present(provenance_cfg.get("lambda_corr"), provenance_cfg.get("training_lambda_corr")),
        "generator_model": clean_first_present(run_signature_cfg.get("generator_model"), saved_config.get("generator_model")),
        "generator_backend": clean_first_present(run_signature_cfg.get("generator_backend"), saved_config.get("generator_backend")),
        "generator_temperature": clean_first_present(run_signature_cfg.get("temperature"), saved_config.get("temperature")),
        "generator_seed": clean_first_present(run_signature_cfg.get("seed"), saved_config.get("seed")),
        "generator_prompt_fingerprint": clean_first_present(run_signature_cfg.get("generator_prompt_fingerprint")),
        "eval_max_length": clean_first_present(run_signature_cfg.get("eval_max_length"), saved_config.get("eval_max_length")),
        "input_k": clean_first_present(run_signature_cfg.get("input_k"), saved_config.get("input_k")),
        "output_k": clean_first_present(run_signature_cfg.get("output_k"), saved_config.get("output_k")),
        "candidate_source": clean_first_present(run_signature_cfg.get("candidate_source"), saved_config.get("candidate_source")),
        "reranking_mode": reranking_mode,
        "retrieval_method": clean_first_present(run_signature_cfg.get("retrieval_method"), saved_config.get("retrieval_method")),
        "retrieval_k": clean_first_present(run_signature_cfg.get("retrieval_k"), saved_config.get("retrieval_k")),
        "embedding_model": clean_first_present(run_signature_cfg.get("embedding_model"), saved_config.get("embedding_model")),
        "allowed_splits": ",".join(normalize_allowed_splits(run_signature_cfg.get("allowed_splits", saved_config.get("allowed_splits", "test")))),
        "test_data_identity": clean_first_present(run_signature_cfg.get("test_data_identity")),
        "study_output_dir": str(output_dir),
    }


def count_records_by_split(records: List) -> dict:
    counts = defaultdict(int)
    for record in records:
        split = str(record.metadata.get("split", "test")).strip().lower() or "test"
        counts[split] += 1
    return dict(counts)


def make_echo_generation(profiles: List[dict], output_k: int) -> str:
    return json.dumps({"recommendations": [{"rank": i, "creator_id": row["creator_id"], "document_id": row["document_id"], "reason": "Selected from the supplied profiles."} for i, row in enumerate(profiles[:output_k], start=1)]})


def build_ranker(checkpoint: dict, args):
    if args.mock_reranker:
        class Result:
            def __init__(self, doc_id, text, score, rank): self.doc_id, self.text, self.score, self.rank = doc_id, text, score, rank
        class Ranked:
            def __init__(self, results): self.results = results
        class MockRanker:
            def rerank(self, query, docs, batch_size=4, max_length=256): return Ranked([Result(i, doc, float(len(docs)-i), i+1) for i, doc in enumerate(docs)])
        return MockRanker()
    run_cfg = checkpoint["run_config"]
    kwargs = {"model_type": checkpoint["model_type"], "verbose": 0}
    if checkpoint["model_type"] in {"llm-decoder", "llm"}:
        kwargs.update({"query_format": run_cfg.get("query_format", "query: {}"), "document_format": run_cfg.get("document_format", "document: {}"), "seq": run_cfg.get("seq", "\n"), "special_token": run_cfg.get("special_token", "\nrelevance"), "device_map": args.device_map})
    from rag_retrieval import Reranker
    ranker = Reranker(args.model_path, **kwargs)
    if ranker is None: raise RuntimeError(f"Repository reranker could not load supported architecture {checkpoint['model_type']!r}")
    return ranker


def rank_candidates(ranker, record, candidate_hits, args) -> List[dict]:
    if getattr(args, "no_reranking", False):
        out = []
        for idx, hit in enumerate(candidate_hits, start=1):
            out.append({
                "rank": idx,
                "creator_id": hit.creator_id,
                "document_id": hit.document_id,
                "content": hit.content,
                "label": hit.label,
                "prior_attention": hit.prior_attention,
                "reranker_score": float(len(candidate_hits) - idx + 1),
            })
        return out

    ranked = ranker.rerank(record.query, [h.content for h in candidate_hits], batch_size=args.eval_batch_size, max_length=args.eval_max_length)
    validate_ranked_output(ranked, candidate_hits, args.input_k)
    out = []
    for result in ranked.results:
        hit = candidate_hits[int(result.doc_id)]
        out.append({"rank": int(result.rank), "creator_id": hit.creator_id, "document_id": hit.document_id, "content": hit.content, "label": hit.label, "prior_attention": hit.prior_attention, "reranker_score": float(result.score)})
    out.sort(key=lambda item: (-item["reranker_score"], item["rank"]))
    return out


def before_generation_metrics(query_id: str, full_qrels: Dict[str, float], before_ids: List[str], prior_attention: Dict[str, float], input_k: int, output_k: int) -> dict:
    cutoffs = tuple(dict.fromkeys((int(output_k), int(input_k))))
    before_ndcg = graded_ndcg(query_id, full_qrels, before_ids, cutoffs)
    before_exposure = exposure_metrics(query_id, before_ids, prior_attention, cutoffs)
    row = {"query_id": query_id}
    for k in cutoffs:
        row[f"graded_nDCG_before@{k}"] = before_ndcg[k]
        row[f"nDCG_before@{k}"] = before_ndcg[k]
        row[f"Exp_before@{k}"] = before_exposure[f"Exp@{k}"]
        row[f"DExp_before@{k}"] = before_exposure[f"DExp@{k}"]
    return row


def metrics_from_cached_before(query_id: str, before_row: dict, full_qrels: Dict[str, float], position_preserving_after_ids: List[str], valid_after_ids: List[str], prior_attention: Dict[str, float], output_k: int, complete_after: bool) -> dict:
    after_ndcg = graded_ndcg(query_id, full_qrels, position_preserving_after_ids, (int(output_k),))
    complete = complete_after and len(valid_after_ids) == output_k
    after_exposure = exposure_metrics(query_id, valid_after_ids, prior_attention, (int(output_k),)) if complete else {}
    for key in (f"Exp@{output_k}", f"DExp@{output_k}"):
        if complete and key not in after_exposure: raise KeyError(f"Missing required generated metric {key}")

    row = dict(before_row)
    row["query_id"] = query_id
    row[f"graded_nDCG_after@{output_k}"] = after_ndcg[int(output_k)]
    row[f"nDCG_after@{output_k}"] = after_ndcg[int(output_k)]
    row[f"Exp_after@{output_k}"] = after_exposure[f"Exp@{output_k}"] if complete else None
    row[f"DExp_after@{output_k}"] = after_exposure[f"DExp@{output_k}"] if complete else None
    row["generated_metric_coverage"] = len(valid_after_ids) / output_k if output_k else 0.0
    row[f"delta_nDCG@{output_k}"] = row[f"graded_nDCG_after@{output_k}"] - row[f"graded_nDCG_before@{output_k}"]
    row[f"delta_Exp@{output_k}"] = row[f"Exp_after@{output_k}"] - row[f"Exp_before@{output_k}"] if complete else None
    row[f"delta_DExp@{output_k}"] = row[f"DExp_after@{output_k}"] - row[f"DExp_before@{output_k}"] if complete else None
    return row


def metrics_for_query(query_id: str, full_qrels: Dict[str, float], before_ids: List[str], position_preserving_after_ids: List[str], valid_after_ids: List[str], prior_attention: Dict[str, float], input_k: int, output_k: int, complete_after: bool) -> dict:
    before_row = before_generation_metrics(query_id, full_qrels, before_ids, prior_attention, input_k, output_k)
    return metrics_from_cached_before(query_id, before_row, full_qrels, position_preserving_after_ids, valid_after_ids, prior_attention, output_k, complete_after)


def metadata_for(record):
    return {key: record.metadata.get(key) for key in PROJECT_FIELDS}


def compact_jsonl(path: Path) -> List[dict]:
    rows = unique_by_qid(read_jsonl(path))
    if rows:
        path.write_text("".join(json_dumps_strict(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return rows


def clear_output_files(output_dir: Path, names) -> None:
    for name in names:
        path = output_dir / name
        if path.exists():
            path.unlink()


def validate_cached_ranked_rows(rows: List[dict], input_k: int) -> None:
    for row in rows:
        ranked = row.get("ranked_candidates", [])
        query_id = row.get("query_id", "unknown")
        if len(ranked) != input_k:
            raise ValueError(f"Rerank cache query_id={query_id} has {len(ranked)} ranked candidates; expected input_k={input_k}")
        scores = [float(item["reranker_score"]) for item in ranked]
        if scores != sorted(scores, reverse=True):
            raise ValueError(f"Rerank cache query_id={query_id} is not sorted by reranker_score descending")
        if [int(item["rank"]) for item in ranked] != list(range(1, input_k + 1)):
            raise ValueError(f"Rerank cache query_id={query_id} ranks must be exactly 1..{input_k}")
        creator_ids = [str(item["creator_id"]) for item in ranked]
        document_ids = [str(item["document_id"]) for item in ranked]
        if len(creator_ids) != len(set(creator_ids)):
            raise ValueError(f"Rerank cache query_id={query_id} has duplicate creator_id values")
        if len(document_ids) != len(set(document_ids)):
            raise ValueError(f"Rerank cache query_id={query_id} has duplicate document_id values")
        for item in ranked:
            for required in ("creator_id", "document_id", "content", "label", "prior_attention", "reranker_score"):
                if required not in item:
                    raise ValueError(f"Rerank cache query_id={query_id} candidate is missing {required}")


def synthetic_no_reranking_checkpoint() -> dict:
    return {
        "loss_type": "no_reranking",
        "model_type": "none",
        "model_path": None,
        "run_config": {
            "loss_type": "no_reranking",
            "model_type": "none",
            "model_name_or_path": None,
        },
    }


def _unique_nonblank_metadata(records, key: str):
    values = sorted({str(record.metadata.get(key)).strip() for record in records if record.metadata.get(key) not in (None, "")})
    if len(values) > 1:
        raise ValueError(f"Input records contain conflicting {key} values: {values}")
    return values[0] if values else None


def resolve_retrieval_provenance_from_records(args, records) -> None:
    input_method = _unique_nonblank_metadata(records, "retrieval_method")
    input_embedding = _unique_nonblank_metadata(records, "embedding_model")
    input_k = _unique_nonblank_metadata(records, "retrieval_k")

    if args.retrieval_method and input_method and str(args.retrieval_method) != input_method:
        raise ValueError(f"--retrieval_method={args.retrieval_method!r} conflicts with input retrieval_method={input_method!r}")
    if args.embedding_model and input_embedding and str(args.embedding_model) != input_embedding:
        raise ValueError(f"--embedding_model={args.embedding_model!r} conflicts with input embedding_model={input_embedding!r}")
    if args.retrieval_k and input_k and int(args.retrieval_k) != int(input_k):
        raise ValueError(f"--retrieval_k={args.retrieval_k!r} conflicts with input retrieval_k={input_k!r}")

    if input_method:
        args.retrieval_method = input_method
    if input_embedding:
        args.embedding_model = input_embedding
    if input_k:
        args.retrieval_k = int(input_k)
    if args.candidate_source != "full" and args.retrieval_k is None:
        raise ValueError("preselected Study A input must provide retrieval_k metadata or an explicit --retrieval_k")


def prepare_study_context(args):
    records, missing_skips = load_records(args.test_jsonl, return_skips=True)
    split_counts = count_records_by_split(records)
    document_id_audit = apply_document_id_strategy(records)
    resolve_retrieval_provenance_from_records(args, records)
    checkpoint = synthetic_no_reranking_checkpoint() if getattr(args, "no_reranking", False) else detect_checkpoint(args.model_path, args.model_type, args.loss_type)
    cfg = checkpoint["run_config"]
    args.eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else int(cfg.get("eval_batch_size", 4))
    args.eval_max_length, eval_length_source = resolve_eval_max_length(args.eval_max_length, cfg)
    candidate_source = build_candidate_source(args.candidate_source)
    input_identity = test_jsonl_identity(Path(args.test_jsonl))
    return records, missing_skips, split_counts, document_id_audit, checkpoint, candidate_source, input_identity, eval_length_source


def collect_eligible_candidates(args, records, missing_skips, split_counts, document_id_audit, candidate_source):
    totals = defaultdict(int)
    totals["total_records"] = len(records) + len(missing_skips)
    totals["skipped_missing_data"] = len(missing_skips)
    skip_rows = list(missing_skips)
    eligible = []
    candidate_counts = []
    for record in records:
        split = str(record.metadata.get("split", "test")).strip().lower() or "test"
        if not split_is_allowed(split, args.allowed_splits):
            totals["skipped_disallowed_split"] += 1
            skip_rows.append({"query_id": record.query_id, "reason": "disallowed_split", "split": record.metadata.get("split"), "allowed_splits": args.allowed_splits})
            continue
        candidate_set = candidate_source.select(record, args.retrieval_k)
        candidate_hits = candidate_set.hits
        validate_unique_document_ids(record.query_id, candidate_hits, "candidate pool")
        candidate_count = len(candidate_hits)
        candidate_counts.append(candidate_count)
        totals["candidate_count"] += candidate_count
        totals["positive_count"] += int(candidate_set.details.get("positive_count", 0))
        totals["negative_count"] += int(candidate_set.details.get("negative_count", 0))
        expected_candidate_count = candidate_set.details.get("expected_candidate_count")
        if expected_candidate_count is not None and candidate_count != int(expected_candidate_count):
            raise ValueError(
                f"candidate_source={args.candidate_source!r} requires exactly {int(expected_candidate_count)} candidates per query, "
                f"but query_id={record.query_id} has {candidate_count}"
            )
        if candidate_count < args.input_k:
            totals["skipped_pool_small"] += 1
            skip_rows.append({"query_id": record.query_id, "reason": "candidate_pool_smaller_than_input_k", "candidate_source": args.candidate_source, "candidate_count": candidate_count})
            logging.info("Skipping %s: candidate pool %s < input_k %s", record.query_id, candidate_count, args.input_k)
            continue
        eligible.append((record, candidate_set, candidate_count))
    if args.limit is not None:
        eligible = eligible[:args.limit]
    totals["candidate_eligible_records"] = len(eligible)
    totals["skipped_records"] = sum(totals[key] for key in ("skipped_missing_data", "skipped_pool_small", "skipped_disallowed_split"))
    total_queries = totals["total_records"] - totals["skipped_disallowed_split"]
    total_allowed_query_count = sum(split_counts.get(split, 0) for split in args.allowed_splits)
    eligible_queries = len(eligible)
    return eligible, {
        **dict(totals),
        "total_queries": total_queries,
        "allowed_splits": args.allowed_splits,
        "test_query_count": split_counts.get("test", 0) if "test" in args.allowed_splits else 0,
        "valid_query_count": split_counts.get("valid", 0) if "valid" in args.allowed_splits else 0,
        "total_allowed_query_count": total_allowed_query_count,
        "query_count_by_split": {split: split_counts.get(split, 0) for split in args.allowed_splits},
        "eligible_queries": eligible_queries,
        "eligible_query_count": eligible_queries,
        "excluded_queries": total_queries - eligible_queries,
        "queries_with_candidate_count_lt_input_k": totals["skipped_pool_small"],
        **candidate_count_stats(candidate_counts),
        "document_id_audit": document_id_audit,
        "skip_rows": skip_rows,
    }


def build_rerank_cache_row(record, candidate_set, candidate_count: int, ranked: List[dict], checkpoint: dict, args) -> dict:
    from rag_retrieval.infer.eval.build_run_qrels import convert_hits_to_graded_qrels

    full_qrels = {str(k): float(v) for k, v in convert_hits_to_graded_qrels([{"creator_id": h.creator_id, "content": h.content, "label": h.label} for h in record.hits]).items()}
    assert_qrels_creator_ids(record.query_id, full_qrels, record.hits, "full")
    top_k = ranked[:args.input_k]
    for idx, row in enumerate(top_k, start=1):
        row["rank"] = idx
    top_k_ids = [row["creator_id"] for row in top_k]
    prior_attention = {h.creator_id: h.prior_attention for h in record.hits}
    before_metrics = before_generation_metrics(record.query_id, full_qrels, top_k_ids, prior_attention, args.input_k, args.output_k)
    metadata = metadata_for(record)
    return {
        "query_id": record.query_id,
        "query": record.query,
        **metadata,
        "candidate_source": args.candidate_source,
        "retrieval_method": args.retrieval_method,
        "retrieval_k": args.retrieval_k,
        "embedding_model": args.embedding_model,
        "reranking_mode": "no_reranking" if getattr(args, "no_reranking", False) else "reranker",
        "candidate_count": candidate_count,
        "loss_type": "no_reranking" if getattr(args, "no_reranking", False) else checkpoint["loss_type"],
        "candidate_details": candidate_set.details,
        "final_candidate_creator_ids": candidate_set.details.get("candidate_creator_ids", []),
        "final_candidate_document_ids": candidate_set.details.get("candidate_document_ids", []),
        "input_k": args.input_k,
        "output_k": args.output_k,
        "top_k_creator_ids": top_k_ids,
        "top_k_document_ids": [row["document_id"] for row in top_k],
        "ranked_candidates": top_k,
        "full_qrels": full_qrels,
        "candidate_prior_attention": prior_attention,
        "before_metrics": before_metrics,
    }


def write_ranked_inputs_from_cache(output_dir: Path, rows: List[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ranked_inputs.jsonl"
    path.write_text("".join(json_dumps_strict(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def run_rerank_stage(args, output_dir: Path, records, missing_skips, split_counts, document_id_audit, checkpoint, candidate_source, input_identity, eval_length_source):
    signature = rerank_cache_signature(args, checkpoint, records, input_identity)
    if not args.overwrite:
        validate_rerank_cache_config(output_dir / "rerank_config.json", signature)
    config = vars(args).copy()
    config["detected_checkpoint"] = checkpoint
    config["rerank_cache_signature"] = signature
    config["resolved_eval_settings"] = {"eval_batch_size": args.eval_batch_size, "eval_max_length": args.eval_max_length, "eval_max_length_source": eval_length_source}
    write_json(output_dir / "rerank_config.json", config)
    eligible, eligibility_summary = collect_eligible_candidates(args, records, missing_skips, split_counts, document_id_audit, candidate_source)
    cached_rows = unique_by_qid(read_jsonl(output_dir / "ranked_inputs.jsonl"))
    cached_by_qid = {row["query_id"]: row for row in cached_rows if row.get("query_id")}
    target_qids = {record.query_id for record, _, _ in eligible}
    missing = [(record, candidate_set, candidate_count) for record, candidate_set, candidate_count in eligible if record.query_id not in cached_by_qid]
    if not missing:
        validate_cached_ranked_rows([cached_by_qid[qid] for qid in sorted(target_qids)], args.input_k)
        eligibility_summary["rerank_cache_complete"] = True
        eligibility_summary["reranked_query_count"] = len(target_qids)
        write_json(output_dir / "eligibility_summary.json", eligibility_summary)
        logging.info("Rerank cache already complete: %s queries in %s", len(target_qids), output_dir)
        return [cached_by_qid[qid] for qid in sorted(target_qids)]

    ranker = None if getattr(args, "no_reranking", False) else build_ranker(checkpoint, args)
    for record, candidate_set, candidate_count in missing:
        ranked = rank_candidates(ranker, record, candidate_set.hits, args)
        row = build_rerank_cache_row(record, candidate_set, candidate_count, ranked, checkpoint, args)
        append_jsonl(output_dir / "ranked_inputs.jsonl", row)
    rows = [row for row in compact_jsonl(output_dir / "ranked_inputs.jsonl") if row.get("query_id") in target_qids]
    validate_cached_ranked_rows(rows, args.input_k)
    eligibility_summary["rerank_cache_complete"] = len(rows) == len(target_qids)
    eligibility_summary["reranked_query_count"] = len(rows)
    write_json(output_dir / "eligibility_summary.json", eligibility_summary)
    logging.info("Rerank stage complete: %s cached queries in %s", len(rows), output_dir)
    return rows


def load_validated_rerank_cache(cache_dir: Path, args, checkpoint, records, input_identity) -> List[dict]:
    current = rerank_cache_signature(args, checkpoint, records, input_identity)
    validate_rerank_cache_config(cache_dir / "rerank_config.json", current)
    rows = unique_by_qid(read_jsonl(cache_dir / "ranked_inputs.jsonl"))
    if not rows:
        raise ValueError(f"Rerank cache is empty or missing: {cache_dir / 'ranked_inputs.jsonl'}")
    validate_cached_ranked_rows(rows, args.input_k)
    summary = read_json(cache_dir / "eligibility_summary.json")
    expected = int(summary.get("eligible_query_count", len(rows)))
    if len(rows) != expected:
        raise ValueError(f"Rerank cache has {len(rows)} rows but eligibility_summary expects {expected}")
    return rows


def run_generate_stage(args, output_dir: Path, cache_dir: Path, records, checkpoint, input_identity, eval_length_source):
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_rows = load_validated_rerank_cache(cache_dir, args, checkpoint, records, input_identity)
    signature = run_signature(args, checkpoint, records, input_identity)
    if not args.overwrite:
        validate_resume_config(output_dir / "config.json", signature)
    config = vars(args).copy()
    config["detected_checkpoint"] = checkpoint
    config["run_signature"] = signature
    config["rerank_cache_dir"] = str(cache_dir)
    config["resolved_eval_settings"] = {"eval_batch_size": args.eval_batch_size, "eval_max_length": args.eval_max_length, "eval_max_length_source": eval_length_source}
    write_json(output_dir / "config.json", config)
    write_ranked_inputs_from_cache(output_dir, cache_rows)

    completed = set() if args.overwrite else load_completed_valid(output_dir / "generations.jsonl")
    totals = defaultdict(int)
    for row in cache_rows:
        if row["query_id"] in completed:
            totals["resumed_completed"] += 1
            continue
        top_k = row["ranked_candidates"][:args.input_k]
        top_k_profiles = [{"creator_id": item["creator_id"], "document_id": item["document_id"], "content": item["content"]} for item in top_k]
        top_k_ids = [item["creator_id"] for item in top_k]
        top_k_document_ids = [item["document_id"] for item in top_k]
        prompt = build_prompt(row["query"], top_k_profiles, args.output_k, args.generator_prompt_path)
        raw = ""; parsed = None; errors = []; validation = {"recommendations": [], "valid_recommendations": []}; ok = False
        try:
            raw = make_echo_generation(top_k_profiles, args.output_k) if args.generator_backend == "echo" else call_ollama(args.ollama_base_url, args.generator_model, prompt, args.temperature, args.output_k, args.seed, top_k_ids, top_k_document_ids)
            parsed = parse_json_output(raw); ok, errors, validation = validate_generation(parsed, top_k_profiles, args.output_k, row["query_id"])
        except Exception as exc:
            errors = [str(exc)]
        valid_after_ids = [rec["creator_id"] for rec in validation.get("valid_recommendations", [])]
        position_preserving_after_ids = validation.get("position_preserving_creator_ids", [f"__INVALID_{row['query_id']}_{idx}" for idx in range(1, args.output_k + 1)])
        metrics = metrics_from_cached_before(row["query_id"], row["before_metrics"], row["full_qrels"], position_preserving_after_ids, valid_after_ids, row["candidate_prior_attention"], args.output_k, ok)
        metadata = {key: row.get(key) for key in PROJECT_FIELDS}
        counts_context = {"candidate_source": row.get("candidate_source", args.candidate_source), "candidate_count": row.get("candidate_count")}
        loss_type = row.get("loss_type", checkpoint["loss_type"])
        provenance_context = {
            "candidate_source": row.get("candidate_source", args.candidate_source),
            "retrieval_method": row.get("retrieval_method", args.retrieval_method),
            "retrieval_k": row.get("retrieval_k", args.retrieval_k),
            "embedding_model": row.get("embedding_model", args.embedding_model),
            "reranking_mode": row.get("reranking_mode", "reranker"),
        }
        metrics.update({**metadata, **counts_context, **provenance_context, "loss_type": loss_type, "parse_success": float(parsed is not None), "generation_success": float(bool(ok and len(valid_after_ids) == args.output_k)), "citation_id_validity_rate": validation.get("citation_id_validity_rate", 0.0), "valid_citation_rate": validation.get("valid_citation_rate", 0.0), "creator_document_match_rate": validation.get("creator_document_match_rate", 0.0), "hallucinated_creator_rate": validation.get("hallucinated_creator_rate", 0.0), "duplicate_creator_rate": validation.get("duplicate_creator_rate", 0.0), "output_completeness": validation.get("output_completeness", 0.0), "exact_output_validity": validation.get("exact_output_validity", 0.0)})
        generation = {"query_id": row["query_id"], "query": row["query"], **metadata, **provenance_context, "candidate_source": row.get("candidate_source", args.candidate_source), "candidate_count": row.get("candidate_count"), "loss_type": loss_type, "candidate_details": row.get("candidate_details", {}), "final_candidate_creator_ids": row.get("final_candidate_creator_ids", row.get("candidate_details", {}).get("candidate_creator_ids", [])), "final_candidate_document_ids": row.get("final_candidate_document_ids", row.get("candidate_details", {}).get("candidate_document_ids", [])), "input_k": args.input_k, "output_k": args.output_k, "top_k_creator_ids": top_k_ids, "top_k_document_ids": top_k_document_ids, "generated_recommendations": validation.get("recommendations", []), "valid_recommendations": validation.get("valid_recommendations", []), "position_preserving_creator_ids": position_preserving_after_ids, "raw_output": raw, "parsed_output": parsed, "prompt": prompt, "errors": errors, "generation_success": bool(metrics["generation_success"]), "metrics": metrics}
        append_jsonl(output_dir / "generations.jsonl", generation)
        if not generation["generation_success"]:
            append_jsonl(output_dir / "generation_failures.jsonl", generation)
        totals["attempted_generations"] += 1
        totals["successful_generations"] += int(generation["generation_success"])
        totals["failed_generations"] += int(not generation["generation_success"])
    generations = compact_jsonl(output_dir / "generations.jsonl")
    rebuild_failure_file(output_dir, generations)
    eligibility_summary = read_json(cache_dir / "eligibility_summary.json")
    eligibility_summary.update({
        "rerank_cache_dir": str(cache_dir),
        "attempted_generations": totals["attempted_generations"],
        "resumed_completed": totals["resumed_completed"],
        "successful_generation_count": sum(bool(row.get("generation_success")) for row in generations),
        "failed_generation_count": sum(not bool(row.get("generation_success")) for row in generations),
        "generated_exposure_query_count": sum(row.get("metrics", {}).get(f"Exp_after@{args.output_k}") is not None for row in generations),
        "generated_ndcg_query_count": sum(f"graded_nDCG_after@{args.output_k}" in row.get("metrics", {}) for row in generations),
    })
    write_json(output_dir / "eligibility_summary.json", eligibility_summary)
    rebuild_reports(output_dir, checkpoint, args.input_k, args.output_k, args.raw_data_dir)
    logging.info("Generate stage complete: %s", dict(totals))


def rebuild_failure_file(output_dir: Path, generations: List[dict]) -> None:
    failures = [row for row in generations if not row.get("generation_success")]
    path = output_dir / "generation_failures.jsonl"
    if failures:
        path.write_text("".join(json_dumps_strict(row, ensure_ascii=False) + "\n" for row in failures), encoding="utf-8")
    elif path.exists():
        path.unlink()


def assert_generation_complete_for_master(output_dir: Path, summary: dict) -> None:
    eligibility = read_json(output_dir / "eligibility_summary.json")
    if "eligible_query_count" not in eligibility:
        raise RuntimeError(f"Study A generation completeness check requires eligible_query_count in {output_dir / 'eligibility_summary.json'}")
    eligible_query_count = int(eligibility["eligible_query_count"])
    successful_generation_count = int(summary.get("successful_generation_count", 0))
    failed_generation_count = int(summary.get("failed_generation_count", 0))
    if successful_generation_count != eligible_query_count or failed_generation_count != 0:
        raise RuntimeError(
            "Study A generation is incomplete; refusing to upsert master CSV. "
            f"eligible_query_count={eligible_query_count}, "
            f"successful_generation_count={successful_generation_count}, "
            f"failed_generation_count={failed_generation_count}"
        )


def rebuild_reports(output_dir: Path, checkpoint: dict, input_k: int, output_k: int, raw_data_dir: Optional[str] = None) -> None:
    generations = unique_by_qid(read_jsonl(output_dir / "generations.jsonl"))
    rebuild_failure_file(output_dir, generations)
    if raw_data_dir:
        ranked_rows_for_skill = unique_by_qid(read_jsonl(output_dir / "ranked_inputs.jsonl"))
        skill_by_qid = per_query_skill_metrics(
            ranked_rows=ranked_rows_for_skill,
            generation_rows=generations,
            source=raw_data_dir,
            cache_dir=output_dir,
            input_k=input_k,
            output_k=output_k,
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
    numeric_keys = sorted({
        key for row in metric_rows for key, value in row.items()
        if key not in SUMMARY_AVERAGE_EXCLUDED_KEYS
        and value is not None
        and isinstance(value, (int, float))
        and not (isinstance(value, float) and math.isnan(value))
    })
    summary = {
        "loss_type": "no_reranking" if read_json(output_dir / "config.json").get("no_reranking") else checkpoint["loss_type"],
        "num_queries": len(metric_rows),
        "successful_generation_count": sum(bool(row.get("generation_success")) for row in generations),
        "failed_generation_count": sum(not bool(row.get("generation_success")) for row in generations),
        "generated_exposure_query_count": sum(row.get(f"Exp_after@{output_k}") is not None for row in metric_rows),
        "generated_ndcg_query_count": sum(f"graded_nDCG_after@{output_k}" in row for row in metric_rows),
        **skill_metric_summary(metric_rows),
        **{key: mean([float(row[key]) for row in metric_rows if row.get(key) is not None and isinstance(row.get(key), (int, float)) and not (isinstance(row[key], float) and math.isnan(row[key]))]) for key in numeric_keys},
    }
    summary.update(summary_provenance(output_dir, checkpoint))
    successful = [row for row in generations if row.get("generation_success")]
    position_rows = []
    for pos in range(1, input_k + 1):
        selected = []
        for row in successful:
            ids = row.get("top_k_creator_ids", [])
            generated = [rec.get("creator_id") for rec in row.get("valid_recommendations", [])]
            if pos <= len(ids) and ids[pos - 1] in generated:
                selected.extend(i + 1 for i, cid in enumerate(generated) if cid == ids[pos - 1])
        position_rows.append({"input_position": pos, "eligible_successful_query_count": len(successful), "selected_count": len(selected), "selection_rate": len(selected) / len(successful) if successful else 0.0, "mean_generated_position_when_selected": mean(selected) if selected else None})
    write_csv(output_dir / "position_selection.csv", position_rows)
    posthoc_succeeded = False
    if raw_data_dir:
        try:
            from .selection_analysis import run_analysis
            posthoc_result = run_analysis(output_dir, raw_data_dir)
            if isinstance(posthoc_result, dict) and isinstance(posthoc_result.get("posthoc_summary"), dict):
                summary.update(posthoc_result["posthoc_summary"])
            posthoc_succeeded = True
        except Exception as exc:
            raise RuntimeError(f"Study A post-hoc selection analysis failed for output_dir={output_dir}: {exc}") from exc
    else:
        logging.warning("Skipping Study A post-hoc selection analysis because --raw_data_dir was not provided")
    write_csv(output_dir / "summary_metrics.csv", [summary])
    saved_config = read_json(output_dir / "config.json")
    is_limited_run = saved_config.get("limit") is not None
    assert_generation_complete_for_master(output_dir, summary)
    if is_limited_run:
        logging.info("Skipping master Study A results upsert for limited run output_dir=%s", output_dir)
    elif posthoc_succeeded:
        upsert_master_results_row(study_master_results_path(output_dir), summary)
    else:
        logging.warning("Skipping master Study A results upsert because post-hoc analysis did not complete successfully")


def main():
    args = parse_args(); validate_cli_values(args)
    args.allowed_splits = normalize_allowed_splits(args.allowed_splits)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.rerank_cache_dir) if args.rerank_cache_dir else output_dir
    setup_logging(output_dir)
    if args.stage in {"rerank", "all"}:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        if args.stage == "rerank":
            clear_output_files(output_dir, RERANK_CACHE_FILES)
        elif args.stage == "generate":
            clear_output_files(output_dir, RUN_FILES)
        else:
            clear_output_files(output_dir, sorted(set(RUN_FILES) | set(RERANK_CACHE_FILES)))

    records, missing_skips, split_counts, document_id_audit, checkpoint, candidate_source, input_identity, eval_length_source = prepare_study_context(args)
    logging.info("Detected model_type=%s loss_type=%s eval_batch_size=%s eval_max_length=%s source=%s", checkpoint["model_type"], checkpoint["loss_type"], args.eval_batch_size, args.eval_max_length, eval_length_source)

    if args.stage in {"rerank", "all"}:
        run_rerank_stage(args, cache_dir, records, missing_skips, split_counts, document_id_audit, checkpoint, candidate_source, input_identity, eval_length_source)
    if args.stage in {"generate", "all"}:
        run_generate_stage(args, output_dir, cache_dir, records, checkpoint, input_identity, eval_length_source)
    logging.info("Study A %s stage complete", args.stage)


if __name__ == "__main__": main()
