import hashlib
import logging
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SAMPLING_ALGORITHM_VERSION = "study_a_v2"
IDENTITY_FIELDS = (
    "project_name", "view", "source_project_file", "source_project_row",
    "project_variant_index", "trial_id", "fold_id", "split",
)


class MissingRequiredData(ValueError):
    pass


@dataclass
class Hit:
    creator_id: str
    content: str
    label: float
    prior_attention: float
    document_id: str = ""
    document_id_source: str = "input"

    def __post_init__(self) -> None:
        if not self.document_id:
            self.document_id = stable_document_id(self.creator_id)
            self.document_id_source = "fallback"


@dataclass
class QueryRecord:
    idx: int
    query_id: str
    query: str
    hits: List[Hit]
    metadata: dict


def stable_document_id(creator_id: str) -> str:
    return f"DOC_{str(creator_id).strip()}"


def stable_content_document_id(creator_id: str, content: str) -> str:
    payload = json.dumps({"creator_id": str(creator_id), "content": str(content)}, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"DOC_{str(creator_id).strip()}_{digest}"


def extract_project_name_from_query(query: str) -> Optional[str]:
    if not query:
        return None
    match = re.search(r"Creators\s+suitable\s+for\s+the\s+(.+?)\s+project", query, re.IGNORECASE)
    return match.group(1).strip().strip(".") if match else None


def stable_query_id(rec: dict, idx: int) -> str:
    metadata = {key: rec.get(key) for key in IDENTITY_FIELDS if rec.get(key) is not None}
    if "project_name" not in metadata:
        project = rec.get("project") or extract_project_name_from_query(str(rec.get("query", "")))
        if project:
            metadata["project_name"] = project
    missing = [key for key in IDENTITY_FIELDS if key not in metadata]
    if missing:
        metadata["query"] = rec.get("query", "")
    if missing:
        logging.warning("Stable query ID metadata incomplete; using query text for query_id derivation (input row %s)", idx)
    if missing and not metadata.get("query"):
        metadata["line_index_fallback"] = idx
        logging.warning("Stable query ID used final line-index fallback for input row %s", idx)
    canonical = json.dumps(metadata, sort_keys=True, ensure_ascii=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"q_{digest}"


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records(path: str, return_skips: bool = False):
    records = []
    skips = []
    seen_query_ids = {}
    for idx, rec in enumerate(read_jsonl(Path(path)), start=1):
        query_id = str(rec.get("query_id") or stable_query_id(rec, idx))
        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate stable query_id {query_id!r} for input rows {seen_query_ids[query_id]} and {idx}")
        seen_query_ids[query_id] = idx
        try:
            hits = []
            creator_ids = set()
            for hit in rec.get("hits", []):
                creator_id = str(hit.get("creator_id", "")).strip()
                content = str(hit.get("content", "")).strip()
                if not creator_id or not content:
                    raise MissingRequiredData(f"empty creator_id/content at input row {idx}")
                if creator_id in creator_ids:
                    raise MissingRequiredData(f"duplicate creator_id {creator_id} at input row {idx}")
                creator_ids.add(creator_id)
                label = float(hit["label"])
                if not math.isfinite(label) or label < 0:
                    raise MissingRequiredData(f"label must be finite and nonnegative for creator {creator_id} at input row {idx}")
                if hit.get("prior_attention") is None or "prior_attention" not in hit:
                    raise MissingRequiredData(f"missing/null prior_attention for creator {creator_id} at input row {idx}")
                prior_attention = float(hit["prior_attention"])
                if not math.isfinite(prior_attention) or prior_attention < 0:
                    raise MissingRequiredData(f"prior_attention must be finite and nonnegative for creator {creator_id} at input row {idx}")
                document_id = str(hit.get("document_id") or hit.get("doc_id") or "").strip()
                hits.append(Hit(creator_id=creator_id, content=content, label=label, prior_attention=prior_attention, document_id=document_id, document_id_source="input" if document_id else "fallback"))
            if not any(hit.label > 0 for hit in hits):
                skips.append({"input_row": idx, "query_id": query_id, "reason": "no_positive_candidates", "metadata": {k: rec.get(k) for k in IDENTITY_FIELDS}})
                continue
            records.append(QueryRecord(
                idx=idx, query_id=query_id, query=str(rec["query"]),
                hits=hits, metadata={k: v for k, v in rec.items() if k not in {"query", "hits"}},
            ))
        except (KeyError, TypeError, ValueError, MissingRequiredData) as exc:
            skips.append({"input_row": idx, "query_id": query_id, "reason": "missing_required_data", "error": str(exc), "metadata": {k: rec.get(k) for k in IDENTITY_FIELDS}})
    return (records, skips) if return_skips else records


def _query_seed(global_seed: int, query_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{query_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def sample_candidate_ids(record: QueryRecord, negative_sample_percent: float, seed: int) -> Tuple[List[str], dict]:
    if not 0.0 <= negative_sample_percent <= 1.0:
        raise ValueError("negative_sample_percent must be between 0 and 1")
    positives = [h.creator_id for h in record.hits if h.label > 0]
    negatives = sorted((h.creator_id for h in record.hits if h.label == 0), key=str)
    if len(set(positives)) != len(positives) or len(set(negatives)) != len(negatives) or set(positives) & set(negatives):
        raise ValueError(f"duplicate or conflicting creator IDs in {record.query_id}")
    sampled_negative_count = min(math.ceil(negative_sample_percent * len(negatives)), max(0, len(positives) - 1))
    rng = random.Random(_query_seed(seed, record.query_id))
    sampled_negatives = sorted(rng.sample(negatives, sampled_negative_count), key=str) if sampled_negative_count else []
    selected = set(positives) | set(sampled_negatives)
    candidate_ids = [h.creator_id for h in record.hits if h.creator_id in selected]
    details = {"query_id": record.query_id, "positive_count": len(positives), "original_negative_count": len(negatives), "sampled_negative_count": sampled_negative_count, "final_candidate_count": len(candidate_ids), "positive_creator_ids": positives, "sampled_negative_creator_ids": sampled_negatives, "candidate_creator_ids": candidate_ids, "seed": _query_seed(seed, record.query_id)}
    return candidate_ids, details


def test_data_identity(records: List[QueryRecord]) -> str:
    payload = []
    for record in sorted(records, key=lambda item: item.query_id):
        payload.append({
            "query_id": record.query_id, "query": record.query,
            "metadata": record.metadata,
            "hits": [{"creator_id": hit.creator_id, "document_id": hit.document_id, "content": hit.content, "label": hit.label, "prior_attention": hit.prior_attention} for hit in record.hits],
        })
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode()).hexdigest()


def _identity_checksum(records: List[QueryRecord]) -> str:
    return test_data_identity(records)


def test_jsonl_identity(path: Path) -> str:
    rows = list(read_jsonl(path))
    normalized = []
    for idx, row in enumerate(rows, start=1):
        normalized.append({"query_id": str(row.get("query_id") or stable_query_id(row, idx)), "record": row})
    normalized.sort(key=lambda item: item["query_id"])
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode()).hexdigest()


def load_or_create_samples(records: List[QueryRecord], path: Path, negative_sample_percent: float, seed: int, test_path: Optional[Path] = None) -> Dict[str, dict]:
    if not 0.0 <= negative_sample_percent <= 1.0:
        raise ValueError("negative_sample_percent must be between 0 and 1")
    expected = {"negative_sample_percent": float(negative_sample_percent), "global_seed": int(seed), "sampling_algorithm_version": SAMPLING_ALGORITHM_VERSION, "test_jsonl_identity": test_jsonl_identity(test_path) if test_path is not None else _identity_checksum(records)}
    if path.exists():
        rows = list(read_jsonl(path))
        metadata = next((row for row in rows if row.get("record_type") == "metadata"), None)
        if metadata is None:
            raise ValueError(f"sample file {path} has no Study A metadata header; regenerate it")
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"sample file {path} incompatible for {key}: saved={metadata.get(key)!r}, current={value!r}")
        samples = {row["query_id"]: row for row in rows if row.get("record_type") != "metadata"}
        current = {record.query_id: record for record in records}
        missing_query_ids = sorted(set(current) - set(samples))
        if missing_query_ids:
            raise ValueError(f"sample file is missing current query_ids: {missing_query_ids[:5]}")
        for qid, sample in samples.items():
            if qid not in current:
                raise ValueError(f"sample file contains unknown query_id {qid}")
            record = current[qid]
            by_id = hits_by_id(record)
            positives = {h.creator_id for h in record.hits if h.label > 0}
            original_negatives = {h.creator_id for h in record.hits if h.label == 0}
            positive_ids = sample.get("positive_creator_ids", [])
            sampled_negative_ids = sample.get("sampled_negative_creator_ids", [])
            candidate_creator_ids = sample.get("candidate_creator_ids", [])
            positive_set = set(positive_ids)
            sampled_negatives = set(sampled_negative_ids)
            candidate_ids = set(candidate_creator_ids)
            if positive_set != positives:
                raise ValueError(f"positive_creator_ids do not match current positive creators for query_id {qid}: saved={sorted(positive_set)}, current={sorted(positives)}")
            if len(positive_set) != len(positive_ids):
                raise ValueError(f"duplicate positive_creator_ids for query_id {qid}")
            if len(sampled_negatives) != len(sampled_negative_ids):
                raise ValueError(f"duplicate sampled_negative_creator_ids for query_id {qid}")
            if len(candidate_ids) != len(candidate_creator_ids):
                raise ValueError(f"duplicate candidate_creator_ids for query_id {qid}")
            if candidate_ids != positive_set | sampled_negatives:
                raise ValueError(f"candidate_creator_ids must equal positives union sampled negatives for query_id {qid}")
            if any(cid not in by_id or by_id[cid].label != 0 for cid in sampled_negatives):
                raise ValueError(f"sampled negatives must exist in the current query with label 0 for query_id {qid}")
            counts = (
                ("positive_count", len(positive_ids)),
                ("original_negative_count", len(original_negatives)),
                ("sampled_negative_count", len(sampled_negative_ids)),
                ("final_candidate_count", len(candidate_creator_ids)),
            )
            for key, expected_count in counts:
                if int(sample.get(key, -1)) != expected_count:
                    raise ValueError(f"saved {key} does not match current data for query_id {qid}")
            if len(sampled_negatives) >= len(positive_set):
                raise ValueError(f"sampled negatives are not fewer than positives for query_id {qid}")
        return samples
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = {}
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"record_type": "metadata", **expected}, ensure_ascii=False, allow_nan=False) + "\n")
        for record in records:
            _, details = sample_candidate_ids(record, negative_sample_percent, seed)
            samples[record.query_id] = details
            f.write(json.dumps(details, ensure_ascii=False, allow_nan=False) + "\n")
    return samples


def hits_by_id(record: QueryRecord) -> Dict[str, Hit]:
    return {h.creator_id: h for h in record.hits}


def graded_qrels_from_hits(hits: List[Hit]) -> Dict[str, float]:
    """Compatibility wrapper delegating to the repository's canonical conversion."""
    from rag_retrieval.infer.eval.build_run_qrels import convert_hits_to_graded_qrels
    return {str(k): float(v) for k, v in convert_hits_to_graded_qrels([{"creator_id": h.creator_id, "content": h.content, "label": h.label} for h in hits]).items()}
