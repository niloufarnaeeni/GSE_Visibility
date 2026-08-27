import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "kaito" / "large_data_creator_profile"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_DIR / "retriever_results"
K_SELECTION_VALUES = (20, 30, 50, 100, 150, 200, 300)
GROUND_TRUTH_K = 50
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._'-][A-Za-z0-9]+)*", re.UNICODE)
IDENTITY_FIELDS = (
    "project_name", "view", "source_project_file", "source_project_row",
    "project_variant_index", "trial_id", "fold_id", "split",
)
HARRIER_MODEL = "microsoft/harrier-oss-v1-0.6b"
BGE_M3_MODEL = "BAAI/bge-m3"
HARRIER_MAX_SEQUENCE_LENGTH = 32768
BGE_M3_MAX_SEQUENCE_LENGTH = 8192


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text))]


def stable_document_id(creator_id: str) -> str:
    return f"DOC_{str(creator_id).strip()}"


def extract_project_name_from_query(query: str) -> Optional[str]:
    if not query:
        return None
    match = re.search(r"Creators\s+suitable\s+for\s+the\s+(.+?)\s+project", query, re.IGNORECASE)
    return match.group(1).strip().strip(".") if match else None


def stable_query_id(record: dict, idx: int) -> str:
    metadata = {key: record.get(key) for key in IDENTITY_FIELDS if record.get(key) is not None}
    if "project_name" not in metadata:
        project = record.get("project") or extract_project_name_from_query(str(record.get("query", "")))
        if project:
            metadata["project_name"] = project
    missing = [key for key in IDENTITY_FIELDS if key not in metadata]
    if missing:
        metadata["query"] = record.get("query", "")
    if missing and not metadata.get("query"):
        metadata["line_index_fallback"] = idx
    canonical = json.dumps(metadata, sort_keys=True, ensure_ascii=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"q_{digest}"


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def hit_creator_id(hit: dict) -> str:
    return str(hit.get("creator_id", "")).strip()


def hit_document_id(hit: dict) -> str:
    return str(hit.get("document_id") or hit.get("doc_id") or stable_document_id(hit_creator_id(hit))).strip()


def hit_content(hit: dict) -> str:
    return str(hit.get("content", "")).strip()


def hit_label(hit: dict) -> float:
    return float(hit.get("label", 0.0))


def hit_prior_attention(hit: dict) -> Optional[float]:
    value = hit.get("prior_attention")
    return None if value is None else float(value)


def valid_hits(record: dict) -> List[dict]:
    output = []
    seen = set()
    for hit in record.get("hits", []):
        creator_id = hit_creator_id(hit)
        content = hit_content(hit)
        if not creator_id or not content or creator_id in seen:
            continue
        seen.add(creator_id)
        output.append(hit)
    return output


def strict_valid_hits(record: dict, query_id: str) -> List[dict]:
    hits = list(record.get("hits", []))
    valid = valid_hits(record)
    if len(valid) != len(hits):
        raise ValueError(f"query_id={query_id} contains invalid or duplicate hits; unique non-empty creator/content pairs are required")
    creator_ids = [hit_creator_id(hit) for hit in valid]
    if len(creator_ids) != len(set(creator_ids)):
        raise ValueError(f"query_id={query_id} does not have unique creator IDs")
    return valid


def query_id_for_record(record: dict, idx: int) -> str:
    return str(record.get("query_id") or stable_query_id(record, idx))


def infer_dataset_split(records: List[dict], input_jsonl: Path) -> str:
    splits = sorted({str(record.get("split", "")).strip().lower() for record in records if str(record.get("split", "")).strip()})
    if len(splits) == 1:
        return splits[0]
    if len(splits) > 1:
        return "mixed"
    stem = input_jsonl.stem.strip().lower()
    return stem or "unknown"


def dataset_provenance(records: List[dict], input_jsonl: Path) -> dict:
    return {
        "input_jsonl": str(input_jsonl),
        "dataset_filename": input_jsonl.name,
        "dataset_split": infer_dataset_split(records, input_jsonl),
    }


def model_slug(embedding_model: Optional[str]) -> str:
    model = str(embedding_model or "").strip()
    if model == HARRIER_MODEL or "harrier" in model.lower():
        return "harrier"
    if model == BGE_M3_MODEL or "bge-m3" in model.lower():
        return "bge_m3"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model.lower()).strip("_")
    return slug or "dense"


def output_dir_for_method(output_root: Path, method: str, embedding_model: Optional[str] = None) -> Path:
    if method == "dense":
        return output_root / model_slug(embedding_model)
    return output_root / method


def preselected_output_dir(output_root: Path, relevant_k: int, irrelevant_k: int) -> Path:
    return output_root / f"preselected_rel{int(relevant_k)}_irrel{int(irrelevant_k)}"


def selected_device(device: str) -> str:
    requested = str(device or "auto").strip()
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def retrieval_provenance(
    base: dict,
    method: str,
    retrieval_k: Optional[int],
    embedding_model: Optional[str] = None,
    embedding_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    effective_max_sequence_length: Optional[int] = None,
) -> dict:
    return {
        **base,
        "retrieval_method": method,
        "embedding_model": embedding_model,
        "retrieval_k": retrieval_k,
        "embedding_batch_size": embedding_batch_size,
        "device": device,
        "effective_max_sequence_length": effective_max_sequence_length,
    }


def bm25_scores(query: str, hits: List[dict], k1: float = 1.5, b: float = 0.75) -> List[float]:
    corpus_tokens = [tokenize(hit_content(hit)) for hit in hits]
    query_terms = tokenize(query)
    if not hits or not query_terms:
        return [0.0 for _ in hits]

    doc_freq = Counter()
    for tokens in corpus_tokens:
        doc_freq.update(set(tokens))
    doc_count = len(corpus_tokens)
    avg_doc_len = sum(len(tokens) for tokens in corpus_tokens) / doc_count if doc_count else 0.0

    scores = []
    for tokens in corpus_tokens:
        freqs = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in query_terms:
            tf = freqs.get(term, 0)
            if not tf:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len if avg_doc_len else 0.0))
            score += idf * ((tf * (k1 + 1.0)) / denom)
        scores.append(score)
    return scores


def bm25_rank_hits(record: dict) -> List[dict]:
    hits = valid_hits(record)
    scores = bm25_scores(str(record.get("query", "")), hits)
    ranked = [
        {"hit": hit, "score": score, "original_index": idx}
        for idx, (hit, score) in enumerate(zip(hits, scores))
    ]
    ranked.sort(key=lambda item: (-item["score"], hit_creator_id(item["hit"]), item["original_index"]))
    return ranked


def l2_normalize(matrix):
    import numpy as np

    values = np.asarray(matrix, dtype="float32")
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return values / norms


@dataclass
class DenseEncoder:
    model_name: str
    batch_size: int
    device: str
    effective_max_sequence_length: int
    backend: object
    backend_type: str

    def encode_queries(self, queries: List[str]):
        if self.backend_type == "harrier_sentence_transformers":
            return self.backend.encode(
                queries,
                batch_size=self.batch_size,
                device=self.device,
                normalize_embeddings=True,
                prompt_name="web_search_query",
                show_progress_bar=False,
            )
        if self.backend_type == "bge_m3_flag_embedding":
            output = self.backend.encode(
                queries,
                batch_size=self.batch_size,
                max_length=self.effective_max_sequence_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            return output["dense_vecs"] if isinstance(output, dict) else output
        return self.backend.encode(queries, batch_size=self.batch_size)

    def encode_documents(self, documents: List[str]):
        if self.backend_type == "harrier_sentence_transformers":
            return self.backend.encode(
                documents,
                batch_size=self.batch_size,
                device=self.device,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        if self.backend_type == "bge_m3_flag_embedding":
            output = self.backend.encode(
                documents,
                batch_size=self.batch_size,
                max_length=self.effective_max_sequence_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            return output["dense_vecs"] if isinstance(output, dict) else output
        return self.backend.encode(documents, batch_size=self.batch_size)


def load_dense_encoder(embedding_model: str, batch_size: int, device: str) -> DenseEncoder:
    model_name = str(embedding_model or "").strip()
    if not model_name:
        raise ValueError("--embedding_model is required when --method dense")
    resolved_device = selected_device(device)
    if model_name == BGE_M3_MODEL:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception as exc:
            raise RuntimeError("Dense retrieval with BAAI/bge-m3 requires FlagEmbedding") from exc
        use_fp16 = resolved_device.startswith("cuda")
        try:
            backend = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=[resolved_device])
        except TypeError:
            backend = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        return DenseEncoder(model_name, batch_size, resolved_device, BGE_M3_MAX_SEQUENCE_LENGTH, backend, "bge_m3_flag_embedding")

    if model_name == HARRIER_MODEL:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError("Dense retrieval with microsoft/harrier-oss-v1-0.6b requires sentence-transformers") from exc
        backend = SentenceTransformer(model_name, device=resolved_device)
        effective = int(getattr(backend, "max_seq_length", HARRIER_MAX_SEQUENCE_LENGTH) or HARRIER_MAX_SEQUENCE_LENGTH)
        effective = min(effective, HARRIER_MAX_SEQUENCE_LENGTH)
        try:
            backend.max_seq_length = effective
        except Exception:
            pass
        return DenseEncoder(model_name, batch_size, resolved_device, effective, backend, "harrier_sentence_transformers")

    raise ValueError(f"Unsupported dense embedding model {model_name!r}; expected {HARRIER_MODEL!r} or {BGE_M3_MODEL!r}")


def dense_rank_hits(record: dict, encoder: DenseEncoder) -> List[dict]:
    hits = valid_hits(record)
    if not hits:
        return []
    query_embeddings = l2_normalize(encoder.encode_queries([str(record.get("query", ""))]))
    document_embeddings = l2_normalize(encoder.encode_documents([hit_content(hit) for hit in hits]))
    scores = (document_embeddings @ query_embeddings[0]).tolist()
    ranked = [
        {"hit": hit, "score": float(score), "original_index": idx}
        for idx, (hit, score) in enumerate(zip(hits, scores))
    ]
    ranked.sort(key=lambda item: (-item["score"], hit_creator_id(item["hit"]), item["original_index"]))
    return ranked


def ground_truth_rank_hits(record: dict) -> List[dict]:
    ranked = [{"hit": hit, "score": None, "original_index": idx} for idx, hit in enumerate(valid_hits(record))]
    ranked.sort(key=lambda item: (-hit_label(item["hit"]), hit_creator_id(item["hit"]), item["original_index"]))
    return ranked


def select_relevant_irrelevant_hits(record: dict, query_id: str, relevant_k: int, irrelevant_k: int) -> List[dict]:
    hits = strict_valid_hits(record, query_id)
    indexed_hits = [{"hit": hit, "original_index": idx} for idx, hit in enumerate(hits)]
    relevant = [item for item in indexed_hits if hit_label(item["hit"]) > 0.0]
    irrelevant = [item for item in indexed_hits if hit_label(item["hit"]) == 0.0]
    relevant.sort(key=lambda item: (-hit_label(item["hit"]), hit_creator_id(item["hit"]), item["original_index"]))
    irrelevant.sort(key=lambda item: (hit_creator_id(item["hit"]), item["original_index"]))
    if len(relevant) < relevant_k:
        raise ValueError(f"query_id={query_id} has only {len(relevant)} relevant candidates; requires relevant_k={relevant_k}")
    if len(irrelevant) < irrelevant_k:
        raise ValueError(f"query_id={query_id} has only {len(irrelevant)} irrelevant candidates; requires irrelevant_k={irrelevant_k}")
    selected = relevant[:relevant_k] + irrelevant[:irrelevant_k]
    selected.sort(key=lambda item: item["original_index"])
    return [item["hit"] for item in selected]


def metric_row(query_id: str, ranked_hits: List[dict], k: int) -> dict:
    selected = ranked_hits[:k]
    selected_ids = [hit_creator_id(item["hit"]) for item in selected]
    qrels = {hit_creator_id(item["hit"]): hit_label(item["hit"]) for item in ranked_hits if hit_label(item["hit"]) > 0.0}
    relevant_total = len(qrels)
    relevant_retrieved = sum(1 for creator_id in selected_ids if creator_id in qrels)
    ndcg = graded_ndcg_at_k(qrels, selected_ids, k) if qrels else 0.0
    return {
        "query_id": query_id,
        "K": k,
        "Recall@K": relevant_retrieved / relevant_total if relevant_total else 0.0,
        "graded_nDCG@K": ndcg,
        "HitRate@K": 1.0 if relevant_retrieved > 0 else 0.0,
        "relevant_retrieved_count": relevant_retrieved,
        "relevant_total_count": relevant_total,
    }


def aggregate_metric_rows(rows: List[dict], k: int, method: str, split: str, provenance: Optional[dict] = None) -> dict:
    return {
        **(provenance or {}),
        "split": split,
        "method": method,
        "K": k,
        "Recall@K": sum(float(row["Recall@K"]) for row in rows) / len(rows) if rows else 0.0,
        "graded_nDCG@K": sum(float(row["graded_nDCG@K"]) for row in rows) / len(rows) if rows else 0.0,
        "HitRate@K": sum(float(row["HitRate@K"]) for row in rows) / len(rows) if rows else 0.0,
        "relevant_retrieved_count": sum(int(row["relevant_retrieved_count"]) for row in rows),
        "relevant_total_count": sum(int(row["relevant_total_count"]) for row in rows),
        "num_queries": len(rows),
    }


def graded_ndcg_at_k(qrels: dict, ordered_creator_ids: List[str], k: int) -> float:
    gains = [float(qrels.get(creator_id, 0.0)) for creator_id in ordered_creator_ids[:k]]
    dcg = sum(gain / math.log2(idx + 2) for idx, gain in enumerate(gains))
    ideal_gains = sorted((float(value) for value in qrels.values()), reverse=True)[:k]
    idcg = sum(gain / math.log2(idx + 2) for idx, gain in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def candidate_stat_rows(records: List[dict], k: int, method: str, split: str, provenance: Optional[dict] = None) -> List[dict]:
    rows = []
    for idx, record in enumerate(records, start=1):
        count = len(valid_hits(record))
        rows.append({
            **(provenance or {}),
            "split": split,
            "method": method,
            "query_id": query_id_for_record(record, idx),
            "K": k,
            "candidate_count": count,
            "selected_candidate_count": min(k, count),
            "fewer_than_k": count < k,
        })
    return rows


def ranked_hits_for_record(record: dict, method: str, encoder: Optional[DenseEncoder] = None) -> List[dict]:
    if method == "bm25":
        return bm25_rank_hits(record)
    if method == "dense":
        if encoder is None:
            raise ValueError("dense ranking requires an encoder")
        return dense_rank_hits(record, encoder)
    if method == "ground_truth":
        return ground_truth_rank_hits(record)
    raise ValueError(f"Unsupported retrieval method {method!r}")


def write_k_selection(
    input_jsonl: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    method: str = "bm25",
    k_values: Iterable[int] = K_SELECTION_VALUES,
    embedding_model: Optional[str] = None,
    embedding_batch_size: Optional[int] = None,
    device: str = "auto",
    encoder: Optional[DenseEncoder] = None,
) -> List[dict]:
    records = read_jsonl(input_jsonl)
    active_encoder = encoder
    if method == "dense" and active_encoder is None:
        active_encoder = load_dense_encoder(str(embedding_model or ""), int(embedding_batch_size or 32), device)
    provenance = retrieval_provenance(
        dataset_provenance(records, input_jsonl),
        method,
        None,
        embedding_model=getattr(active_encoder, "model_name", embedding_model) if method == "dense" else None,
        embedding_batch_size=getattr(active_encoder, "batch_size", embedding_batch_size) if method == "dense" else None,
        device=getattr(active_encoder, "device", selected_device(device)) if method == "dense" else None,
        effective_max_sequence_length=getattr(active_encoder, "effective_max_sequence_length", None) if method == "dense" else None,
    )
    split = provenance["dataset_split"]
    rows = []
    for k in k_values:
        per_query = []
        for idx, record in enumerate(records, start=1):
            per_query.append(metric_row(query_id_for_record(record, idx), ranked_hits_for_record(record, method, active_encoder), int(k)))
        rows.append(aggregate_metric_rows(per_query, int(k), "bm25", split, provenance))
        rows[-1]["method"] = method
    output_path = output_dir_for_method(output_root, method, embedding_model) / "k_selection_metrics.csv"
    write_csv(output_path, rows)
    return rows


def write_validation_k_selection(input_jsonl: Path, output_root: Path = DEFAULT_OUTPUT_ROOT, k_values: Iterable[int] = K_SELECTION_VALUES) -> List[dict]:
    return write_k_selection(input_jsonl, output_root, "bm25", k_values)


def selected_output_record(record: dict, selected_hits: List[dict], provenance: Optional[dict] = None) -> dict:
    output = dict(record)
    output["hits"] = [dict(hit) for hit in selected_hits]
    if provenance:
        output["retrieval_method"] = provenance.get("retrieval_method")
        output["retrieval_k"] = provenance.get("retrieval_k")
        output["embedding_model"] = provenance.get("embedding_model")
    return output


def preselected_candidate_rows(records: List[dict], relevant_k: int, irrelevant_k: int, provenance: Optional[dict] = None) -> List[dict]:
    output = []
    total_k = int(relevant_k) + int(irrelevant_k)
    for idx, record in enumerate(records, start=1):
        query_id = query_id_for_record(record, idx)
        selected_hits = select_relevant_irrelevant_hits(record, query_id, int(relevant_k), int(irrelevant_k))
        for rank, hit in enumerate(selected_hits, start=1):
            output.append({
                **(provenance or {}),
                "query_id": query_id,
                "creator_id": hit_creator_id(hit),
                "document_id": hit_document_id(hit),
                "retrieval_rank": rank,
                "retrieval_score": None,
                "label": hit_label(hit),
                "prior_attention": hit_prior_attention(hit),
                "selection_bucket": "relevant" if hit_label(hit) > 0.0 else "irrelevant",
                "selected_candidate_count": total_k,
            })
    return output


def preselected_metric_rows(records: List[dict], relevant_k: int, irrelevant_k: int, provenance: Optional[dict] = None) -> List[dict]:
    rows = []
    total_k = int(relevant_k) + int(irrelevant_k)
    for idx, record in enumerate(records, start=1):
        query_id = query_id_for_record(record, idx)
        hits = strict_valid_hits(record, query_id)
        relevant_available = sum(1 for hit in hits if hit_label(hit) > 0.0)
        irrelevant_available = sum(1 for hit in hits if hit_label(hit) == 0.0)
        select_relevant_irrelevant_hits(record, query_id, int(relevant_k), int(irrelevant_k))
        rows.append({
            **(provenance or {}),
            "query_id": query_id,
            "K": total_k,
            "candidate_count": len(hits),
            "selected_candidate_count": total_k,
            "relevant_k": int(relevant_k),
            "irrelevant_k": int(irrelevant_k),
            "relevant_available_count": relevant_available,
            "irrelevant_available_count": irrelevant_available,
            "selected_relevant_count": int(relevant_k),
            "selected_irrelevant_count": int(irrelevant_k),
        })
    return rows


def retrieval_candidate_rows(records: List[dict], k: int, method: str, provenance: Optional[dict] = None, encoder: Optional[DenseEncoder] = None) -> List[dict]:
    output = []
    for idx, record in enumerate(records, start=1):
        query_id = query_id_for_record(record, idx)
        ranked = ranked_hits_for_record(record, method, encoder)
        for rank, item in enumerate(ranked[:k], start=1):
            hit = item["hit"]
            output.append({
                **(provenance or {}),
                "query_id": query_id,
                "creator_id": hit_creator_id(hit),
                "document_id": hit_document_id(hit),
                "retrieval_rank": rank,
                "retrieval_score": item["score"],
                "label": hit_label(hit),
                "prior_attention": hit_prior_attention(hit),
            })
    return output


def write_topk_outputs(input_jsonl: Path, output_dir: Path, k: int, method: str) -> dict:
    return write_topk_for_method(input_jsonl, output_dir, k, method)


def write_topk_for_method(
    input_jsonl: Path,
    output_dir: Path,
    k: int,
    method: str,
    embedding_model: Optional[str] = None,
    embedding_batch_size: Optional[int] = None,
    device: str = "auto",
    encoder: Optional[DenseEncoder] = None,
) -> dict:
    records = read_jsonl(input_jsonl)
    active_encoder = encoder
    if method == "dense" and active_encoder is None:
        active_encoder = load_dense_encoder(str(embedding_model or ""), int(embedding_batch_size or 32), device)
    provenance = retrieval_provenance(
        dataset_provenance(records, input_jsonl),
        method,
        k,
        embedding_model=getattr(active_encoder, "model_name", embedding_model) if method == "dense" else None,
        embedding_batch_size=getattr(active_encoder, "batch_size", embedding_batch_size) if method == "dense" else None,
        device=getattr(active_encoder, "device", selected_device(device)) if method == "dense" else None,
        effective_max_sequence_length=getattr(active_encoder, "effective_max_sequence_length", None) if method == "dense" else None,
    )
    split = provenance["dataset_split"]
    topk_records = []
    per_query_metrics = []
    for idx, record in enumerate(records, start=1):
        ranked = ranked_hits_for_record(record, method, active_encoder)
        selected_hits = [item["hit"] for item in ranked[:k]]
        topk_records.append(selected_output_record(record, selected_hits, provenance))
        if method in {"bm25", "dense"}:
            per_query_metrics.append(metric_row(query_id_for_record(record, idx), ranked, k))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "test_topK.jsonl", topk_records)
    write_jsonl(output_dir / "retrieval_candidates.jsonl", retrieval_candidate_rows(records, k, method, provenance, active_encoder))
    if method in {"bm25", "dense"}:
        write_csv(output_dir / "retrieval_metrics.csv", [aggregate_metric_rows(per_query_metrics, k, method, split, provenance)])
    else:
        write_csv(output_dir / "retrieval_metrics.csv", candidate_stat_rows(records, k, method, split, provenance))

    candidate_counts = [len(valid_hits(record)) for record in records]
    return {
        **provenance,
        "method": method,
        "retrieval_k": k,
        "num_queries_written": len(topk_records),
        "queries_with_fewer_than_k_candidates": sum(1 for count in candidate_counts if count < k),
        "output_dir": str(output_dir),
    }


def prepare_topk_datasets(input_jsonl: Path, retrieval_k: int, output_root: Path = DEFAULT_OUTPUT_ROOT) -> List[dict]:
    return [
        write_topk_outputs(input_jsonl, output_root / "ground_truth", GROUND_TRUTH_K, "ground_truth"),
        write_topk_outputs(input_jsonl, output_root / "bm25", retrieval_k, "bm25"),
    ]


def prepare_outputs(
    input_jsonl: Path,
    retrieval_k: int,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    method: str = "bm25",
    embedding_model: Optional[str] = None,
    embedding_batch_size: int = 32,
    device: str = "auto",
    encoder: Optional[DenseEncoder] = None,
) -> List[dict]:
    summaries = [write_topk_for_method(input_jsonl, output_root / "ground_truth", GROUND_TRUTH_K, "ground_truth")]
    if method == "bm25":
        summaries.append(write_topk_for_method(input_jsonl, output_root / "bm25", retrieval_k, "bm25"))
    elif method == "dense":
        summaries.append(write_topk_for_method(
            input_jsonl,
            output_dir_for_method(output_root, "dense", embedding_model),
            retrieval_k,
            "dense",
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            device=device,
            encoder=encoder,
        ))
    else:
        raise ValueError(f"Unsupported --method {method!r}")
    return summaries


def write_preselected_pool(
    input_jsonl: Path,
    output_root: Path,
    relevant_k: int,
    irrelevant_k: int,
) -> dict:
    records = read_jsonl(input_jsonl)
    total_k = int(relevant_k) + int(irrelevant_k)
    provenance = retrieval_provenance(
        dataset_provenance(records, input_jsonl),
        "preselected",
        total_k,
    )
    output_dir = preselected_output_dir(output_root, int(relevant_k), int(irrelevant_k))
    selected_records = []
    for idx, record in enumerate(records, start=1):
        query_id = query_id_for_record(record, idx)
        selected_hits = select_relevant_irrelevant_hits(record, query_id, int(relevant_k), int(irrelevant_k))
        selected_records.append(selected_output_record(record, selected_hits, provenance))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "test_topK.jsonl", selected_records)
    write_jsonl(output_dir / "retrieval_candidates.jsonl", preselected_candidate_rows(records, int(relevant_k), int(irrelevant_k), provenance))
    write_csv(output_dir / "retrieval_metrics.csv", preselected_metric_rows(records, int(relevant_k), int(irrelevant_k), provenance))
    return {
        **provenance,
        "method": "preselected",
        "relevant_k": int(relevant_k),
        "irrelevant_k": int(irrelevant_k),
        "retrieval_k": total_k,
        "num_queries_written": len(selected_records),
        "output_dir": str(output_dir),
        "output_jsonl": str(output_dir / "test_topK.jsonl"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone retrieval/preparation stage before Study A")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--mode", choices=("select_k", "prepare"), default="select_k")
    parser.add_argument("--method", choices=("bm25", "dense"), default="bm25")
    parser.add_argument("--embedding_model", default=None)
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrieval_k", type=int, default=None)
    parser.add_argument("--relevant_k", type=int, default=None)
    parser.add_argument("--irrelevant_k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl)
    output_root = Path(args.output_root)
    if args.mode == "select_k":
        write_k_selection(
            input_jsonl,
            output_root,
            method=args.method,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            device=args.device,
        )
        return
    if (args.relevant_k is None) ^ (args.irrelevant_k is None):
        raise ValueError("--relevant_k and --irrelevant_k must be provided together")
    if args.relevant_k is not None and args.relevant_k <= 0:
        raise ValueError("--relevant_k must be positive")
    if args.irrelevant_k is not None and args.irrelevant_k <= 0:
        raise ValueError("--irrelevant_k must be positive")
    if args.relevant_k is not None and args.irrelevant_k is not None and args.retrieval_k is not None:
        expected_total = int(args.relevant_k) + int(args.irrelevant_k)
        if int(args.retrieval_k) != expected_total:
            raise ValueError(f"--retrieval_k={args.retrieval_k} must equal relevant_k + irrelevant_k = {expected_total}")
    if args.retrieval_k is not None:
        if args.retrieval_k <= 0:
            raise ValueError("--retrieval_k must be a positive integer for mode=prepare")
        prepare_outputs(
            input_jsonl,
            int(args.retrieval_k),
            output_root,
            method=args.method,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            device=args.device,
        )
    elif args.relevant_k is None or args.irrelevant_k is None:
        raise ValueError("--retrieval_k must be a positive integer for mode=prepare unless both --relevant_k and --irrelevant_k are provided")
    if args.relevant_k is not None and args.irrelevant_k is not None:
        write_preselected_pool(input_jsonl, output_root, int(args.relevant_k), int(args.irrelevant_k))


if __name__ == "__main__":
    main()
