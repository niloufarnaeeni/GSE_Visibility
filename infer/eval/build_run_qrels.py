import json
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import tqdm
from rag_retrieval import Reranker

_CREATOR_RE = re.compile(r"\bCreator\s+(C\d+)\b", re.IGNORECASE)



def make_qid(project: str, trial_id: int, fold_id: int, split: str, idx: int) -> str:
    return f"{project}|t{trial_id}|f{fold_id}|{split}|i{idx:04d}"


def parse_qid_project(qid: str) -> str:
    return qid.split("|")[0].strip()


def load_creator_prior_attention(raw_dir):
    prior_attention_df = pd.read_csv(raw_dir / "prior_attention_scores.csv")
    creators_df = pd.read_csv(raw_dir / "creator_details.csv")

    prior_attention_df["username_norm"] = prior_attention_df["username"].str.lower().str.lstrip("@")
    creators_df["creator_id_norm"] = creators_df["Creator_ID"].str.lower().str.lstrip("@")

    merged = creators_df.merge(
        prior_attention_df,
        left_on="creator_id_norm",
        right_on="username_norm",
        how="left",
    )

    prior_attention_by_creator = {
        row["creator_code"]: float(row["prior_attention"])
        for _, row in merged.iterrows()
        if not pd.isna(row["prior_attention"])
    }

    return prior_attention_by_creator


def extract_creator_id(doc_text: str) -> Optional[str]:
    if not doc_text:
        return None
    m = _CREATOR_RE.search(doc_text)
    return m.group(1) if m else None


def extract_project_name_from_query(query: str) -> Optional[str]:
    if not query:
        return None
    m = re.search(
        r"Creators\s+suitable\s+for\s+the\s+(.+?)\s+project",
        query,
        re.IGNORECASE,
    )
    return m.group(1).strip().strip(".") if m else None


def get_project_name_from_record(rec: dict, idx: int) -> str:
    project = str(rec.get("project_name", "")).strip()

    if project:
        return project

    return extract_project_name_from_query(rec.get("query", "")) or f"q{idx}"


def convert_hits_to_graded_qrels(hits: List[dict], relevance_if_label_gt: float = 0.0) -> Dict[str, int]:
    """Canonical graded-label conversion shared by evaluation and Study A."""
    positives = []
    expected_ids = set()
    for hit in hits:
        explicit_id = str(hit.get("creator_id", "")).strip()
        parsed_id = extract_creator_id(hit.get("content", ""))
        docid = explicit_id or parsed_id
        if not docid:
            raise ValueError("Cannot convert hit to graded qrels: missing creator_id and parsable content")
        label = float(hit["label"])
        if label > relevance_if_label_gt:
            expected_ids.add(docid)
            positives.append((docid, label))
    positives_sorted = sorted(positives, key=lambda x: x[1], reverse=True)
    n_pos = len(positives_sorted)
    qrels = {docid: n_pos - rank_idx + 1 for rank_idx, (docid, _) in enumerate(positives_sorted, start=1)}
    if set(qrels) != expected_ids:
        raise ValueError(f"Canonical graded qrels ID mismatch: expected={sorted(expected_ids)}, actual={sorted(qrels)}")
    return qrels


def load_grouped_labels_from_jsonl(jsonl_path: str) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      qid -> {docid: graded_label}
    Must use the exact same qid construction as build_qrel_and_run_from_grouped_jsonl.
    """
    qid_to_labels: Dict[str, Dict[str, float]] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            query = rec["query"]
            hits = rec["hits"]

            project = get_project_name_from_record(rec, idx)
            trial_id = rec.get("trial_id", 0)
            fold_id = rec.get("fold_id", 0)
            split = rec.get("split", "unknown")
            qid = make_qid(project, trial_id, fold_id, split, idx)

            label_map = {}
            for h in hits:
                docid = str(h.get("creator_id", "")).strip() or extract_creator_id(h["content"])
                if docid is not None:
                    label_map[docid] = float(h["label"])
            qid_to_labels[qid] = label_map

    return qid_to_labels


def creator_id_for_hit(hit: dict) -> Optional[str]:
    explicit_id = str(hit.get("creator_id", "")).strip()
    return explicit_id or extract_creator_id(hit.get("content", ""))


def hit_for_rerank_result(hits: List[dict], result) -> Optional[dict]:
    try:
        idx = int(result.doc_id)
    except (TypeError, ValueError):
        return None
    return hits[idx] if 0 <= idx < len(hits) else None


def build_qrel_and_run_from_grouped_jsonl(
    jsonl_path: str,
    ranker: Reranker,
    k: Optional[int] = None,
    relevance_if_label_gt: float = 0.0,
    eval_batch_size: int = 4,
    eval_max_length: int = 256,
):
    qrels_binary: Dict[str, Dict[str, int]] = {}
    qrels_graded: Dict[str, Dict[str, int]] = {}
    run: Dict[str, Dict[str, float]] = {}
    qid_to_query = {}

    total_queries = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        total_queries = sum(1 for line in f if line.strip())

    print(f"[EVAL] Reranking {total_queries} queries from {jsonl_path}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        progress = tqdm.tqdm(
            enumerate(f, start=1),
            total=total_queries,
            desc="Reranking queries",
            unit="query",
        )
        for idx, line in progress:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            query = rec["query"]
            hits = rec["hits"]

            docs: List[str] = [h["content"] for h in hits]
            labels: List[float] = [float(h["label"]) for h in hits]

            project = get_project_name_from_record(rec, idx)
            trial_id = rec.get("trial_id", 0)
            fold_id = rec.get("fold_id", 0)
            split = rec.get("split", "unknown")
            qid = make_qid(project, trial_id, fold_id, split, idx)
            qid_to_query[qid] = query

            ranked = ranker.rerank(
                query,
                docs,
                batch_size=eval_batch_size,
                max_length=eval_max_length,
            )
            results = ranked.results[:k] if k else ranked.results
            progress.set_postfix({"docs": len(docs), "qid": qid}, refresh=False)

            run[qid] = {}
            for r in results:
                original_hit = hit_for_rerank_result(hits, r)
                docid = creator_id_for_hit(original_hit) if original_hit is not None else extract_creator_id(r.text)
                docid = docid or f"DOC_{r.doc_id}"
                run[qid][docid] = float(r.score)

            qrels_binary[qid] = {}
            qrels_graded[qid] = {}

            qrels_graded[qid] = convert_hits_to_graded_qrels(
                [{"creator_id": creator_id_for_hit(h), "content": h["content"], "label": h["label"]} for h in hits],
                relevance_if_label_gt,
            )
            qrels_binary[qid] = {docid: 1 for docid in qrels_graded[qid]}

    return qrels_binary, qrels_graded, run, qid_to_query
