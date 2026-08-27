from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import scipy.sparse as sp

from rag_retrieval.infer.eval.skill_coverage import (
    build_Cid_to_member_index,
    compute_skill_coverage_at_k,
    extract_required_skill_indices,
    get_skipteams_from_splits,
    gen_member_skill_cooccurrence,
    load_skill_keyword_index_map,
)


SKILL_RESOURCE_FILES = ("teamsvecs.pkl", "indexes.pkl", "splits.t5.r0.85.pkl", "gpt5_skills.csv")


def resolve_skill_raw_dir(source: str | Path) -> Optional[Path]:
    source = Path(source)
    candidates = [
        source,
        source / "raw",
        source.parent / "raw",
        source.parent.parent / "raw",
    ]
    for candidate in candidates:
        if all((candidate / name).exists() for name in SKILL_RESOURCE_FILES):
            return candidate
    return None


def _creator_member_index(docid_to_member_idx: Dict[str, int], creator_id: str) -> Optional[int]:
    creator_id = str(creator_id).strip()
    if not creator_id:
        return None
    midx = docid_to_member_idx.get(creator_id)
    if midx is None:
        midx = docid_to_member_idx.get(creator_id.lower())
    if midx is None and creator_id.upper().startswith("C"):
        midx = docid_to_member_idx.get(creator_id.lower())
    return midx


def skill_split_metadata(row: dict) -> tuple[int, int, str]:
    split_name = str(row.get("split", "")).strip().lower()
    trial_raw = row.get("trial_id")
    fold_raw = row.get("fold_id")
    query_id = row.get("query_id", "unknown")
    if split_name in {"valid", "test"} and (trial_raw is None or fold_raw is None):
        raise ValueError(
            f"Skill coverage for valid/test query_id={query_id} requires explicit trial_id, fold_id, and split metadata"
        )
    if not split_name:
        split_name = "unknown"
    try:
        trial_id = int(trial_raw) if trial_raw is not None else 0
        fold_id = int(fold_raw) if fold_raw is not None else 0
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Skill coverage query_id={query_id} has non-integer trial_id/fold_id metadata") from exc
    if split_name in {"valid", "test"} and split_name == "unknown":
        raise ValueError(f"Skill coverage for valid/test query_id={query_id} cannot use split='unknown'")
    return trial_id, fold_id, split_name


class SkillCoverageContext:
    def __init__(self, raw_dir: Path, cache_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.cache_dir = Path(cache_dir)
        with (self.raw_dir / "teamsvecs.pkl").open("rb") as f:
            self.teamsvecs = pickle.load(f)
        with (self.raw_dir / "indexes.pkl").open("rb") as f:
            self.indexes = pickle.load(f)
        with (self.raw_dir / "splits.t5.r0.85.pkl").open("rb") as f:
            self.splits_bundle = pickle.load(f)
        self.skills_csv = self.raw_dir / "gpt5_skills.csv"
        self.keyword_to_skill_index = load_skill_keyword_index_map(self.skills_csv, self.indexes["s2i"])
        self.docid_to_member_idx = build_Cid_to_member_index(self.indexes)
        self._member_skill_co_by_key: Dict[tuple, sp.csr_matrix] = {}

    def required_skill_indices(self, query_text: str) -> Optional[List[int]]:
        required = extract_required_skill_indices(query_text, self.indexes["s2i"], self.keyword_to_skill_index)
        return required or None

    def member_skill_co(self, trial_id: int, fold_id: int, split_name: str) -> sp.csr_matrix:
        split_name = str(split_name).strip().lower() or "unknown"
        if split_name == "unknown":
            raise ValueError("Skill coverage requires explicit split metadata; refusing split='unknown'")
        key = (trial_id, fold_id, split_name)
        if key not in self._member_skill_co_by_key:
            skipteams = get_skipteams_from_splits(self.splits_bundle, trial_id, fold_id, split_name)
            cache_path = self.cache_dir / f"skillcoverage_member_skill_co_t{trial_id}_f{fold_id}_{split_name}.pkl"
            self._member_skill_co_by_key[key] = gen_member_skill_cooccurrence(
                teamsvecs=self.teamsvecs,
                cache_path=cache_path,
                skipteams=skipteams,
            )
        return self._member_skill_co_by_key[key]

    def list_skill_coverage(
        self,
        query_id: str,
        query_text: str,
        creator_ids: Iterable[str],
        k: int,
        trial_id: int,
        fold_id: int,
        split_name: str,
    ) -> Optional[float]:
        required = self.required_skill_indices(query_text)
        if not required:
            return None
        ordered = [str(creator_id).strip() for creator_id in creator_ids if str(creator_id).strip()][:k]
        if not ordered:
            return None
        if not any(_creator_member_index(self.docid_to_member_idx, creator_id) is not None for creator_id in ordered):
            return None
        run = {query_id: {creator_id: float(len(ordered) - idx) for idx, creator_id in enumerate(ordered)}}
        result = compute_skill_coverage_at_k(
            run=run,
            qid_to_query={query_id: query_text},
            indexes=self.indexes,
            member_skill_co=self.member_skill_co(trial_id, fold_id, split_name),
            ks=(int(k),),
            skill_keywords_csv=self.skills_csv,
        )
        value = result.get(query_id, {}).get(f"skill_coverage_{int(k)}")
        return float(value) if value is not None and math.isfinite(float(value)) else None

    def individual_skill_coverage(
        self,
        query_id: str,
        query_text: str,
        creator_id: str,
        trial_id: int,
        fold_id: int,
        split_name: str,
    ) -> Optional[float]:
        required = self.required_skill_indices(query_text)
        if not required:
            return None
        midx = _creator_member_index(self.docid_to_member_idx, creator_id)
        if midx is None:
            return None
        member_skill_bool = (self.member_skill_co(trial_id, fold_id, split_name) > 0).tocsr()
        covered = set(member_skill_bool[midx].indices).intersection(set(required))
        return len(covered) / len(set(required))


def load_skill_context(source: str | Path, cache_dir: str | Path) -> Optional[SkillCoverageContext]:
    raw_dir = resolve_skill_raw_dir(source)
    if raw_dir is None:
        return None
    return SkillCoverageContext(raw_dir=raw_dir, cache_dir=Path(cache_dir))


def per_query_skill_metrics(
    ranked_rows: List[dict],
    generation_rows: List[dict],
    source: str | Path,
    cache_dir: str | Path,
    input_k: int,
    output_k: int,
) -> Dict[str, dict]:
    context = load_skill_context(source, cache_dir)
    if context is None:
        return {}
    ranked_by_qid = {row.get("query_id"): row for row in ranked_rows if row.get("query_id")}
    out: Dict[str, dict] = {}
    for generation in generation_rows:
        query_id = generation.get("query_id")
        if not query_id:
            continue
        ranked = ranked_by_qid.get(query_id) or {}
        meta_source = {**ranked, **generation}
        trial_id, fold_id, split_name = skill_split_metadata(meta_source)
        query_text = generation.get("query") or ranked.get("query") or ""
        ranked_ids = [item.get("creator_id") for item in (ranked.get("ranked_candidates") or [])[:input_k]]
        valid_recs = generation.get("valid_recommendations") or []
        after_ids = [rec.get("creator_id") for rec in sorted(valid_recs, key=lambda item: int(item.get("rank", 9999)))]
        complete = bool(generation.get("generation_success")) and len(after_ids) == int(output_k)
        before10 = context.list_skill_coverage(query_id, query_text, ranked_ids, min(10, int(input_k)), trial_id, fold_id, split_name)
        before5 = context.list_skill_coverage(query_id, query_text, ranked_ids, min(5, int(input_k)), trial_id, fold_id, split_name)
        after5 = context.list_skill_coverage(query_id, query_text, after_ids, int(output_k), trial_id, fold_id, split_name) if complete else None
        out[query_id] = {
            "skill_coverage_before@10": before10,
            "skill_coverage_before@5": before5,
            "skill_coverage_after@5": after5,
            "delta_skill_coverage@5": (
                after5 - before5 if after5 is not None and before5 is not None else None
            ),
        }
    return out


def skill_metric_summary(metric_rows: List[dict]) -> dict:
    total = len(metric_rows)
    usable = sum(
        row.get("skill_coverage_before@10") is not None
        and row.get("skill_coverage_before@5") is not None
        and row.get("skill_coverage_after@5") is not None
        for row in metric_rows
    )
    return {
        "skill_metric_query_count": usable,
        "skill_metric_coverage": usable / total if total else 0.0,
    }
