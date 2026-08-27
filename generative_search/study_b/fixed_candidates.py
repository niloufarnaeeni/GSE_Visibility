from dataclasses import dataclass
from typing import List

from rag_retrieval.generative_search.study_a.data import Hit, QueryRecord


FIXED_CANDIDATE_COUNT = 10
CANDIDATE_SOURCE = "prepared_ground_truth_top10"
FIXED_CANDIDATE_RULE = "ground_truth_label_desc_creator_id_asc_top10"


@dataclass
class FixedCandidateSet:
    query_id: str
    hits: List[Hit]
    details: dict


def build_fixed_top10(record: QueryRecord, k: int = FIXED_CANDIDATE_COUNT) -> FixedCandidateSet:
    valid_hits = list(record.hits)
    if len(valid_hits) != len({hit.creator_id for hit in valid_hits}):
        raise ValueError(f"duplicate creator_id values in query_id={record.query_id}")
    if len(valid_hits) < k:
        raise ValueError(f"query_id={record.query_id} has {len(valid_hits)} valid creators; Study B requires at least {k}")
    ranked = sorted(valid_hits, key=lambda hit: (-float(hit.label), str(hit.creator_id)))[:k]
    if len(ranked) != k:
        raise ValueError(f"query_id={record.query_id} produced {len(ranked)} fixed candidates; expected {k}")
    return FixedCandidateSet(
        query_id=record.query_id,
        hits=ranked,
        details={
            "query_id": record.query_id,
            "candidate_source": CANDIDATE_SOURCE,
            "fixed_candidate_rule": FIXED_CANDIDATE_RULE,
            "fixed_candidate_count": k,
            "fixed_candidate_creator_ids": [hit.creator_id for hit in ranked],
            "fixed_candidate_document_ids": [hit.document_id for hit in ranked],
            "construction_rule": "sort record.hits by label descending, creator_id ascending; keep exactly 10",
        },
    )


def fixed_from_prepared(record: QueryRecord, k: int = FIXED_CANDIDATE_COUNT) -> FixedCandidateSet:
    valid_hits = list(record.hits)
    if len(valid_hits) != k:
        raise ValueError(f"query_id={record.query_id} must contain exactly {k} prepared Study B hits; got {len(valid_hits)}")
    if len(valid_hits) != len({hit.creator_id for hit in valid_hits}):
        raise ValueError(f"duplicate creator_id values in prepared query_id={record.query_id}")
    deterministic = [hit.creator_id for hit in sorted(valid_hits, key=lambda hit: (-float(hit.label), str(hit.creator_id)))]
    current = [hit.creator_id for hit in valid_hits]
    if current != deterministic:
        raise ValueError(
            f"prepared Study B query_id={record.query_id} is not ordered by label descending and creator_id ascending"
        )
    return FixedCandidateSet(
        query_id=record.query_id,
        hits=valid_hits,
        details={
            "query_id": record.query_id,
            "candidate_source": CANDIDATE_SOURCE,
            "fixed_candidate_rule": FIXED_CANDIDATE_RULE,
            "fixed_candidate_count": k,
            "fixed_candidate_creator_ids": current,
            "fixed_candidate_document_ids": [hit.document_id for hit in valid_hits],
            "construction_rule": "precomputed Study B JSONL with exactly 10 hits sorted by label descending, creator_id ascending",
        },
    )


def assert_same_creator_set(query_id: str, before_creator_ids: List[str], after_creator_ids: List[str]) -> None:
    before = [str(value) for value in before_creator_ids]
    after = [str(value) for value in after_creator_ids]
    if len(before) != FIXED_CANDIDATE_COUNT or len(after) != FIXED_CANDIDATE_COUNT:
        raise ValueError(f"query_id={query_id} must have exactly {FIXED_CANDIDATE_COUNT} creators before and after reranking")
    if len(set(before)) != len(before):
        raise ValueError(f"query_id={query_id} fixed candidate set contains duplicate creator IDs")
    if len(set(after)) != len(after):
        raise ValueError(f"query_id={query_id} reranked output contains duplicate creator IDs")
    if set(before) != set(after):
        raise ValueError(f"query_id={query_id} reranker changed the fixed creator set: before={sorted(before)}, after={sorted(after)}")
