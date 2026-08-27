from dataclasses import dataclass
from typing import List

from .data import Hit, QueryRecord


SUPPORTED_CANDIDATE_SOURCES = ("full", "preselected")


@dataclass
class CandidateSet:
    query_id: str
    source: str
    hits: List[Hit]
    details: dict


class CandidateSource:
    name = ""

    def select(self, record: QueryRecord, retrieval_k: int) -> CandidateSet:
        raise NotImplementedError


class FullCandidateSource(CandidateSource):
    name = "full"

    def select(self, record: QueryRecord, retrieval_k: int) -> CandidateSet:
        valid_hits = list(record.hits)
        positives = [hit.creator_id for hit in valid_hits if hit.label > 0]
        negatives = [hit.creator_id for hit in valid_hits if hit.label == 0]
        return CandidateSet(
            query_id=record.query_id,
            source=self.name,
            hits=valid_hits,
            details={
                "query_id": record.query_id,
                "candidate_source": self.name,
                "candidate_count": len(valid_hits),
                "positive_count": len(positives),
                "negative_count": len(negatives),
                "candidate_creator_ids": [hit.creator_id for hit in valid_hits],
                "candidate_document_ids": [hit.document_id for hit in valid_hits],
            },
        )


class PreselectedCandidateSource(CandidateSource):
    name = "preselected"

    def select(self, record: QueryRecord, retrieval_k: int) -> CandidateSet:
        valid_hits = list(record.hits)
        positives = [hit.creator_id for hit in valid_hits if hit.label > 0]
        negatives = [hit.creator_id for hit in valid_hits if hit.label == 0]
        return CandidateSet(
            query_id=record.query_id,
            source=self.name,
            hits=valid_hits,
            details={
                "query_id": record.query_id,
                "candidate_source": self.name,
                "candidate_count": len(valid_hits),
                "positive_count": len(positives),
                "negative_count": len(negatives),
                "candidate_creator_ids": [hit.creator_id for hit in valid_hits],
                "candidate_document_ids": [hit.document_id for hit in valid_hits],
                "expected_candidate_count": int(retrieval_k),
                "exact_candidate_count_match": len(valid_hits) == int(retrieval_k),
            },
        )


def build_candidate_source(name: str) -> CandidateSource:
    normalized = str(name).strip().lower()
    if normalized == "full":
        return FullCandidateSource()
    if normalized == "preselected":
        return PreselectedCandidateSource()
    raise ValueError(f"Unsupported candidate_source {name!r}; choose from {SUPPORTED_CANDIDATE_SOURCES}")
