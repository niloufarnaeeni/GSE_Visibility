import argparse
import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Tuple

from rag_retrieval.generative_search.study_a.data import stable_query_id


FIXED_CANDIDATE_COUNT = 10
CONSTRUCTION_RULE = "sort valid hits by ground-truth label descending, creator_id ascending; keep exactly 10"
DEFAULT_INPUT = Path(__file__).resolve().parents[3] / "data" / "kaito" / "large_data_creator_profile" / "test&valid.jsonl"
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "study_b" / "test&valid.jsonl"


def read_jsonl(path: Path) -> Iterable[Tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield idx, json.loads(line)


def finite_nonnegative(value, label: str, query_id: str, creator_id: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"query_id={query_id} creator_id={creator_id} has nonnumeric {label}: {value!r}") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"query_id={query_id} creator_id={creator_id} has invalid {label}: {value!r}")
    return numeric


def validate_hit(hit: dict, query_id: str, input_row: int) -> Tuple[str, float]:
    if not isinstance(hit, dict):
        raise ValueError(f"query_id={query_id} input_row={input_row} has a non-object hit")
    creator_id = str(hit.get("creator_id", "")).strip()
    if not creator_id:
        raise ValueError(f"query_id={query_id} input_row={input_row} has a hit with empty creator_id")
    content = str(hit.get("content", "")).strip()
    if not content:
        raise ValueError(f"query_id={query_id} creator_id={creator_id} has empty content")
    if "label" not in hit:
        raise ValueError(f"query_id={query_id} creator_id={creator_id} is missing label")
    if "prior_attention" not in hit or hit.get("prior_attention") is None:
        raise ValueError(f"query_id={query_id} creator_id={creator_id} is missing prior_attention")
    label = finite_nonnegative(hit.get("label"), "label", query_id, creator_id)
    finite_nonnegative(hit.get("prior_attention"), "prior_attention", query_id, creator_id)
    return creator_id, label


def deterministic_top10_hits(record: dict, input_row: int, k: int = FIXED_CANDIDATE_COUNT) -> Tuple[str, List[dict]]:
    query_id = str(record.get("query_id") or stable_query_id(record, input_row))
    hits = record.get("hits")
    if not isinstance(hits, list):
        raise ValueError(f"query_id={query_id} input_row={input_row} has missing/non-list hits")
    keyed = []
    seen = set()
    for hit in hits:
        creator_id, label = validate_hit(hit, query_id, input_row)
        if creator_id in seen:
            raise ValueError(f"query_id={query_id} has duplicate creator_id={creator_id}")
        seen.add(creator_id)
        keyed.append((creator_id, label, hit))
    if len(keyed) < k:
        raise ValueError(f"query_id={query_id} has {len(keyed)} valid creators; Study B requires at least {k}")
    selected = sorted(keyed, key=lambda item: (-item[1], item[0]))[:k]
    return query_id, [deepcopy(hit) for _, _, hit in selected]


def make_study_b_record(record: dict, input_row: int, k: int = FIXED_CANDIDATE_COUNT) -> Tuple[str, dict]:
    query_id, selected_hits = deterministic_top10_hits(record, input_row, k)
    out = deepcopy(record)
    out["hits"] = selected_hits
    return query_id, out


def verify_output(input_path: Path, output_path: Path, eligible_query_ids: List[str], k: int = FIXED_CANDIDATE_COUNT) -> dict:
    expected = {}
    split_counts = Counter()
    for input_row, record in read_jsonl(input_path):
        query_id, selected_hits = deterministic_top10_hits(record, input_row, k)
        expected[query_id] = [str(hit["creator_id"]).strip() for hit in selected_hits]

    output_query_ids = []
    for output_row, record in read_jsonl(output_path):
        query_id = str(record.get("query_id") or stable_query_id(record, output_row))
        hits = record.get("hits")
        if not isinstance(hits, list) or len(hits) != k:
            raise ValueError(f"output query_id={query_id} has {len(hits) if isinstance(hits, list) else 'non-list'} hits; expected {k}")
        creator_ids = [str(hit.get("creator_id", "")).strip() for hit in hits]
        if len(creator_ids) != len(set(creator_ids)):
            raise ValueError(f"output query_id={query_id} has duplicate creator IDs")
        if creator_ids != expected.get(query_id):
            raise ValueError(f"output query_id={query_id} does not match deterministic Study B top 10")
        output_query_ids.append(query_id)
        split_counts[str(record.get("split", "test")).strip().lower() or "test"] += 1

    if set(output_query_ids) != set(eligible_query_ids):
        raise ValueError("output query_id set does not match eligible input query_id set")
    if len(output_query_ids) != len(eligible_query_ids):
        raise ValueError(f"output query count {len(output_query_ids)} != eligible input query count {len(eligible_query_ids)}")
    return dict(split_counts)


def prepare_dataset(input_path: Path, output_path: Path, summary_path: Path = None, k: int = FIXED_CANDIDATE_COUNT) -> dict:
    rows = []
    eligible_query_ids = []
    total_input = 0
    for input_row, record in read_jsonl(input_path):
        total_input += 1
        query_id, out = make_study_b_record(record, input_row, k)
        eligible_query_ids.append(query_id)
        rows.append(out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    split_counts = verify_output(input_path, output_path, eligible_query_ids, k)
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_input_queries": total_input,
        "total_output_queries": len(rows),
        "test_query_count": split_counts.get("test", 0),
        "valid_query_count": split_counts.get("valid", 0),
        "excluded_query_count": total_input - len(rows),
        "fixed_candidate_count": k,
        "construction_rule": CONSTRUCTION_RULE,
    }
    summary_path = summary_path or output_path.parent / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare fixed-candidate Study B JSONL")
    parser.add_argument("--input_jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--output_jsonl", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary_json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json) if args.summary_json else output_path.parent / "dataset_summary.json"
    summary = prepare_dataset(Path(args.input_jsonl), output_path, summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
