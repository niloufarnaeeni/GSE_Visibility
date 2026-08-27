import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import tqdm
import yaml
from transformers import AutoConfig, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze JSONL dataset token lengths using the same tokenizer "
            "as LLM decoder training. Can run directly on a dataset folder."
        )
    )

    parser.add_argument(
        "--data_dir",
        required=True,
        help=(
            "Dataset folder containing train.jsonl, valid.jsonl, and test.jsonl. "
            "Also supports files inside <data_dir>/processed/."
        ),
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional path to training YAML config. If provided, model_name_or_path, "
            "max_len, query_format, document_format, seq, and special_token are read from it "
            "unless explicitly overridden by CLI arguments."
        ),
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        help="Hugging Face model name/path for tokenizer. Required if --config is not provided.",
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Training max_len. If not provided, read from config or default to 512.",
    )

    parser.add_argument(
        "--dataset_type",
        default="auto",
        choices=["auto", "prompt", "pointwise", "grouped"],
        help=(
            "Dataset schema type. Use auto to infer from JSONL fields. "
            "prompt = prompt/target_text schema, pointwise = query/content schema, "
            "grouped = query/hits schema."
        ),
    )

    parser.add_argument(
        "--query_format",
        default=None,
        help="Optional query format, e.g. 'query: {}'. Read from config if omitted.",
    )

    parser.add_argument(
        "--document_format",
        default=None,
        help="Optional document format, e.g. 'document: {}'. Read from config if omitted.",
    )

    parser.add_argument(
        "--seq",
        default=None,
        help="Separator string between query and document. Read from config if omitted.",
    )

    parser.add_argument(
        "--special_token",
        default=None,
        help="Special suffix token/string. Read from config if omitted.",
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of longest examples to print per split.",
    )

    parser.add_argument(
        "--debug_print_text",
        action="store_true",
        help="Print full text for the longest examples.",
    )

    return parser.parse_args()


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object in {path}, got {type(data).__name__}")

    return data


def get_value(
    *,
    cli_value: Any,
    cfg: Dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    if cli_value is not None:
        return cli_value
    return cfg.get(key, default)


def build_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )

    if hasattr(tokenizer, "deprecation_warnings"):
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    tokenizer.padding_side = "right"
    return tokenizer


def maybe_load_model_config(model_name_or_path: str):
    try:
        return AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"[WARN] Could not load model config for {model_name_or_path}: {exc}")
        return None


def summarize_model_limits(tokenizer, model_config) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "tokenizer.model_max_length": getattr(tokenizer, "model_max_length", None),
    }

    candidate_fields = [
        "max_position_embeddings",
        "n_positions",
        "max_seq_len",
        "max_sequence_length",
        "seq_length",
        "sliding_window",
    ]

    for field in candidate_fields:
        out[f"model.config.{field}"] = (
            getattr(model_config, field, None) if model_config is not None else None
        )

    return out


def safe_percentile(values: List[int], percentile: float) -> float:
    if not values:
        return 0.0

    if len(values) == 1:
        return float(values[0])

    vals = sorted(values)
    rank = (len(vals) - 1) * (percentile / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))

    if low == high:
        return float(vals[low])

    weight = rank - low
    return float(vals[low] * (1.0 - weight) + vals[high] * weight)


def find_split_file(data_dir: Path, split_name: str) -> Optional[Path]:
    """
    Supports:
      data_dir/train.jsonl
      data_dir/valid.jsonl
      data_dir/test.jsonl

    Also supports:
      data_dir/processed/train.jsonl
      data_dir/processed/valid.jsonl
      data_dir/processed/test.jsonl
    """

    aliases = {
        "train": ["train.jsonl"],
        "valid": ["valid.jsonl", "val.jsonl", "dev.jsonl"],
        "test": ["test.jsonl"],
    }

    candidates: List[Path] = []

    for filename in aliases[split_name]:
        candidates.append(data_dir / filename)
        candidates.append(data_dir / "processed" / filename)

    for path in candidates:
        if path.exists():
            return path

    return None


def infer_dataset_type(row: Dict[str, Any]) -> str:
    if "hits" in row and "query" in row:
        return "grouped"

    if "prompt" in row:
        return "prompt"

    if "query" in row and any(k in row for k in ["content", "document", "text"]):
        return "pointwise"

    if "text" in row:
        return "prompt"

    if "messages" in row:
        return "prompt"

    return "prompt"


def messages_to_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages)

    parts: List[str] = []

    for msg in messages:
        if isinstance(msg, dict):
            role = str(msg.get("role", "")).strip()
            content = str(msg.get("content", "")).strip()

            if role:
                parts.append(f"{role}: {content}")
            else:
                parts.append(content)
        else:
            parts.append(str(msg))

    return "\n".join(parts)


def get_prompt_and_target(row: Dict[str, Any]) -> Tuple[str, str]:
    prompt = ""

    for key in ["prompt", "input", "text", "question", "query"]:
        if key in row:
            prompt = str(row.get(key, "")).strip()
            break

    if not prompt and "messages" in row:
        prompt = messages_to_text(row["messages"]).strip()

    target = ""

    for key in ["target_text", "target", "completion", "response", "answer", "label_text"]:
        if key in row:
            target = str(row.get(key, "")).strip()
            break

    return prompt, target


def iter_samples(
    jsonl_path: Path,
    dataset_type: str,
    query_format: str,
    document_format: str,
    seq: str,
    special_token: str,
) -> Iterator[Dict[str, Any]]:
    dataset_type = str(dataset_type).strip().lower()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            current_type = dataset_type
            if current_type == "auto":
                current_type = infer_dataset_type(row)

            if current_type == "grouped":
                query = str(row.get("query", "")).strip()
                trial_id = row.get("trial_id", "na")
                fold_id = row.get("fold_id", "na")
                split = row.get("split", "unknown")
                hits = row.get("hits", [])

                if not isinstance(hits, list):
                    hits = []

                for hit_idx, hit in enumerate(hits, start=1):
                    if not isinstance(hit, dict):
                        continue

                    document = str(
                        hit.get("content")
                        or hit.get("document")
                        or hit.get("text")
                        or ""
                    ).strip()

                    formatted_query = query_format.format(query)
                    formatted_document = document_format.format(document)
                    full_text = (
                        formatted_query
                        + seq
                        + formatted_document
                        + special_token
                    )

                    sample_id = (
                        f"line={line_no};hit={hit_idx};trial={trial_id};"
                        f"fold={fold_id};split={split}"
                    )

                    yield {
                        "schema": "grouped",
                        "sample_id": sample_id,
                        "line_no": line_no,
                        "prompt_text": full_text,
                        "target_text": "",
                        "query_text": query,
                        "document_text": document,
                    }

            elif current_type == "pointwise":
                query = str(row.get("query", "")).strip()
                document = str(
                    row.get("content")
                    or row.get("document")
                    or row.get("text")
                    or ""
                ).strip()

                formatted_query = query_format.format(query)
                formatted_document = document_format.format(document)
                full_text = formatted_query + seq + formatted_document + special_token

                sample_id = str(
                    row.get("id")
                    or row.get("sample_id")
                    or row.get("qid")
                    or row.get("docid")
                    or f"line={line_no}"
                )

                yield {
                    "schema": "pointwise",
                    "sample_id": sample_id,
                    "line_no": line_no,
                    "prompt_text": full_text,
                    "target_text": "",
                    "query_text": query,
                    "document_text": document,
                }

            elif current_type == "prompt":
                prompt, target = get_prompt_and_target(row)

                sample_id = str(
                    row.get("id")
                    or row.get("sample_id")
                    or row.get("project_id")
                    or row.get("target_type")
                    or f"line={line_no}"
                )

                yield {
                    "schema": "prompt",
                    "sample_id": sample_id,
                    "line_no": line_no,
                    "prompt_text": prompt,
                    "target_text": target,
                    "query_text": "",
                    "document_text": "",
                }

            else:
                raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def token_count(tokenizer, text: str) -> int:
    if not text:
        return 0

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def analyze_split(
    *,
    split_name: str,
    jsonl_path: Optional[Path],
    dataset_type: str,
    tokenizer,
    max_len: int,
    query_format: str,
    document_format: str,
    seq: str,
    special_token: str,
    top_n: int,
    debug_print_text: bool,
) -> None:
    print(f"\n=== {split_name.upper()} ===")

    if jsonl_path is None:
        print("Missing file.")
        return

    if not jsonl_path.exists():
        print(f"Missing file: {jsonl_path}")
        return

    print(f"file: {jsonl_path}")
    print(f"dataset_type: {dataset_type}")
    print(f"configured training max_len: {max_len}")

    prompt_lengths: List[int] = []
    target_lengths: List[int] = []
    full_lengths: List[int] = []

    over_prompt_limit_count = 0
    over_full_limit_count = 0

    schema_counts: Dict[str, int] = {}
    seen_lines = set()
    longest_examples: List[Dict[str, Any]] = []

    for sample in tqdm.tqdm(
        iter_samples(
            jsonl_path=jsonl_path,
            dataset_type=dataset_type,
            query_format=query_format,
            document_format=document_format,
            seq=seq,
            special_token=special_token,
        ),
        desc=f"Analyzing {split_name}",
        unit="sample",
    ):
        seen_lines.add(sample["line_no"])

        schema = sample["schema"]
        schema_counts[schema] = schema_counts.get(schema, 0) + 1

        prompt_text = sample["prompt_text"]
        target_text = sample["target_text"]

        prompt_len = token_count(tokenizer, prompt_text)
        target_len = token_count(tokenizer, target_text)

        full_len = prompt_len + target_len

        prompt_lengths.append(prompt_len)
        target_lengths.append(target_len)
        full_lengths.append(full_len)

        if prompt_len > max_len:
            over_prompt_limit_count += 1

        if full_len > max_len:
            over_full_limit_count += 1

        longest_examples.append(
            {
                "sample_id": sample["sample_id"],
                "schema": schema,
                "line_no": sample["line_no"],
                "prompt_length": prompt_len,
                "target_length": target_len,
                "full_length": full_len,
                "prompt_text": prompt_text,
                "target_text": target_text,
                "query_text": sample.get("query_text", ""),
                "document_text": sample.get("document_text", ""),
            }
        )

    jsonl_record_count = len(seen_lines)
    sample_count = len(full_lengths)

    print(f"jsonl records: {jsonl_record_count}")
    print(f"number of analyzed samples: {sample_count}")
    print(f"detected schema counts: {schema_counts}")

    if not full_lengths:
        print("No samples found.")
        return

    def print_stats(name: str, values: List[int]) -> None:
        avg_len = sum(values) / len(values)
        median_len = statistics.median(values)
        min_len = min(values)
        max_observed_len = max(values)
        p90 = safe_percentile(values, 90)
        p95 = safe_percentile(values, 95)
        p99 = safe_percentile(values, 99)

        print(f"\n{name}:")
        print(f"  average: {avg_len:.2f}")
        print(f"  median: {median_len:.2f}")
        print(f"  min: {min_len}")
        print(f"  max: {max_observed_len}")
        print(f"  p90: {p90:.2f}")
        print(f"  p95: {p95:.2f}")
        print(f"  p99: {p99:.2f}")

    print_stats("Prompt/input token length before truncation", prompt_lengths)
    print_stats("Target/output token length before truncation", target_lengths)
    print_stats("Full sample token length before truncation", full_lengths)

    over_prompt_pct = 100.0 * over_prompt_limit_count / sample_count
    over_full_pct = 100.0 * over_full_limit_count / sample_count

    print(
        f"\nPrompt/input samples longer than max_len: "
        f"{over_prompt_limit_count} ({over_prompt_pct:.2f}%)"
    )
    print(
        f"Full prompt+target samples longer than max_len: "
        f"{over_full_limit_count} ({over_full_pct:.2f}%)"
    )

    longest_examples.sort(key=lambda x: x["full_length"], reverse=True)
    longest_examples = longest_examples[: max(0, top_n)]

    print(f"\nLongest {len(longest_examples)} samples by full prompt+target length:")

    for idx, item in enumerate(longest_examples, start=1):
        print(
            f"  {idx}. sample_id={item['sample_id']} | schema={item['schema']} "
            f"| line={item['line_no']} | prompt_tokens={item['prompt_length']} "
            f"| target_tokens={item['target_length']} | full_tokens={item['full_length']}"
        )

        if debug_print_text:
            print("     prompt/input text:")
            print(item["prompt_text"])
            if item["target_text"]:
                print("     target/output text:")
                print(item["target_text"])


def main() -> None:
    args = parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_yaml(args.config)

    model_name_or_path = get_value(
        cli_value=args.model_name_or_path,
        cfg=cfg,
        key="model_name_or_path",
        default=None,
    )

    if not model_name_or_path:
        raise ValueError(
            "model_name_or_path is required. Provide it with --model_name_or_path "
            "or include it in --config."
        )

    max_len = int(
        get_value(
            cli_value=args.max_len,
            cfg=cfg,
            key="max_len",
            default=512,
        )
    )

    query_format = str(
        get_value(
            cli_value=args.query_format,
            cfg=cfg,
            key="query_format",
            default="{}",
        )
    )

    document_format = str(
        get_value(
            cli_value=args.document_format,
            cfg=cfg,
            key="document_format",
            default="{}",
        )
    )

    seq = str(
        get_value(
            cli_value=args.seq,
            cfg=cfg,
            key="seq",
            default="",
        )
    )

    special_token = str(
        get_value(
            cli_value=args.special_token,
            cfg=cfg,
            key="special_token",
            default="",
        )
    )

    data_dir = Path(args.data_dir).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    if not data_dir.is_dir():
        raise NotADirectoryError(f"data_dir is not a folder: {data_dir}")

    tokenizer = build_tokenizer(str(model_name_or_path))
    model_config = maybe_load_model_config(str(model_name_or_path))
    limits = summarize_model_limits(tokenizer, model_config)

    print("=== MODEL / TOKENIZER LIMITS ===")
    print(f"model_name_or_path: {model_name_or_path}")
    print(f"configured training max_len: {max_len}")
    print(f"query_format: {query_format}")
    print(f"document_format: {document_format}")
    print(f"seq repr: {seq!r}")
    print(f"special_token repr: {special_token!r}")

    for key, value in limits.items():
        print(f"{key}: {value}")

    split_paths = {
        "train": find_split_file(data_dir, "train"),
        "valid": find_split_file(data_dir, "valid"),
        "test": find_split_file(data_dir, "test"),
    }

    print("\n=== DATASET FILES ===")
    for split_name, path in split_paths.items():
        print(f"{split_name}: {path}")

    for split_name, jsonl_path in split_paths.items():
        analyze_split(
            split_name=split_name,
            jsonl_path=jsonl_path,
            dataset_type=args.dataset_type,
            tokenizer=tokenizer,
            max_len=max_len,
            query_format=query_format,
            document_format=document_format,
            seq=seq,
            special_token=special_token,
            top_n=args.top_n,
            debug_print_text=args.debug_print_text,
        )


if __name__ == "__main__":
    main()