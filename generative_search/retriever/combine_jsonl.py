import argparse
import json
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine JSONL files without modifying records")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def validated_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL does not exist: {path}")
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path} at line {line_num}: {exc}") from exc
            lines.append(line)
    return lines


def combine_jsonl_files(inputs: List[Path], output: Path) -> dict:
    counts = []
    combined_lines = []
    for path in inputs:
        lines = validated_lines(path)
        counts.append({"input_path": str(path), "record_count": len(lines)})
        combined_lines.extend(lines)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for line in combined_lines:
            f.write(line if line.endswith("\n") else line + "\n")

    total_output_record_count = len(combined_lines)
    total_input_record_count = sum(item["record_count"] for item in counts)
    if total_output_record_count != total_input_record_count:
        raise AssertionError(
            "output_count mismatch: "
            f"output={total_output_record_count}, inputs={total_input_record_count}"
        )

    return {
        "valid_record_count": counts[0]["record_count"] if len(counts) >= 1 else 0,
        "test_record_count": counts[1]["record_count"] if len(counts) >= 2 else 0,
        "total_output_record_count": total_output_record_count,
        "output_path": str(output),
        "input_record_counts": counts,
    }


def main() -> None:
    args = parse_args()
    summary = combine_jsonl_files([Path(path) for path in args.inputs], Path(args.output))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
