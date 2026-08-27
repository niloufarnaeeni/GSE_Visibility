import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_yaml(
    path: str | Path,
) -> Dict[str, Any]:
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    return data if isinstance(data, dict) else {}


def _write_yaml(
    path: str | Path,
    data: Dict[str, Any],
) -> None:
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
        )


def _resolve_relative_path(
    path_value: Any,
    base_dir: Path,
) -> Any:
    if (
        not isinstance(path_value, str)
        or not path_value.strip()
    ):
        return path_value

    path = Path(path_value)

    if path.is_absolute():
        return path_value

    config_relative = base_dir / path

    if config_relative.exists():
        return str(config_relative)

    return path_value


def _validate_baseline_config(
    baseline_config: Dict[str, Any],
) -> Dict[str, Any]:
    method = baseline_config.get("method")

    if method != "boratto_reg":
        raise ValueError(
            "baseline method must be 'boratto_reg', "
            f"got {method!r}"
        )

    lambda_value = baseline_config.get(
        "lambda_corr",
        baseline_config.get("lambda"),
    )

    try:
        lambda_corr = float(lambda_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "baseline lambda_corr must be a finite "
            "number within [0, 1]"
        ) from exc

    if (
        not math.isfinite(lambda_corr)
        or not 0.0 <= lambda_corr <= 1.0
    ):
        raise ValueError(
            "baseline lambda_corr must be finite "
            "and within [0, 1], "
            f"got {lambda_corr}"
        )

    if "seed" in baseline_config:
        seed = baseline_config["seed"]

        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ValueError(
                "baseline seed must be an integer "
                f"when provided, got {seed!r}"
            )

    cleaned = dict(baseline_config)

    cleaned["lambda_corr"] = lambda_corr
    cleaned.setdefault(
        "display_name",
        "Boratto-reg",
    )
    return cleaned


def build_runtime_config(
    train_config: Dict[str, Any],
    baseline_config: Dict[str, Any],
    base_dir: str | Path | None = None,
) -> Dict[str, Any]:
    baseline_config = (
        _validate_baseline_config(
            baseline_config
        )
    )

    base_path = (
        Path(base_dir)
        if base_dir is not None
        else Path.cwd()
    )

    runtime = dict(train_config)

    if (
        str(
            runtime.get(
                "train_dataset_type",
                "grouped",
            )
        ).strip().lower()
        != "grouped"
    ):
        raise ValueError(
            "Boratto-reg requires "
            "train_dataset_type='grouped'."
        )

    for dataset_key in (
        "train_dataset",
        "val_dataset",
        "test_dataset",
    ):
        runtime[dataset_key] = (
            _resolve_relative_path(
                runtime.get(dataset_key),
                base_path,
            )
        )

    runtime["loss_type"] = "boratto_reg"

    runtime["baseline"] = {
        "method": "boratto_reg",
        "display_name": baseline_config.get(
            "display_name",
            "Boratto-reg",
        ),
        "lambda_corr": float(
            baseline_config["lambda_corr"]
        ),
    }

    if "seed" in baseline_config:
        runtime["seed"] = int(
            baseline_config["seed"]
        )

        runtime["baseline"]["seed"] = int(
            baseline_config["seed"]
        )

    return runtime


def _run_module(
    module: str,
    args: list[str],
) -> subprocess.CompletedProcess:
    env = dict(os.environ)

    package_root = str(
        Path(__file__).resolve().parents[3]
    )

    env["PYTHONPATH"] = (
        package_root
        if not env.get("PYTHONPATH")
        else package_root
        + os.pathsep
        + env["PYTHONPATH"]
    )

    return subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            *args,
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )


def _run_training_module(
    runtime_path: Path,
) -> subprocess.CompletedProcess:
    env = dict(os.environ)

    package_root = str(
        Path(__file__).resolve().parents[3]
    )

    env["PYTHONPATH"] = (
        package_root
        if not env.get("PYTHONPATH")
        else package_root
        + os.pathsep
        + env["PYTHONPATH"]
    )

    try:
        return subprocess.run(
            [
                "accelerate",
                "launch",
                "-m",
                (
                    "rag_retrieval.baseline."
                    "boratto_reg.train_reranker"
                ),
                "--config",
                str(runtime_path),
            ],
            text=True,
            capture_output=True,
            check=True,
            env=env,
        )

    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(
                exc.stdout,
                end="",
            )

        if exc.stderr:
            print(
                exc.stderr,
                end="",
                file=sys.stderr,
            )

        raise


def _extract_run_dir(
    stdout: str,
) -> Path:
    matches = re.findall(
        r"RUN_DIR::(.+)",
        stdout,
    )

    if not matches:
        raise RuntimeError(
            "Could not find RUN_DIR:: marker "
            "in trainer output"
        )

    return Path(
        matches[-1].strip()
    )


def _save_resolved_baseline(
    run_dir: Path,
    runtime_config: Dict[str, Any],
) -> None:
    output_path = (
        run_dir
        / "baseline_boratto_reg_config.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            runtime_config["baseline"],
            file,
            indent=2,
            ensure_ascii=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated Boratto-reg "
            "baseline train/eval."
        )
    )

    parser.add_argument(
        "--train-config",
        required=True,
    )

    parser.add_argument(
        "--baseline-config",
        required=True,
    )

    parser.add_argument(
        "--ks",
        default="2,5,10",
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--eval-max-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--device-map",
        default=None,
    )

    parser.add_argument(
        "--raw-data-dir",
        default=None,
    )

    parser.add_argument(
        "--skip-eval",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_config_path = Path(
        args.train_config
    )

    train_config = _load_yaml(
        train_config_path
    )

    baseline_config = _load_yaml(
        args.baseline_config
    )

    runtime_config = build_runtime_config(
        train_config=train_config,
        baseline_config=baseline_config,
        base_dir=train_config_path.parent,
    )

    with tempfile.TemporaryDirectory(
        prefix="boratto_reg_"
    ) as tmpdir:
        runtime_path = (
            Path(tmpdir)
            / "runtime.yaml"
        )

        _write_yaml(
            runtime_path,
            runtime_config,
        )

        print(
            "[Boratto-reg resolved config]"
        )

        print(
            json.dumps(
                runtime_config["baseline"],
                indent=2,
            )
        )

        train_proc = (
            _run_training_module(
                runtime_path
            )
        )

        print(
            train_proc.stdout,
            end="",
        )

        if train_proc.stderr:
            print(
                train_proc.stderr,
                end="",
                file=sys.stderr,
            )

        run_dir = _extract_run_dir(
            train_proc.stdout
        )

        _save_resolved_baseline(
            run_dir,
            runtime_config,
        )

        if args.skip_eval:
            return

        test_dataset = (
            runtime_config.get(
                "test_dataset"
            )
        )

        if not test_dataset:
            raise ValueError(
                "Runtime config must define "
                "test_dataset unless "
                "--skip-eval is used"
            )

        eval_args = [
            "--jsonl",
            str(test_dataset),
            "--model",
            str(run_dir / "model"),
            "--output_dir",
            str(run_dir),
            "--ks",
            args.ks,
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--eval_max_length",
            str(args.eval_max_length),
        ]

        if args.device_map:
            eval_args += [
                "--device_map",
                args.device_map,
            ]

        if args.raw_data_dir:
            eval_args += [
                "--raw-data-dir",
                args.raw_data_dir,
            ]

        eval_proc = _run_module(
            (
                "rag_retrieval.infer.eval."
                "evaluate_reranker"
            ),
            eval_args,
        )

        print(
            eval_proc.stdout,
            end="",
        )

        if eval_proc.stderr:
            print(
                eval_proc.stderr,
                end="",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
