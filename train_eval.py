import argparse
import os
import subprocess
import yaml
import sys
import tempfile
import json
from pathlib import Path
from itertools import product
from copy import deepcopy
from collections import deque


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_env_file(start_dir: Path | None = None) -> Path | None:
    search_dir = (start_dir or Path.cwd()).resolve()
    for candidate_dir in [search_dir, *search_dir.parents]:
        env_file = candidate_dir / ".env"
        if env_file.exists():
            return env_file
    return None


def load_dotenv_into_environ(env_file: Path):
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip("\"")

        if key and key not in os.environ:
            os.environ[key] = value


def configure_huggingface_auth():
    env_file = find_env_file()
    if env_file is not None:
        load_dotenv_into_environ(env_file)
        print(f"[ENV] Loaded {env_file}")

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        return

    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
    os.environ.setdefault("HUGGINGFACE_TOKEN", hf_token)
    os.environ.setdefault("HF_TOKEN", hf_token)
    print("[ENV] Hugging Face token detected and exported for subprocesses")


def save_yaml(cfg: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def infer_datasets_from_test_jsonl(test_jsonl: str, require_val: bool = True):
    """
    Given:
      data/<base>/processed/test.jsonl

    Returns:
      train, valid, test dataset paths
    """
    test_path = Path(test_jsonl).resolve()

    processed_dir = test_path.parent
    base_dir = processed_dir.parent

    train_dataset = processed_dir / "train.jsonl"
    val_dataset = processed_dir / "valid.jsonl"
    test_dataset = processed_dir / "test.jsonl"

    required_paths = [train_dataset, test_dataset]
    if require_val:
        required_paths.append(val_dataset)

    for p in required_paths:
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")

    return {
        "train_dataset": str(train_dataset),
        "val_dataset": str(val_dataset),
        "test_dataset": str(test_dataset),
        "base_folder": base_dir.name,
    }


def run_command_streaming(cmd, capture_run_dir=False, capture_last_checkpoint=False, tail_lines=200):
    """
    Stream stdout/stderr line by line.
    Good for Slurm because Slurm already saves logs.
    Avoids storing the full output in RAM.
    """
    print("\n" + "=" * 90)
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 90)

    tail_buffer = deque(maxlen=tail_lines)
    run_dir = None
    last_checkpoint = None

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")
        tail_buffer.append(line)

        if capture_run_dir and line.startswith("RUN_DIR::"):
            run_dir = Path(line.replace("RUN_DIR::", "").strip())
        if capture_last_checkpoint and line.startswith("LAST_CHECKPOINT::"):
            last_checkpoint = line.replace("LAST_CHECKPOINT::", "").strip()

    process.wait()

    if process.returncode != 0:
        tail_text = "".join(tail_buffer)
        error = RuntimeError(
            f"Command failed with exit code {process.returncode}.\n\n"
            f"Last checkpoint: {last_checkpoint}\n\n"
            f"Last {len(tail_buffer)} log lines:\n{tail_text}"
        )
        error.last_checkpoint = last_checkpoint
        error.run_dir = run_dir
        raise error

    return {
        "returncode": process.returncode,
        "run_dir": run_dir,
        "last_checkpoint": last_checkpoint,
    }


def infer_train_group_size(base_folder: str, mapping: dict) -> int:
    for key, group_size in mapping.items():
        if key in base_folder:
            return group_size
    raise ValueError(f"Cannot infer train_group_size from dataset folder: {base_folder}")


def parse_config_override_value(raw: str):
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null" or lowered == "none":
        return None

    try:
        if any(ch in raw for ch in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def apply_config_overrides(cfg: dict, override_items: list[str] | None) -> dict:
    if not override_items:
        return dict(cfg)

    updated = dict(cfg)
    for item in override_items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --set override: {item}. Expected key=value format."
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set override key in: {item}")
        updated[key] = parse_config_override_value(raw_value.strip())

    return updated


SUPPORTED_OBJECTIVES = {
    "ranknet": "RankNet",
    "ear": "EAR",
    "ear_sym": "EAR-Sym",
    "pairwise_reg": "Pairwise Reg",
}


def is_1p5b_model(model_name: str) -> bool:
    name = model_name.lower()
    return "1.5b" in name or "1_5b" in name


def apply_model_specific_overrides(cfg: dict, model_name: str) -> dict:
    return dict(cfg)


def build_experiment_settings(loss_type: str, args):
    if loss_type == "pairwise_reg":
        return [
            {
                "lambda_corr": float(v),
            }
            for v in deepcopy(args.lambda_corr)
        ]

    if loss_type in {"ear", "ear_sym"}:
        return [
            {
                "lambda_prior_attention": float(v),
            }
            for v in deepcopy(args.lambda_prior_attention)
        ]

    if loss_type == "ranknet":
        return [{}]

    raise ValueError(
        f"Unsupported objective '{loss_type}'. Supported objectives: "
        f"{', '.join(SUPPORTED_OBJECTIVES)}"
    )


def build_all_jobs(models, losses, args):
    jobs = []
    peft_methods = args.peft_methods or ["none"]
    for model_name, loss_type, peft_method in product(models, losses, peft_methods):
        experiment_settings = build_experiment_settings(loss_type, args)
        for exp_cfg in experiment_settings:
            jobs.append(
                {
                    "model_name": model_name,
                    "loss_type": loss_type,
                    "peft_method": peft_method,
                    "lora_target_modules": args.lora_target_modules,
                    "exp_cfg": exp_cfg,
                }
            )
    return jobs


def print_job_header(job_idx: int, total_jobs: int, model_name: str, loss_type: str, peft_method: str, exp_cfg: dict):
    done_before = job_idx - 1
    remaining_after_this_starts = total_jobs - job_idx

    print("\n" + "#" * 100)
    print(f"EXPERIMENT {job_idx}/{total_jobs}")
    print(f"DONE SO FAR: {done_before}")
    print(f"LEFT AFTER THIS ONE STARTS: {remaining_after_this_starts}")
    print(f"MODEL={model_name}")
    print(f"LOSS={loss_type} ({SUPPORTED_OBJECTIVES[loss_type]})")
    print(f"PEFT_METHOD={peft_method}")
    if loss_type == "pairwise_reg":
        print(f"LAMBDA_CORR={exp_cfg['lambda_corr']}")
    elif loss_type in {"ear", "ear_sym"}:
        print(f"LAMBDA_PRIOR_ATTENTION={exp_cfg['lambda_prior_attention']}")

    print("#" * 100)


# -------------------------------------------------
# Resume helpers
# -------------------------------------------------

def make_job_key(
    model_name: str,
    loss_type: str,
    peft_method: str,
    exp_cfg: dict,
    lora_target_modules: str | None = None,
) -> str:
    return "||".join([
        f"model={model_name}",
        f"loss={loss_type}",
        f"peft_method={peft_method}",
        f"lora_target_modules={lora_target_modules or 'default'}",
        f"lambda_prior_attention={float(exp_cfg.get('lambda_prior_attention', 0.0))}",
        f"lambda_corr={float(exp_cfg.get('lambda_corr', 0.0))}",
    ])


def get_sweep_state_paths(output_dir: Path):
    sweep_state_dir = output_dir / "sweep_state"
    status_path = sweep_state_dir / "status.json"
    return sweep_state_dir, status_path


def load_status(status_path: Path) -> dict:
    if not status_path.exists():
        return {
            "completed": {},
            "failed": {},
            "running": {},
        }

    with open(status_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data.setdefault("completed", {})
    data.setdefault("failed", {})
    data.setdefault("running", {})
    return data


def save_status(status_path: Path, status: dict):
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)


def mark_job_running(status: dict, job_key: str, payload: dict):
    status["running"][job_key] = payload
    status["failed"].pop(job_key, None)


def mark_job_completed(status: dict, job_key: str, payload: dict):
    status["completed"][job_key] = payload
    status["running"].pop(job_key, None)
    status["failed"].pop(job_key, None)


def mark_job_failed(status: dict, job_key: str, payload: dict):
    # Keep only the most recent failed job.
    # This prevents an old failed job from accidentally receiving
    # a manually provided checkpoint override in a later resume run.
    status["failed"] = {job_key: payload}
    status["running"].pop(job_key, None)


def count_completed_jobs(jobs, status: dict) -> int:
    count = 0
    for job in jobs:
        key = make_job_key(
            job["model_name"],
            job["loss_type"],
            job["peft_method"],
            job["exp_cfg"],
            job.get("lora_target_modules"),
        )
        if key in status["completed"]:
            count += 1
    return count


def get_resume_checkpoint_for_failed_job(
    status: dict,
    job_key: str,
    checkpoint_override: str | None,
) -> str | None:
    failed_payload = status.get("failed", {}).get(job_key)
    if not failed_payload:
        return None

    if checkpoint_override:
        print(f"Using checkpoint override: {checkpoint_override}")
        return checkpoint_override

    last_checkpoint = failed_payload.get("last_checkpoint")
    if last_checkpoint:
        print(f"Using stored resume checkpoint: {last_checkpoint}")
        return str(last_checkpoint)

    return None


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_config", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--output_dir", default="output/kaito")
    parser.add_argument("--ks", default="2,5,10")
    parser.add_argument("--raw-data-dir", default=None)

    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--losses", nargs="*", default=None)

    parser.add_argument(
        "--lambda_prior_attention",
        nargs="*",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="EAR and EAR-Sym sweep values.",
    )
    parser.add_argument(
        "--lambda_corr",
        nargs="*",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="Pairwise Reg sweep values.",
    )
    parser.add_argument(
        "--peft_methods",
        nargs="*",
        default=["none"],
        help='choose from [none, lora, qlora]',
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help='Optional comma-separated LoRA target modules, e.g. "q_proj,k_proj,v_proj,o_proj"',
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        nargs="*",
        default=None,
        help="Override base config values with key=value pairs, e.g. --set max_len=256 batch_size=4",
    )
    parser.add_argument(
        "--accelerate_num_processes",
        type=int,
        default=1,
        help="Number of processes for accelerate launch. Keep 1 for single-GPU.",
    )
    parser.add_argument(
        "--accelerate_config_file",
        type=str,
        default=None,
        help="Optional accelerate/DeepSpeed config file passed to accelerate launch.",
    )
    parser.add_argument(
        "--eval_device_map",
        type=str,
        default=None,
        help='Optional evaluation device_map. Use "auto" to shard eval across visible GPUs.',
    )
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_max_length", type=int, default=256)
    parser.add_argument(
        "--checkpoint_override",
        type=str,
        default=None,
        help="Optional checkpoint path to force when resuming failed runs from sweep_state/status.json.",
    )

    args = parser.parse_args()
    configure_huggingface_auth()

    models = args.models or [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "BAAI/bge-reranker-base",
        "cross-encoder/ms-marco-electra-base",
        "cross-encoder/ms-marco-TinyBERT-L2-v2",
    ]

    losses = args.losses or [
        "ranknet",
        "ear",
        "ear_sym",
        "pairwise_reg",
    ]

    invalid_losses = [loss for loss in losses if loss not in SUPPORTED_OBJECTIVES]
    if invalid_losses:
        raise ValueError(
            f"Unsupported objectives requested: {invalid_losses}. "
            f"Supported objectives: {list(SUPPORTED_OBJECTIVES)}"
        )

    DATASET_TRAIN_GROUP_SIZE = {
        "giverep": 4,
        "kaito": 8,
        "cookie_fun": 8,
    }

    base_cfg = apply_config_overrides(load_yaml(Path(args.base_config)), args.config_overrides)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_state_dir, status_path = get_sweep_state_paths(output_dir)
    status = load_status(status_path)

    disable_validation = bool(base_cfg.get("disable_validation", False))
    dataset_info = infer_datasets_from_test_jsonl(
        args.test_jsonl,
        require_val=not disable_validation,
    )
    print(f"[DATASET] base_folder = {dataset_info['base_folder']}")

    explicit_train_group_size = base_cfg.get("train_group_size", None)

    if explicit_train_group_size is not None:
        train_group_size = int(explicit_train_group_size)
        print(
            f"[INFO] Using explicit train_group_size={train_group_size} "
            f"from config or --set override"
        )
    else:
        train_group_size = infer_train_group_size(
            dataset_info["base_folder"],
            DATASET_TRAIN_GROUP_SIZE,
        )
        print(
            f"[INFO] Inferred train_group_size={train_group_size} "
            f"for dataset={dataset_info['base_folder']}"
        )

    jobs = build_all_jobs(models, losses, args)
    total_jobs = len(jobs)

    completed_before_start = count_completed_jobs(jobs, status)

    print("\n" + "=" * 100)
    print(f"TOTAL EXPERIMENTS TO RUN: {total_jobs}")
    print(f"MODELS: {len(models)}")
    print(f"LOSSES: {len(losses)}")
    print(f"ALREADY COMPLETED: {completed_before_start}")
    print(f"SWEEP STATE FILE: {status_path}")
    print("=" * 100)

    completed_jobs = completed_before_start

    for job_idx, job in enumerate(jobs, start=1):
        model_name = job["model_name"]
        loss_type = job["loss_type"]
        peft_method = job["peft_method"]
        exp_cfg = job["exp_cfg"]

        job_key = make_job_key(
            model_name,
            loss_type,
            peft_method,
            exp_cfg,
            args.lora_target_modules,
        )

        if job_key in status["completed"]:
            print("\n" + "-" * 100)
            print(f"SKIPPING COMPLETED EXPERIMENT {job_idx}/{total_jobs}")
            print(f"MODEL={model_name}")
            print(f"LOSS={loss_type}")
            print(f"PEFT_METHOD={peft_method}")
            print(f"JOB_KEY={job_key}")
            print("-" * 100)
            continue

        print_job_header(job_idx, total_jobs, model_name, loss_type, peft_method, exp_cfg)

        resume_checkpoint = get_resume_checkpoint_for_failed_job(
            status=status,
            job_key=job_key,
            checkpoint_override=args.checkpoint_override,
        )

        cfg = dict(base_cfg)
        cfg["train_dataset"] = dataset_info["train_dataset"]
        cfg["val_dataset"] = None if disable_validation else dataset_info["val_dataset"]
        cfg["test_dataset"] = dataset_info["test_dataset"]
        cfg["output_dir"] = str(output_dir)
        cfg["model_name_or_path"] = model_name
        cfg["loss_type"] = loss_type
        cfg["peft_method"] = peft_method
        if args.lora_target_modules is not None:
            cfg["lora_target_modules"] = args.lora_target_modules
        if cfg.get("model_type") == "llm_decoder" and peft_method == "qlora":
            if "gradient_checkpointing" not in cfg or cfg.get("gradient_checkpointing") is None:
                cfg["gradient_checkpointing"] = True
        print(
            f"[CONFIG] Final gradient_checkpointing={cfg.get('gradient_checkpointing')} "
            f"for model_type={cfg.get('model_type')} peft_method={peft_method}"
        )
        cfg["train_group_size"] = train_group_size
        cfg["save_on_epoch_end"] = 1

        if loss_type in {"ear", "ear_sym"}:
            cfg["lambda_prior_attention"] = float(exp_cfg["lambda_prior_attention"])
            cfg["lambda_corr"] = float(cfg.get("lambda_corr", 0.05))
        elif loss_type == "pairwise_reg":
            cfg["lambda_corr"] = float(exp_cfg["lambda_corr"])
            cfg["lambda_prior_attention"] = float(cfg.get("lambda_prior_attention", 0.0))
        else:
            cfg["lambda_prior_attention"] = float(cfg.get("lambda_prior_attention", 0.0))
            cfg["lambda_corr"] = float(cfg.get("lambda_corr", 0.05))

        cfg = apply_model_specific_overrides(cfg, model_name)

        running_payload = {
            "job_idx": job_idx,
            "total_jobs": total_jobs,
            "model_name": model_name,
            "loss_type": loss_type,
            "peft_method": peft_method,
            "lora_target_modules": args.lora_target_modules,
            "exp_cfg": exp_cfg,
        }
        mark_job_running(status, job_key, running_payload)
        save_status(status_path, status)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            save_yaml(cfg, tmp_path)

        print("\nUSING CONFIG:")
        print(yaml.safe_dump(cfg, sort_keys=False))

        train_cmd = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
        ]
        if args.accelerate_config_file:
            train_cmd.extend(["--config_file", args.accelerate_config_file])
        train_cmd.extend(
            [
                f"--num_processes={int(args.accelerate_num_processes)}",
                "-m",
                "rag_retrieval.train.reranker.train_reranker",
                "--config",
                str(tmp_path),
            ]
        )
        if resume_checkpoint:
            train_cmd.extend(["--last_checkpoint", str(resume_checkpoint)])

        train_result = None

        try:
            train_result = run_command_streaming(
                train_cmd,
                capture_run_dir=True,
                capture_last_checkpoint=True,
            )
            run_dir = train_result["run_dir"]

            if run_dir is None:
                raise RuntimeError("RUN_DIR not found in training output.")

            model_dir = run_dir / "model"

            if not model_dir.exists():
                raise RuntimeError(f"Model dir missing: {model_dir}")

            eval_base = [
                sys.executable,
                "-m",
                "rag_retrieval.infer.eval.evaluate_reranker",
                "--jsonl",
                args.test_jsonl,
                "--model",
                str(model_dir),
                "--ks",
                args.ks,
                "--output_dir",
                str(output_dir),
                "--eval_batch_size",
                str(args.eval_batch_size),
                "--eval_max_length",
                str(args.eval_max_length),
            ]
            if args.raw_data_dir:
                eval_base.extend(["--raw-data-dir", args.raw_data_dir])
            if args.eval_device_map:
                eval_base.extend(["--device_map", args.eval_device_map])

            eval_failed = False
            eval_error = None

            try:
                run_command_streaming(eval_base, capture_run_dir=False)

            except Exception as e:
                eval_failed = True
                eval_error = str(e)
                print("\n" + "!" * 100)
                print(f"[WARN] Evaluation failed for experiment {job_idx}/{total_jobs}")
                print(f"[WARN] MODEL={model_name}")
                print(f"[WARN] LOSS={loss_type}")
                print(f"[WARN] PEFT_METHOD={peft_method}")
                print("[WARN] Training finished, but evaluation failed. Continuing to next training job.")
                print(f"[WARN] Evaluation error: {eval_error}")
                print("!" * 100)

            completed_jobs += 1
            remaining_jobs = total_jobs - completed_jobs

            completed_payload = {
                "job_idx": job_idx,
                "total_jobs": total_jobs,
                "model_name": model_name,
                "loss_type": loss_type,
                "peft_method": peft_method,
                "lora_target_modules": args.lora_target_modules,
                "exp_cfg": exp_cfg,
                "run_dir": str(run_dir),
                "model_dir": str(model_dir),
                "eval_failed": eval_failed,
                "eval_error": eval_error,
            }
            mark_job_completed(status, job_key, completed_payload)
            save_status(status_path, status)

            print("\n" + "-" * 100)
            print(f"FINISHED EXPERIMENT {job_idx}/{total_jobs}")
            print(f"COMPLETED: {completed_jobs}")
            print(f"REMAINING: {remaining_jobs}")
            print("-" * 100)

        except Exception as e:
            failed_payload = {
                "job_idx": job_idx,
                "total_jobs": total_jobs,
                "model_name": model_name,
                "loss_type": loss_type,
                "peft_method": peft_method,
                "lora_target_modules": args.lora_target_modules,
                "exp_cfg": exp_cfg,
                "error": str(e),
                "last_checkpoint": (
                    train_result.get("last_checkpoint")
                    if isinstance(train_result, dict) and train_result.get("last_checkpoint")
                    else getattr(e, "last_checkpoint", None) or resume_checkpoint
                ),
            }
            mark_job_failed(status, job_key, failed_payload)
            save_status(status_path, status)
            raise

        finally:
            tmp_path.unlink(missing_ok=True)

    print("\nALL EXPERIMENTS FINISHED SUCCESSFULLY")
    print(f"Final status file: {status_path}")


if __name__ == "__main__":
    main()
