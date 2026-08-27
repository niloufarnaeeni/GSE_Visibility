import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from accelerate.utils import set_seed, ProjectConfiguration
from transformers import get_cosine_schedule_with_warmup, get_wsd_schedule
from torch.utils.data import DataLoader
from accelerate import Accelerator

try:
    from peft import set_peft_model_state_dict
except ImportError:
    set_peft_model_state_dict = None

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:
    load_safetensors_file = None

# FIXED IMPORTS (absolute, package-rooted)
from rag_retrieval.train.reranker.model_bert import CrossEncoder
from rag_retrieval.train.reranker.model_llm import LLMDecoder
from rag_retrieval.train.reranker.data import (
    PointwiseRankerDataset,
    GroupedRankerDataset,
    GroupedRankerDatasetWithPriorAttention,
)
from rag_retrieval.train.reranker.trainer import (
    Trainer,
    prepare_clean_dir,
    should_stage_to_local,
    build_local_stage_dir,
    save_unwrapped_model_pretrained,
    verify_saved_artifact_dir,
    log_save_stats,
    copy_directory_contents,
)

import gc
import psutil

TORONTO_TZ = ZoneInfo("America/Toronto")


def create_adamw_optimizer(
    model,
    lr,
    weight_decay=1e-2,
    no_decay_keywords=("bias", "LayerNorm", "layernorm"),
):
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in parameters if not any(nd in n for nd in no_decay_keywords)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in parameters if any(nd in n for nd in no_decay_keywords)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def save_run_config(model_dir: Path, cfg: dict, run_id: str) -> None:
    cfg = dict(cfg)
    cfg["run_time"] = run_id

    out_path = model_dir / "run_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def load_trainer_resume_state(checkpoint_dir: str | Path) -> dict:
    trainer_state_path = Path(checkpoint_dir) / "trainer_state.json"
    if not trainer_state_path.exists():
        return {}

    with open(trainer_state_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {}

    return data


def make_tracker_safe_config(config: dict) -> dict:
    safe_config = {}
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool)):
            safe_config[key] = value
        elif value is None:
            safe_config[key] = "None"
        else:
            safe_config[key] = json.dumps(value, sort_keys=True, default=str)
    return safe_config


def _load_standard_state_dict(path: Path) -> dict:
    if path.suffix == ".safetensors":
        if load_safetensors_file is None:
            raise ImportError(
                f"safetensors is required to load checkpoint file: {path}"
            )
        return load_safetensors_file(str(path))
    return torch.load(path, map_location="cpu")


def _find_model_state_file(checkpoint_dir: Path) -> Path | None:
    for name in ["model.safetensors", "pytorch_model.bin"]:
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate
    return None


def _infer_state_dict_format_from_file(state_path: Path | None) -> str | None:
    if state_path is None or not state_path.exists():
        return None

    state_dict = _load_standard_state_dict(state_path)
    if not isinstance(state_dict, dict) or not state_dict:
        return None

    sample_keys = list(state_dict.keys())
    if any(str(key).startswith("model.") for key in sample_keys):
        return "wrapper_prefixed"
    return "hf_inner"


def detect_checkpoint_format(checkpoint_dir: str | Path) -> str:
    checkpoint_dir = Path(checkpoint_dir)

    peft_markers = [
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]

    if any((checkpoint_dir / name).exists() for name in peft_markers):
        return "peft_adapter"

    root_state_path = _find_model_state_file(checkpoint_dir)
    root_state_format = _infer_state_dict_format_from_file(root_state_path)

    if root_state_format == "wrapper_prefixed":
        return "accelerate_full_state"

    if root_state_format == "hf_inner":
        return "crossencoder_hf_inner_model_root"

    hf_model_dir = checkpoint_dir / "hf_model"
    hf_subdir_state_path = _find_model_state_file(hf_model_dir)
    hf_subdir_state_format = _infer_state_dict_format_from_file(hf_subdir_state_path)
    if hf_subdir_state_format == "hf_inner":
        return "crossencoder_hf_inner_model_subdir"

    return "unknown"


def _load_peft_adapter_state_dict(checkpoint_dir: Path) -> dict:
    adapter_safetensors = checkpoint_dir / "adapter_model.safetensors"
    adapter_bin = checkpoint_dir / "adapter_model.bin"

    if adapter_safetensors.exists():
        if load_safetensors_file is None:
            raise ImportError(
                "safetensors is required to load adapter_model.safetensors checkpoints."
            )
        return load_safetensors_file(str(adapter_safetensors))

    if adapter_bin.exists():
        return torch.load(adapter_bin, map_location="cpu")

    raise FileNotFoundError(
        f"No PEFT adapter weights found in checkpoint: {checkpoint_dir}"
    )


def load_peft_resume_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    optimizer,
    lr_scheduler,
    checkpoint_dir: str | Path,
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    unwrapped_model = accelerator.unwrap_model(model)
    target_model = getattr(unwrapped_model, "model", unwrapped_model)

    if not hasattr(target_model, "peft_config"):
        raise ValueError(
            "PEFT adapter checkpoint detected, but the current training model is not a PEFT model."
        )
    if set_peft_model_state_dict is None:
        raise ImportError(
            "peft is required to resume adapter checkpoints, but it is not installed."
        )

    adapter_state_dict = _load_peft_adapter_state_dict(checkpoint_dir)
    try:
        set_peft_model_state_dict(target_model, adapter_state_dict, adapter_name="default")
    except TypeError:
        set_peft_model_state_dict(target_model, adapter_state_dict)

    optimizer_path = checkpoint_dir / "optimizer.bin"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        accelerator.print(f"[CHECKPOINT] Loaded optimizer state from {optimizer_path}")

    scheduler_path = checkpoint_dir / "scheduler.bin"
    if scheduler_path.exists():
        lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        accelerator.print(f"[CHECKPOINT] Loaded scheduler state from {scheduler_path}")

    resume_state = load_trainer_resume_state(checkpoint_dir)
    if resume_state:
        accelerator.print(f"[CHECKPOINT] Loaded trainer_state: {resume_state}")

    accelerator.print(
        f"[CHECKPOINT] Loaded PEFT adapter checkpoint from {checkpoint_dir}"
    )
    return resume_state


def load_crossencoder_hf_resume_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    optimizer,
    lr_scheduler,
    checkpoint_dir: str | Path,
    model_subdir: str | None = None,
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    model_dir = checkpoint_dir / model_subdir if model_subdir else checkpoint_dir

    unwrapped_model = accelerator.unwrap_model(model)
    target_model = getattr(unwrapped_model, "model", None)
    if target_model is None:
        raise ValueError(
            "CrossEncoder HF checkpoint detected, but the current training model does not expose an inner `.model`."
        )

    state_path = _find_model_state_file(model_dir)
    if state_path is None:
        raise FileNotFoundError(
            f"No HF inner-model state file found in checkpoint: {model_dir}"
        )

    state_dict = _load_standard_state_dict(state_path)
    load_result = target_model.load_state_dict(state_dict, strict=True)
    accelerator.print(
        f"[CHECKPOINT] Loaded CrossEncoder inner HF model state from {state_path} "
        f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
    )

    optimizer_path = checkpoint_dir / "optimizer.bin"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        accelerator.print(f"[CHECKPOINT] Loaded optimizer state from {optimizer_path}")

    scheduler_path = checkpoint_dir / "scheduler.bin"
    if scheduler_path.exists():
        lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        accelerator.print(f"[CHECKPOINT] Loaded scheduler state from {scheduler_path}")

    resume_state = load_trainer_resume_state(checkpoint_dir)
    if resume_state:
        accelerator.print(f"[CHECKPOINT] Loaded trainer_state: {resume_state}")

    return resume_state


def parse_args():
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--model_name_or_path", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument(
        "--model_type",
        type=str,
        help="choose from [bert_encoder,llm_decoder]",
    )
    parser.add_argument("--train_dataset", help="training file")
    parser.add_argument("--train_dataset_type", help="the type of training file", default="pointwise")
    parser.add_argument("--train_group_size", type=int, default=8)
    parser.add_argument("--train_label_key", help="label key of training", default="label")

    parser.add_argument("--val_dataset", help="validation file", default=None)
    parser.add_argument("--val_dataset_type", help="the type of validation file", default="pointwise")
    parser.add_argument("--val_label_key", help="label key of validation", default="label")
    parser.add_argument("--shuffle_rate", type=float, default=0.0)
    parser.add_argument("--output_dir", help="output dir", default="./output")
    parser.add_argument("--save_on_epoch_end", type=int, default=0)
    parser.add_argument("--num_max_checkpoints", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--save_every_n_batches", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--stable_proportion", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="ranknet",
        choices=["ranknet", "ear", "ear_sym", "pairwise_reg"],
        help="Ranking objective: ranknet (RankNet), ear (EAR), ear_sym (EAR-Sym), or pairwise_reg (Pairwise Reg).",
    )
    parser.add_argument(
        "--log_with", type=str, default="wandb", help="wandb, tensorboard"
    )
    parser.add_argument("--mixed_precision", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_labels", type=int, default=1, help="mlp dim")
    parser.add_argument("--query_format", type=str, default="{}")
    parser.add_argument("--document_format", type=str, default="{}")
    parser.add_argument("--seq", type=str, default="")
    parser.add_argument("--special_token", type=str, default="")
    parser.add_argument("--max_label", type=int, default=1)
    parser.add_argument("--min_label", type=int, default=0)
    parser.add_argument("--lambda_corr", type=float, default=0.05, help="Pairwise Reg correlation weight.")
    parser.add_argument("--lambda_prior_attention", type=float, default=0.0, help="EAR/EAR-Sym prior-attention weight.")
    parser.add_argument("--peft_method", type=str, default="none", help="choose from [none, lora, qlora]")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--gradient_checkpointing", type=bool, default=False)
    parser.add_argument("--qlora_compute_dtype", type=str, default="bfloat16")
    parser.add_argument("--qlora_quant_type", type=str, default="nf4")
    parser.add_argument("--qlora_use_double_quant", type=bool, default=True)
    parser.add_argument("--last_checkpoint", type=str, default=None)
    parser.add_argument("--disable_validation", action="store_true")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for key, value in config.items():
        setattr(args, key, value)

    save_every_n_batches = getattr(args, "save_every_n_batches", 0)
    save_steps = getattr(args, "save_steps", 0)
    if save_every_n_batches in (None, 0):
        save_every_n_batches = save_steps
    args.save_every_n_batches = int(save_every_n_batches or 0)
    args.save_steps = int(save_steps or 0)

    args.disable_validation = bool(getattr(args, "disable_validation", False))
    if getattr(args, "val_dataset", None) is not None and not str(args.val_dataset).strip():
        args.val_dataset = None
    if args.disable_validation:
        args.val_dataset = None

    return args


def print_system_memory(tag: str, accelerator=None):
    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / 1024**3
    vms_gb = process.memory_info().vms / 1024**3

    prefix = f"[{tag}]"
    msg = f"{prefix} RAM rss={rss_gb:.2f} GB | vms={vms_gb:.2f} GB"

    if accelerator is not None:
        accelerator.print(msg)
    else:
        print(msg)


def print_gpu_memory(tag: str, accelerator=None):
    prefix = f"[{tag}]"

    if not torch.cuda.is_available():
        msg = f"{prefix} CUDA not available"
        if accelerator is not None:
            accelerator.print(msg)
        else:
            print(msg)
        return

    lines = [f"{prefix} GPU memory"]
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        free, total = torch.cuda.mem_get_info(i)
        free_gb = free / 1024**3
        total_gb = total / 1024**3
        lines.append(
            f"  cuda:{i} | allocated={allocated:.2f} GB | reserved={reserved:.2f} GB | free={free_gb:.2f} GB / total={total_gb:.2f} GB"
        )

    msg = "\n".join(lines)
    if accelerator is not None:
        accelerator.print(msg)
    else:
        print(msg)


def print_memory_snapshot(tag: str, accelerator=None):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print_system_memory(tag, accelerator=accelerator)
    print_gpu_memory(tag, accelerator=accelerator)


def main():
    args = parse_args()

    set_seed(args.seed)

    run_time = datetime.now(TORONTO_TZ)
    run_id = run_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.output_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(run_dir)

    project_config = ProjectConfiguration(
        project_dir=str(args.output_dir) + "/runs",
        automatic_checkpoint_naming=True,
        total_limit=args.num_max_checkpoints,
        logging_dir=str(args.output_dir),
    )

    accelerator = Accelerator(
        project_config=project_config,
        log_with=args.log_with,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    accelerator.init_trackers("ranker", config=make_tracker_safe_config(vars(args)))
    accelerator.print(f"Train Args from User Input: {vars(args)}")
    accelerator.print(
        "[CHECKPOINT] Final save_every_n_batches="
        f"{args.save_every_n_batches} (legacy save_steps={args.save_steps})"
    )
    accelerator.print(f"Requested mixed_precision arg = {args.mixed_precision}")
    accelerator.print(f"Actual accelerator mixed_precision = {accelerator.mixed_precision}")
    print_memory_snapshot("after_accelerator_init", accelerator)

    if args.model_type == "bert_encoder":
        model = CrossEncoder.from_pretrained(
            model_name_or_path=args.model_name_or_path,
            loss_type=args.loss_type,
            num_labels=args.num_labels,
            query_format=args.query_format,
            document_format=args.document_format,
        )
    elif args.model_type == "llm_decoder":
        model = LLMDecoder.from_pretrained(
            model_name_or_path=args.model_name_or_path,
            loss_type=args.loss_type,
            num_labels=args.num_labels,
            query_format=args.query_format,
            document_format=args.document_format,
            seq=args.seq,
            special_token=args.special_token,
            peft_method=getattr(args, "peft_method", "none"),
            lora_r=getattr(args, "lora_r", 16),
            lora_alpha=getattr(args, "lora_alpha", 32),
            lora_dropout=getattr(args, "lora_dropout", 0.05),
            lora_target_modules=getattr(args, "lora_target_modules", None),
            gradient_checkpointing=bool(getattr(args, "gradient_checkpointing", False)),
            qlora_compute_dtype=getattr(args, "qlora_compute_dtype", "bfloat16"),
            qlora_quant_type=getattr(args, "qlora_quant_type", "nf4"),
            qlora_use_double_quant=bool(getattr(args, "qlora_use_double_quant", True)),
        )
    else:
        raise ValueError("Model type not currently supported")

    print_memory_snapshot("after_model_init", accelerator)

    if "pointwise" == args.train_dataset_type:
        train_dataset = PointwiseRankerDataset(
            data_path=args.train_dataset,
            label_key=args.train_label_key,
            target_model=model,
            max_len=args.max_len,
            max_label=args.max_label,
            min_label=args.min_label,
            shuffle_rate=args.shuffle_rate,
            tag="training",
        )
        model.train_group_size = 1

    elif "grouped" == args.train_dataset_type and args.loss_type in {"ear", "ear_sym", "pairwise_reg"}:
        train_dataset = GroupedRankerDatasetWithPriorAttention(
            data_path=args.train_dataset,
            label_key=args.train_label_key,
            target_model=model,
            max_len=args.max_len,
            shuffle_rate=args.shuffle_rate,
            train_group_size=args.train_group_size,
            tag="training",
        )
        model.train_group_size = args.train_group_size
        model.lambda_prior_attention = float(args.lambda_prior_attention)
        model.lambda_corr = float(args.lambda_corr)

    elif "grouped" == args.train_dataset_type:
        train_dataset = GroupedRankerDataset(
            data_path=args.train_dataset,
            label_key=args.train_label_key,
            target_model=model,
            max_len=args.max_len,
            shuffle_rate=args.shuffle_rate,
            train_group_size=args.train_group_size,
            tag="training",
        )
        model.train_group_size = args.train_group_size

    else:
        raise ValueError(f"Train dataset type {args.train_dataset_type} not currently supported")

    val_dataset = None
    if args.val_dataset:
        if "pointwise" == args.val_dataset_type:
            val_dataset = PointwiseRankerDataset(
                data_path=args.val_dataset,
                label_key=args.val_label_key,
                target_model=model,
                max_len=args.max_len,
                max_label=args.max_label,
                min_label=args.min_label,
                shuffle_rate=args.shuffle_rate,
                tag="validation",
            )
        elif "grouped" == args.val_dataset_type and args.loss_type in {"ear", "ear_sym", "pairwise_reg"}:
            val_dataset = GroupedRankerDatasetWithPriorAttention(
                data_path=args.val_dataset,
                label_key=args.val_label_key,
                target_model=model,
                max_len=args.max_len,
                shuffle_rate=args.shuffle_rate,
                train_group_size=args.train_group_size,
                tag="validation",
            )
        elif "grouped" == args.val_dataset_type:
            val_dataset = GroupedRankerDataset(
                data_path=args.val_dataset,
                label_key=args.val_label_key,
                target_model=model,
                max_len=args.max_len,
                shuffle_rate=args.shuffle_rate,
                train_group_size=args.train_group_size,
                tag="validation",
            )
        else:
            raise ValueError(f"Val dataset type {args.val_dataset_type} not supported")
    else:
        accelerator.print("[VALIDATION] Disabled during training")

    print_memory_snapshot("after_dataset_build", accelerator)

    num_workers = min(8, os.cpu_count() or 1)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_dataloader = None
    if args.val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=val_dataset.collate_fn,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    print_memory_snapshot("after_dataloader_build", accelerator)

    optimizer = create_adamw_optimizer(model, lr=float(args.lr))
    assert 0 <= args.warmup_proportion <= 1
    assert 0 <= args.stable_proportion <= 1
    assert args.warmup_proportion + args.stable_proportion <= 1

    total_steps = (
        len(train_dataloader) * args.epochs
    ) // accelerator.gradient_state.num_steps
    num_warmup_steps = int(args.warmup_proportion * total_steps)
    num_stable_steps = int(args.stable_proportion * total_steps)

    lr_scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=total_steps - num_warmup_steps - num_stable_steps,
    )

    if val_dataloader is not None:
        model, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader, val_dataloader
        )
    else:
        model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader
        )

    print_memory_snapshot("after_accelerator_prepare", accelerator)

    resume_state = {}
    if args.last_checkpoint:
        print(f"[CHECKPOINT] Resuming from checkpoint: {args.last_checkpoint}")
        checkpoint_format = detect_checkpoint_format(args.last_checkpoint)
        accelerator.print(f"[CHECKPOINT] Detected checkpoint format: {checkpoint_format}")

        if checkpoint_format == "accelerate_full_state":
            accelerator.print(
                "[CHECKPOINT] Loading Accelerate full-state checkpoint from root directory."
            )
            accelerator.load_state(args.last_checkpoint)
            resume_state = load_trainer_resume_state(args.last_checkpoint)
            if resume_state:
                accelerator.print(f"[CHECKPOINT] Loaded trainer_state: {resume_state}")
        elif checkpoint_format == "peft_adapter":
            resume_state = load_peft_resume_checkpoint(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                checkpoint_dir=args.last_checkpoint,
            )
        elif checkpoint_format == "crossencoder_hf_inner_model_root":
            accelerator.print(
                "[CHECKPOINT] Root checkpoint contains HF inner-model keys. "
                "Loading into CrossEncoder.model instead of accelerator.load_state()."
            )
            resume_state = load_crossencoder_hf_resume_checkpoint(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                checkpoint_dir=args.last_checkpoint,
            )
        elif checkpoint_format == "crossencoder_hf_inner_model_subdir":
            accelerator.print(
                "[CHECKPOINT] Found CrossEncoder HF export in hf_model/. "
                "Loading inner model manually while leaving root checkpoint files untouched."
            )
            resume_state = load_crossencoder_hf_resume_checkpoint(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                checkpoint_dir=args.last_checkpoint,
                model_subdir="hf_model",
            )
        else:
            raise ValueError(
                "Unsupported checkpoint format. Expected one of: "
                "an Accelerate full-state checkpoint with wrapper-prefixed root model keys, "
                "a CrossEncoder HF inner-model checkpoint (root or hf_model/), "
                "or a PEFT adapter checkpoint. "
                f"Checkpoint dir: {args.last_checkpoint}"
            )

        print_memory_snapshot("after_resume_load_state", accelerator)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    trainer = Trainer(
        model=model,
        tokenizer=unwrapped_model.tokenizer,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        accelerator=accelerator,
        epochs=args.epochs,
        lr_scheduler=lr_scheduler,
        log_interval=args.log_interval * accelerator.gradient_state.num_steps,
        save_on_epoch_end=args.save_on_epoch_end,
        save_every_n_batches=args.save_every_n_batches,
        resume_state=resume_state,
    )

    print_memory_snapshot("before_training", accelerator)
    accelerator.print(f"Start training for {args.epochs} epochs ...")

    trainer.train()

    accelerator.print("Training finished!")
    print_memory_snapshot("after_training", accelerator)

    accelerator.print("Saving model ...")
    print_memory_snapshot("before_save", accelerator)

    save_dir = Path(args.output_dir) / "model"
    stage_dir = build_local_stage_dir(save_dir, "final_model") if should_stage_to_local(save_dir) else save_dir

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        prepare_clean_dir(stage_dir)

        cfg = dict(vars(args))
        cfg["training_lambda_prior_attention"] = cfg.get("lambda_prior_attention", 0.0)
        cfg["training_lambda_corr"] = cfg.get("lambda_corr", 0.05)

        save_run_config(stage_dir, cfg, run_id)
        save_unwrapped_model_pretrained(
            unwrapped_model,
            model,
            accelerator,
            stage_dir,
        )
        tokenizer_to_save = getattr(unwrapped_model, "tokenizer", None)
        if tokenizer_to_save is None:
            raise RuntimeError("Final model save failed: unwrapped model is missing tokenizer")
        tokenizer_to_save.save_pretrained(stage_dir)

        local_stats = verify_saved_artifact_dir(stage_dir, "final model local")
        log_save_stats("final model local", local_stats)

        if stage_dir.resolve() != save_dir.resolve():
            accelerator.print(f"[final model] local save directory: {stage_dir}")
            accelerator.print(f"[final model] drive save directory: {save_dir}")
            copy_directory_contents(stage_dir, save_dir)
            drive_stats = verify_saved_artifact_dir(save_dir, "final model drive")
            log_save_stats("final model drive", drive_stats)
        else:
            accelerator.print(f"[final model] save directory: {save_dir}")

    accelerator.wait_for_everyone()
    print_memory_snapshot("after_save", accelerator)

    if accelerator.is_main_process:
        accelerator.print(f"RUN_DIR::{run_dir}")
        accelerator.print("Saving Successfully!")


if __name__ == "__main__":
    main()
