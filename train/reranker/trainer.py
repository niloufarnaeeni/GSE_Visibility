from __future__ import annotations

import json
import os
import re
import shutil
import gc
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Sized

import psutil
import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


def mem_debug(tag: str):
    process = psutil.Process(os.getpid())
    rss_gb = process.memory_info().rss / 1024**3
    vms_gb = process.memory_info().vms / 1024**3
    print(f"[{tag}] RAM rss={rss_gb:.2f} GB | vms={vms_gb:.2f} GB")

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        for i in range(torch.cuda.device_count()):
            allocated_gb = torch.cuda.memory_allocated(i) / 1024**3
            reserved_gb = torch.cuda.memory_reserved(i) / 1024**3
            peak_allocated_gb = torch.cuda.max_memory_allocated(i) / 1024**3
            peak_reserved_gb = torch.cuda.max_memory_reserved(i) / 1024**3
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
            free_gb = free_bytes / 1024**3
            total_gb = total_bytes / 1024**3
            print(
                f"[{tag}] cuda:{i} alloc={allocated_gb:.2f} GB | "
                f"reserved={reserved_gb:.2f} GB | "
                f"peak_alloc={peak_allocated_gb:.2f} GB | "
                f"peak_reserved={peak_reserved_gb:.2f} GB | "
                f"free={free_gb:.2f} GB / total={total_gb:.2f} GB"
            )


def mem_debug_gc(tag: str):
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    mem_debug(tag)


IMPORTANT_SAVE_FILENAMES = {
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
    "config.json",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "trainer_state.json",
}


def _format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def _list_files_recursive(path: str | Path) -> list[Path]:
    root = Path(path)
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


def collect_save_stats(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    files = _list_files_recursive(root)
    important_files = sorted({p.name for p in files if p.name in IMPORTANT_SAVE_FILENAMES})
    important_file_sizes = {
        p.name: int(p.stat().st_size)
        for p in sorted(files)
        if p.name in IMPORTANT_SAVE_FILENAMES
    }
    total_size = sum(p.stat().st_size for p in files)
    return {
        "path": str(root),
        "file_count": len(files),
        "total_size_bytes": int(total_size),
        "total_size_human": _format_size(total_size),
        "important_files": important_files,
        "important_file_sizes": important_file_sizes,
    }


def verify_saved_artifact_dir(path: str | Path, artifact_name: str) -> dict[str, Any]:
    root = Path(path)
    if not root.exists():
        raise RuntimeError(f"[{artifact_name}] Save failed: directory does not exist: {root}")

    stats = collect_save_stats(root)
    if stats["file_count"] <= 0:
        raise RuntimeError(
            f"[{artifact_name}] Save failed: directory is empty after save: {root}"
        )
    if not stats["important_files"]:
        raise RuntimeError(
            f"[{artifact_name}] Save failed: no important model/tokenizer/checkpoint files were found in {root}. "
            f"Found {stats['file_count']} files totaling {stats['total_size_human']}."
        )
    return stats


def log_save_stats(prefix: str, stats: dict[str, Any]) -> None:
    print(
        f"[{prefix}] dir={stats['path']} | file_count={stats['file_count']} | "
        f"total_size={stats['total_size_human']} | important_files={stats['important_files']} | "
        f"important_file_sizes={stats['important_file_sizes']}"
    )


def resolve_save_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    try:
        return resolved.resolve()
    except Exception:
        return Path(os.path.abspath(str(resolved)))



def should_stage_to_local(path: str | Path) -> bool:
    dest = resolve_save_path(path)
    return str(dest).startswith("/content/drive/") and Path("/content").exists()


def build_local_stage_dir(dest_dir: str | Path, artifact_name: str) -> Path:
    dest = resolve_save_path(dest_dir)
    digest = hashlib.md5(str(dest).encode("utf-8")).hexdigest()[:12]
    return Path("/content") / "save_staging" / artifact_name / f"{dest.name}_{digest}"


def prepare_clean_dir(path: str | Path) -> Path:
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def sync_filesystem() -> None:
    sync_fn = getattr(os, "sync", None)
    if callable(sync_fn):
        sync_fn()


def compare_saved_artifact_dirs(
    src: str | Path,
    dest: str | Path,
    artifact_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    src_stats = verify_saved_artifact_dir(src, f"{artifact_name} source")
    dest_stats = verify_saved_artifact_dir(dest, artifact_name)

    if src_stats["file_count"] != dest_stats["file_count"]:
        raise RuntimeError(
            f"[{artifact_name}] Verification failed: file count mismatch between "
            f"{src_stats['path']} ({src_stats['file_count']}) and "
            f"{dest_stats['path']} ({dest_stats['file_count']})."
        )

    if src_stats["total_size_bytes"] != dest_stats["total_size_bytes"]:
        raise RuntimeError(
            f"[{artifact_name}] Verification failed: total size mismatch between "
            f"{src_stats['path']} ({src_stats['total_size_human']}) and "
            f"{dest_stats['path']} ({dest_stats['total_size_human']})."
        )

    if src_stats["important_files"] != dest_stats["important_files"]:
        raise RuntimeError(
            f"[{artifact_name}] Verification failed: important file mismatch between "
            f"{src_stats['path']} ({src_stats['important_files']}) and "
            f"{dest_stats['path']} ({dest_stats['important_files']})."
        )

    for file_name, src_size in src_stats["important_file_sizes"].items():
        dest_size = dest_stats["important_file_sizes"].get(file_name)
        if dest_size != src_size:
            raise RuntimeError(
                f"[{artifact_name}] Verification failed: size mismatch for {file_name}: "
                f"source={src_size} bytes dest={dest_size} bytes."
            )

    return src_stats, dest_stats



def copy_directory_contents(src: str | Path, dest: str | Path) -> None:
    src_path = Path(src)
    dest_path = Path(dest)
    dest_tmp_path = dest_path.with_name(dest_path.name + ".tmp_copying")
    backup_path = dest_path.with_name(dest_path.name + ".backup_failed_replace")

    if dest_tmp_path.exists():
        shutil.rmtree(dest_tmp_path)

    if shutil.which("rsync"):
        dest_tmp_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "-rv",
                "--no-o",
                "--no-g",
                "--no-perms",
                "--omit-dir-times",
                "--info=progress2",
                "--partial",
                f"{src_path}/",
                f"{dest_tmp_path}/",
            ],
            check=True,
        )
    else:
        shutil.copytree(src_path, dest_tmp_path)

    sync_filesystem()

    tmp_stats = verify_saved_artifact_dir(dest_tmp_path, f"{dest_path.name} temporary copy")
    log_save_stats(f"{dest_path.name} temporary copy", tmp_stats)
    compare_saved_artifact_dirs(src_path, dest_tmp_path, f"{dest_path.name} temporary copy")

    if backup_path.exists():
        shutil.rmtree(backup_path)

    if dest_path.exists():
        dest_path.rename(backup_path)

    try:
        dest_tmp_path.rename(dest_path)
    except Exception:
        if backup_path.exists() and not dest_path.exists():
            backup_path.rename(dest_path)
        raise

    sync_filesystem()

    final_stats = verify_saved_artifact_dir(dest_path, f"{dest_path.name} final copy")
    log_save_stats(f"{dest_path.name} final copy", final_stats)
    compare_saved_artifact_dirs(src_path, dest_path, f"{dest_path.name} final copy")

    if backup_path.exists():
        shutil.rmtree(backup_path)


def save_unwrapped_model_pretrained(
    unwrapped_model,
    wrapped_model,
    accelerator: Accelerator,
    save_dir: str | Path,
    hf_subdir: str | None = None,
) -> None:
    save_dir = Path(save_dir)
    if hasattr(getattr(unwrapped_model, "model", None), "peft_config") or hasattr(unwrapped_model, "peft_config"):
        unwrapped_model.save_pretrained(
            str(save_dir),
            safe_serialization=True,
        )
        return

    # CrossEncoder wraps a Hugging Face encoder model at `.model`. For final
    # exported models we save the inner encoder directly at the root so
    # evaluation can load standard Hugging Face keys. For training checkpoints
    # we optionally place that HF export in a subdirectory so the root
    # Accelerate state keeps its wrapper-compatible `model.*` keys for resume.
    if unwrapped_model.__class__.__name__ == "CrossEncoder" and hasattr(unwrapped_model, "model"):
        target_dir = save_dir / hf_subdir if hf_subdir else save_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model.model.save_pretrained(
            str(target_dir),
            safe_serialization=True,
        )
        print(f"[SAVE] Saved CrossEncoder HF export to {target_dir}")
        return

    model_state_dict = accelerator.get_state_dict(wrapped_model)
    unwrapped_model.save_pretrained(
        str(save_dir),
        state_dict=model_state_dict,
        safe_serialization=True,
    )


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        optimizer: Optimizer,
        accelerator: Accelerator,
        validation_dataloader: DataLoader | None = None,
        epochs: int = 3,
        lr_scheduler: LRScheduler,
        log_interval: int = 10,
        save_on_epoch_end: bool = True,
        save_every_n_batches: int = 0,
        resume_state: dict[str, Any] | None = None,
        tokenizer,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.epochs = epochs
        self.log_interval = log_interval
        self.save_on_epoch_end = save_on_epoch_end
        self.save_every_n_batches = max(int(save_every_n_batches or 0), 0)
        self.tokenizer = tokenizer
        self.min_val_loss = 100000
        self.resume_state = dict(resume_state or {})

        self.train_loss_tracker = LossTracker()
        self.validation_loss_tracker = LossTracker()
        if isinstance(self.train_dataloader.dataset, Sized):
            num_steps_per_epoch = len(self.train_dataloader)
        else:
            num_steps_per_epoch = None
        self.progress_bar = DistributedTqdmProgressBar(
            self.accelerator, self.epochs, num_steps_per_epoch=num_steps_per_epoch
        )
        self.current_step = int(self.resume_state.get("current_step", 0))
        self.global_step = int(self.resume_state.get("global_step", 0))
        self.resume_epoch = int(self.resume_state.get("current_epoch", 1))
        self.resume_batches_completed = int(
            self.resume_state.get("completed_batches_in_epoch", 0)
        )
        self.next_batch_checkpoint = self._compute_next_batch_checkpoint(
            self.current_step
        )

    def _compute_next_batch_checkpoint(self, completed_batches: int) -> int | None:
        if self.save_every_n_batches <= 0:
            return None
        return (
            (int(completed_batches) // self.save_every_n_batches) + 1
        ) * self.save_every_n_batches

    def train(self):
        for current_epoch in range(self.resume_epoch, self.epochs + 1):
            print(f"\n=== starting epoch {current_epoch} ===")
            mem_debug_gc(f"epoch_{current_epoch}_start")

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            start_batch_index = 0
            epoch_dataloader = self.train_dataloader
            if current_epoch == self.resume_epoch and self.resume_batches_completed > 0:
                start_batch_index = self.resume_batches_completed
                print(
                    "[CHECKPOINT] Skipping already completed batches in resumed epoch: "
                    f"epoch={current_epoch} batches={start_batch_index}"
                )
                epoch_dataloader = self.accelerator.skip_first_batches(
                    self.train_dataloader,
                    num_batches=start_batch_index,
                )

            self.progress_bar.on_epoch_start(
                current_epoch=current_epoch,
                initial_step=start_batch_index,
            )

            for batch_index, batch in enumerate(epoch_dataloader, start=start_batch_index):
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        try:
                            torch.cuda.reset_peak_memory_stats(i)
                        except Exception:
                            pass

                if batch_index == 0:
                    print(f"--- epoch {current_epoch} batch 0 ---")
                    mem_debug(f"epoch_{current_epoch}_batch0")

                if batch_index in {95, 100, 101, 102, 103, 104, 105}:
                    mem_debug(f"epoch_{current_epoch}_batch_{batch_index}_before_step")

                with self.accelerator.accumulate(self.model):
                    batch_output = self.model(batch[0], batch[1])
                    loss = batch_output["loss"]

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1

                    self.train_loss_tracker.update(loss)

                completed_batch_step = self.current_step + 1
                if (
                    self.next_batch_checkpoint is not None
                    and completed_batch_step >= self.next_batch_checkpoint
                    and self.accelerator.sync_gradients
                ):
                    self.save_step_checkpoint(
                        batch_step=completed_batch_step,
                        global_step=self.global_step,
                        current_epoch=current_epoch,
                        completed_batches_in_epoch=batch_index + 1,
                    )
                    while (
                        self.next_batch_checkpoint is not None
                        and completed_batch_step >= self.next_batch_checkpoint
                    ):
                        self.next_batch_checkpoint += self.save_every_n_batches

                if batch_index in {95, 100, 101, 102, 103, 104, 105}:
                    mem_debug(f"epoch_{current_epoch}_batch_{batch_index}_after_step")

                if batch_index % self.log_interval == 0:
                    log_dic = {
                        "cur_loss": batch_output["loss"],
                        "lr": float(self.lr_scheduler.get_lr()[0]),
                        "avg_loss": self.train_loss_tracker.loss,
                    }
                    self.log_metrics(
                        log_dic,
                        step=self.current_step,
                    )

                if (
                    self.validation_dataloader
                    and batch_index % (self.log_interval * 10) == 0
                ):
                    print(
                        f"*** validation start | epoch={current_epoch} "
                        f"batch={batch_index} step={self.current_step} ***"
                    )
                    mem_debug_gc(
                        f"before_validation_e{current_epoch}_b{batch_index}"
                    )

                    validation_loss = evaluate(
                        self.model,
                        self.validation_dataloader,
                        self.validation_loss_tracker,
                    )

                    mem_debug_gc(
                        f"after_validation_e{current_epoch}_b{batch_index}"
                    )
                    print(
                        f"*** validation end | epoch={current_epoch} "
                        f"batch={batch_index} step={self.current_step} | "
                        f"val_loss={validation_loss} ***"
                    )

                    self.accelerator.log(
                        {"val_loss": validation_loss}, step=self.current_step
                    )
                    # If you want to save the model with min validation loss, uncomment the following code.
                    # if validation_loss < self.min_val_loss:
                    #     if self.accelerator.is_local_main_process and self.current_step > 0:
                    #         save_dir = self.get_checkpoint_dir(current_epoch)
                    #         save_dir = os.path.join(
                    #             save_dir,
                    #             f"_min_val_loss",
                    #         )
                    #         self.min_val_loss = validation_loss
                    #         print(f"Saving model with min validation loss: {validation_loss}, step: {self.current_step}")
                    #         unwrapped_model = self.accelerator.unwrap_model(self.model)
                    #         unwrapped_model.save_pretrained(
                    #             save_dir, safe_serialization=True
                    #         )
                    #         self.tokenizer.save_pretrained(save_dir)
                    #     self.accelerator.wait_for_everyone()

                self.progress_bar.update()
                self.current_step += 1

            mem_debug_gc(f"epoch_{current_epoch}_end")

            self.train_loss_tracker.on_epoch_end()
            self.progress_bar.on_epoch_end()

            if self.save_on_epoch_end:
                self.save_epoch_checkpoint(current_epoch)

            self.resume_batches_completed = 0

        mem_debug_gc("end_training_before_accelerator_end")
        self.accelerator.end_training()

    def log_metrics(self, metrics: dict[str, float], step: int):
        self.accelerator.log(metrics, step=step)
        self.progress_bar.show_metrics(metrics)

    @staticmethod
    def add_prefix(values: dict[str, Any], prefix: str):
        return {f"{prefix}/{k}": v for k, v in values.items()}

    def get_checkpoint_root(self) -> str:
        self.accelerator.project_configuration.automatic_checkpoint_naming = False
        output_dir = os.path.join(self.accelerator.project_dir, "checkpoints")
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def get_latest_checkpoint_manifest_path(self) -> Path:
        return Path(self.accelerator.project_dir).parent / "latest_checkpoint.json"

    def write_latest_checkpoint_manifest(self, payload: dict[str, Any]) -> None:
        manifest_path = self.get_latest_checkpoint_manifest_path()
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _list_epoch_checkpoint_dirs(self, output_dir: str) -> list[str]:
        epoch_pattern = re.compile(r"^checkpoint_(\d+)$")
        folders = []
        for folder in os.listdir(output_dir):
            folder_path = os.path.join(output_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            if epoch_pattern.match(folder):
                folders.append(folder_path)
        return folders

    def get_epoch_checkpoint_dir(self, current_epoch: int) -> str:
        output_dir = os.path.join(self.get_checkpoint_root(), f"checkpoint_{current_epoch-1}")
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def get_step_checkpoint_dir(self, batch_step: int) -> str:
        output_dir = os.path.join(
            self.get_checkpoint_root(),
            f"checkpoint-step-{batch_step}",
        )
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def save_trainer_state(
        self,
        checkpoint_dir: str,
        *,
        current_epoch: int,
        completed_batches_in_epoch: int,
        current_step: int | None = None,
    ) -> None:
        state = {
            "current_epoch": int(current_epoch),
            "completed_batches_in_epoch": int(completed_batches_in_epoch),
            "current_step": int(self.current_step if current_step is None else current_step),
            "global_step": int(self.global_step),
        }
        trainer_state_path = Path(checkpoint_dir) / "trainer_state.json"
        with open(trainer_state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def remove_step_checkpoints(self) -> int:
        removed = 0
        checkpoint_root = Path(self.get_checkpoint_root())
        for checkpoint_dir in checkpoint_root.glob("checkpoint-step-*"):
            if checkpoint_dir.is_dir():
                shutil.rmtree(checkpoint_dir)
                removed += 1
        return removed

    def remove_old_epoch_checkpoints(self) -> int:
        total_limit = self.accelerator.project_configuration.total_limit
        if total_limit is None:
            return 0

        checkpoint_root = self.get_checkpoint_root()
        folders = self._list_epoch_checkpoint_dirs(checkpoint_root)
        if len(folders) <= total_limit:
            return 0

        def _inner(folder):
            return list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[0]

        folders.sort(key=_inner)
        removed = 0
        for folder in folders[: len(folders) - total_limit]:
            shutil.rmtree(folder)
            removed += 1
        return removed

    def _prepare_checkpoint_stage_dir(self, checkpoint_dir: str) -> Path:
        checkpoint_path = Path(checkpoint_dir)
        if should_stage_to_local(checkpoint_path):
            stage_dir = build_local_stage_dir(checkpoint_path, "checkpoint")
            if self.accelerator.is_main_process:
                prepare_clean_dir(stage_dir)
            self.accelerator.wait_for_everyone()
            return stage_dir

        if self.accelerator.is_main_process:
            prepare_clean_dir(checkpoint_path)
        self.accelerator.wait_for_everyone()
        return checkpoint_path

    def _finalize_verified_save(self, stage_dir: Path, dest_dir: Path, artifact_name: str) -> dict[str, Any]:
        local_stats = verify_saved_artifact_dir(stage_dir, f"{artifact_name} local")
        log_save_stats(f"{artifact_name} local", local_stats)

        if stage_dir.resolve() != dest_dir.resolve():
            print(f"[{artifact_name}] local save directory: {stage_dir}")
            print(f"[{artifact_name}] drive save directory: {dest_dir}")
            copy_directory_contents(stage_dir, dest_dir)
            drive_stats = verify_saved_artifact_dir(dest_dir, f"{artifact_name} drive")
            log_save_stats(f"{artifact_name} drive", drive_stats)
            return drive_stats

        print(f"[{artifact_name}] save directory: {dest_dir}")
        return local_stats

    def save_step_checkpoint(
        self,
        *,
        batch_step: int,
        global_step: int,
        current_epoch: int,
        completed_batches_in_epoch: int,
    ) -> None:
        checkpoint_dir = Path(self.get_step_checkpoint_dir(batch_step))
        stage_dir = self._prepare_checkpoint_stage_dir(str(checkpoint_dir))

        self.accelerator.save_state(str(stage_dir))
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            save_unwrapped_model_pretrained(
                unwrapped_model,
                self.model,
                self.accelerator,
                stage_dir,
                hf_subdir="hf_model",
            )
            self.tokenizer.save_pretrained(stage_dir)
            self.save_trainer_state(
                stage_dir,
                current_epoch=current_epoch,
                completed_batches_in_epoch=completed_batches_in_epoch,
                current_step=batch_step,
            )
            self._finalize_verified_save(stage_dir, checkpoint_dir, f"checkpoint-step-{batch_step}")
            self.write_latest_checkpoint_manifest(
                {
                    "checkpoint_path": str(checkpoint_dir),
                    "checkpoint_type": "step",
                    "batch_step": int(batch_step),
                    "global_step": int(global_step),
                    "epoch": int(current_epoch),
                }
            )
            print(
                "[CHECKPOINT] Saved verified step checkpoint at "
                f"batch_step={batch_step} global_step={global_step} epoch={current_epoch}"
            )
            print(f"LAST_CHECKPOINT::{checkpoint_dir}")
        self.accelerator.wait_for_everyone()

    def save_epoch_checkpoint(self, current_epoch: int) -> None:
        checkpoint_dir = Path(self.get_epoch_checkpoint_dir(current_epoch))
        stage_dir = self._prepare_checkpoint_stage_dir(str(checkpoint_dir))
        mem_debug_gc(f"before_checkpoint_save_epoch_{current_epoch}")
        self.accelerator.save_state(str(stage_dir))
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            save_unwrapped_model_pretrained(
                unwrapped_model,
                self.model,
                self.accelerator,
                stage_dir,
                hf_subdir="hf_model",
            )
            self.tokenizer.save_pretrained(stage_dir)
            self.save_trainer_state(
                stage_dir,
                current_epoch=current_epoch + 1,
                completed_batches_in_epoch=0,
            )
            self._finalize_verified_save(stage_dir, checkpoint_dir, f"checkpoint-epoch-{current_epoch}")
            self.write_latest_checkpoint_manifest(
                {
                    "checkpoint_path": str(checkpoint_dir),
                    "checkpoint_type": "epoch",
                    "epoch": int(current_epoch),
                }
            )
            print(f"[CHECKPOINT] Saved verified epoch checkpoint at epoch={current_epoch}")
            print(f"LAST_CHECKPOINT::{checkpoint_dir}")
            removed = self.remove_step_checkpoints()
            print(
                f"[CHECKPOINT] Removed {removed} step checkpoints after verified epoch checkpoint save"
            )
            removed_epochs = self.remove_old_epoch_checkpoints()
            print(
                f"[CHECKPOINT] Removed {removed_epochs} old epoch checkpoints after verified epoch checkpoint save"
            )
            mem_debug_gc(f"after_checkpoint_save_epoch_{current_epoch}")
        self.accelerator.wait_for_everyone()


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_tracker: LossTracker | None = None,
):
    print(">>> evaluate() start")
    mem_debug_gc("evaluate_start")

    loss_tracker = loss_tracker or LossTracker()
    for eval_batch_index, batch in enumerate(dataloader):
        if eval_batch_index == 0:
            print("--- evaluate batch 0 ---")
            mem_debug("evaluate_batch0")

        if eval_batch_index in {1, 2, 3, 4, 5}:
            mem_debug(f"evaluate_batch_{eval_batch_index}")

        with torch.inference_mode():
            batch_output = model(batch[0], batch[1])
            loss_tracker.update(batch_output["loss"])

    loss = loss_tracker.loss
    loss_tracker.on_epoch_end()

    mem_debug_gc("evaluate_end")
    print(f">>> evaluate() end | loss={loss}")
    return loss


class DummyProgressBar:
    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass

    def set_description(self, description: str) -> None:
        pass


class DistributedTqdmProgressBar:
    def __init__(
        self, accelerator, epochs: int, num_steps_per_epoch: int | None, **kwargs
    ) -> None:
        self.accelerator = accelerator
        self.epochs = epochs
        self.current_epoch = 1
        self.num_steps_per_epoch = num_steps_per_epoch
        self.tqdm_kwargs = kwargs

    def on_epoch_start(
        self,
        current_epoch: int | None = None,
        initial_step: int = 0,
    ):
        if current_epoch is not None:
            self.current_epoch = int(current_epoch)
        initial_step = max(int(initial_step), 0)
        if self.num_steps_per_epoch is not None:
            initial_step = min(initial_step, int(self.num_steps_per_epoch))
        if self.accelerator.is_main_process:
            self.progress_bar = tqdm(
                total=self.num_steps_per_epoch,
                initial=initial_step,
                **self.tqdm_kwargs,
            )
        else:
            self.progress_bar = DummyProgressBar()

    def update(self, n: int = 1) -> None:
        self.progress_bar.update(n)

    def close(self) -> None:
        self.progress_bar.close()

    def on_epoch_end(self) -> None:
        self.current_epoch += 1
        self.progress_bar.close()

    def show_metrics(self, metrics: dict[str, float]) -> None:
        description = f"Epoch {self.current_epoch}/{self.epochs}"
        for name, score in metrics.items():
            description += f" - {name}: {score:.6f}"
        self.progress_bar.set_description(description)


class LossTracker:
    def __init__(
        self,
        ndigits=4,
    ) -> None:
        self.ndigits = ndigits
        self._loss: float = 0.0
        self.loss_count: int = 0
        self.history: list[float] = []

    def update(self, loss_tensor: torch.Tensor):
        loss = loss_tensor.item()
        self._loss = (self._loss * self.loss_count + loss) / (self.loss_count + 1)
        self.loss_count += 1

    def reset(self):
        self._loss = 0
        self.loss_count = 0

    def on_epoch_end(self, reset: bool = True):
        self.history.append(self.loss)
        if reset:
            self.reset()

    @property
    def loss(self) -> float:
        return round(float(self._loss), self.ndigits)