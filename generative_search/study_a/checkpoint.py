from pathlib import Path
from typing import Optional

from rag_retrieval.infer.eval.eval_metrics import load_run_config_or_default

MODEL_TYPE_MAP = {
    "bert_encoder": "cross-encoder", "llm_decoder": "llm-decoder",
    "cross-encoder": "cross-encoder", "llm-decoder": "llm-decoder",
    "llm": "llm",
}
SUPPORTED_MODEL_TYPES = {"cross-encoder", "llm-decoder", "llm"}


def detect_checkpoint(model_path: str, model_type_override: Optional[str] = None, loss_type_override: Optional[str] = None) -> dict:
    try:
        run_cfg = load_run_config_or_default(model_path)
    except Exception as exc:
        raise RuntimeError(f"Could not load checkpoint metadata for {model_path}: {exc}") from exc
    raw_model_type = model_type_override or run_cfg.get("model_type")
    if not raw_model_type:
        raise RuntimeError("Checkpoint metadata has no model_type; provide --model_type explicitly")
    model_type = MODEL_TYPE_MAP.get(str(raw_model_type).strip().lower(), str(raw_model_type).strip().lower())
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise RuntimeError(f"Unsupported checkpoint architecture {raw_model_type!r}; supported types: {sorted(SUPPORTED_MODEL_TYPES)}")
    loss_type = loss_type_override or run_cfg.get("loss_type")
    if not loss_type or str(loss_type).lower() == "unknown":
        raise RuntimeError("Checkpoint metadata has no true internal loss_type; provide --loss_type explicitly")
    return {
        "model_path": str(Path(model_path)), "model_type": model_type,
        "raw_model_type": raw_model_type, "loss_type": str(loss_type),
        "run_config": run_cfg, "manual_model_type_override": model_type_override,
        "manual_loss_type_override": loss_type_override,
    }
