import argparse
import json
from pathlib import Path
from typing import Optional, List
import pickle

from rag_retrieval import Reranker

from .eval_metrics import (
    evaluate_binary_and_graded,
    aggregate_metrics,
    save_metrics_csv,
    append_agg_metrics_csv,
    analyze_run_binary,
    load_run_config_or_default,
    add_exposure_metrics_to_agg,
    add_bias_metrics_to_agg,
)

from .build_run_qrels import (
    build_qrel_and_run_from_grouped_jsonl,
    load_creator_prior_attention,
    load_grouped_labels_from_jsonl,
)

from .skill_coverage import (
    parse_qid_meta,
    get_skipteams_from_splits,
    gen_member_skill_cooccurrence,
    compute_skill_coverage_at_k,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate grouped reranker JSONL with pytrec_eval (P@k, Recall@k, NDCG@k)"
    )
    p.add_argument("--jsonl", required=True, help="Path to grouped dataset JSONL")
    p.add_argument(
        "--model",
        required=True,
        help="HF model id OR local path to model folder (e.g. output/kaito/model)",
    )
    p.add_argument(
        "--ks",
        default="2,5,10",
        help="Comma-separated cutoffs, e.g. 2,5,10",
    )
    p.add_argument(
        "--output_dir",
        default="eval_outputs",
        help="Base output dir. Outputs will be written to <output_dir>/eval/",
    )

    p.add_argument(
        "--relevance_gt",
        type=float,
        default=0.0,
        help="Binary relevance threshold: label > relevance_gt is relevant (default: 0.0)",
    )
    p.add_argument(
        "--k_run",
        type=int,
        default=None,
        help="Optional: only keep top-k docs in the run (None keeps all).",
    )

    p.add_argument(
        "--extra_measures",
        default="",
        help='Optional extra measures, comma-separated (e.g. "map,recip_rank"). Default: none.',
    )
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--eval_max_length", type=int, default=256)
    p.add_argument(
        "--device_map",
        default=None,
        help='Optional model loading device_map. Use "auto" to shard evaluation across visible GPUs.',
    )
    p.add_argument(
        "--raw-data-dir",
        default=None,
        help=(
            "Path to raw data directory containing the prior-attention file, "
            "creator_details.csv, indexes.pkl, teamsvecs.pkl, "
            "splits.t5.r0.85.pkl, and gpt5_skills.csv. If omitted, "
            "uses <jsonl parent parent>/raw."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())

    eval_dir = Path(args.output_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Infer raw data paths from jsonl location
    # --------------------------------------------------
    jsonl_path = Path(args.jsonl).resolve()
    if args.raw_data_dir:
        raw_dir = Path(args.raw_data_dir).resolve()
    else:
        root = jsonl_path.parent.parent
        raw_dir = root / "raw"

    teamsvecs_pkl = raw_dir / "teamsvecs.pkl"
    indexes_pkl = raw_dir / "indexes.pkl"
    splits_pkl = raw_dir / "splits.t5.r0.85.pkl"
    skills_csv = raw_dir / "gpt5_skills.csv"

    creator_prior_attention = load_creator_prior_attention(raw_dir)

    # --------------------------------------------------
    # Load run config
    # --------------------------------------------------
    run_cfg = load_run_config_or_default(args.model)

    template_name = run_cfg.get("template_name", "v2")
    loss_type = run_cfg.get("loss_type", "unknown")

    training_peft_method = run_cfg.get("peft_method", "none")
    training_lambda_corr = run_cfg.get("lambda_corr", 0.0)
    training_lambda_prior_attention = run_cfg.get("lambda_prior_attention", 0.0)

    # --------------------------------------------------
    # Resolve model_type for Reranker
    # --------------------------------------------------
    raw_model_type = run_cfg.get("model_type", "cross-encoder")

    MODEL_TYPE_MAP = {
        "bert_encoder": "cross-encoder",
        "llm_decoder": "llm-decoder",
        "cross-encoder": "cross-encoder",
        "llm-decoder": "llm-decoder",
        "llm": "llm",
    }

    model_type = MODEL_TYPE_MAP.get(raw_model_type)
    if model_type is None:
        raise ValueError(
            f"Unsupported model_type in run config: {raw_model_type}"
        )

    # --------------------------------------------------
    # Load reranker
    # --------------------------------------------------
    ranker_kwargs = {
        "model_type": model_type,
        "verbose": 0,
    }
    if model_type == "llm-decoder":
        ranker_kwargs.update(
            {
                "query_format": run_cfg.get("query_format", "query: {}"),
                "document_format": run_cfg.get("document_format", "document: {}"),
                "seq": run_cfg.get("seq", "\n"),
                "special_token": run_cfg.get("special_token", "\nrelevance"),
                "device_map": args.device_map,
            }
        )

    ranker = Reranker(
        args.model,
        **ranker_kwargs,
    )
    if ranker is None:
        raise RuntimeError(f"Could not construct reranker for model_type={model_type!r} from model={args.model!r}")

    # --------------------------------------------------
    # Build qrels + run
    # --------------------------------------------------
    qrels_binary, qrels_graded, run, qid_to_query = build_qrel_and_run_from_grouped_jsonl(
        jsonl_path=args.jsonl,
        ranker=ranker,
        k=args.k_run,
        relevance_if_label_gt=args.relevance_gt,
        eval_batch_size=args.eval_batch_size,
        eval_max_length=args.eval_max_length,
    )

    qid_to_labels = load_grouped_labels_from_jsonl(str(jsonl_path))

    analyze_run_binary(qrels_binary, run)

    extra: Optional[List[str]] = (
        [m.strip() for m in args.extra_measures.split(",") if m.strip()] or None
    )

    measures, per_query = evaluate_binary_and_graded(
        qrels_binary=qrels_binary,
        qrels_graded=qrels_graded,
        run=run,
        ks=ks,
        extra_measures=extra,
    )

    # --------------------------------------------------
    # Skill coverage
    # --------------------------------------------------
    with open(teamsvecs_pkl, "rb") as f:
        teamsvecs = pickle.load(f)

    with open(indexes_pkl, "rb") as f:
        indexes = pickle.load(f)

    with open(splits_pkl, "rb") as f:
        splits_bundle = pickle.load(f)

    if not run:
        raise RuntimeError("Run is empty — cannot compute skill coverage.")

    any_qid = next(iter(run))
    trial_id, fold_id, split_name = parse_qid_meta(any_qid)

    skipteams = get_skipteams_from_splits(
        splits_bundle,
        trial_id,
        fold_id,
        split_name,
    )

    cache_path = (
        eval_dir
        / f"skillcoverage_member_skill_co_t{trial_id}_f{fold_id}_{split_name}.pkl"
    )

    member_skill_co = gen_member_skill_cooccurrence(
        teamsvecs=teamsvecs,
        cache_path=cache_path,
        skipteams=skipteams,
    )

    skc_per_query = compute_skill_coverage_at_k(
        run=run,
        qid_to_query=qid_to_query,
        indexes=indexes,
        member_skill_co=member_skill_co,
        skill_keywords_csv=skills_csv,
        ks=ks,
    )

    for qid in per_query:
        per_query[qid].update(skc_per_query.get(qid, {}))

    measures |= {f"skill_coverage_{k}" for k in ks}

    # --------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------
    agg = aggregate_metrics(per_query, measures)

    agg = add_exposure_metrics_to_agg(
        agg=agg,
        run=run,
        creator_prior_attention=creator_prior_attention,
        ks=ks,
    )

    agg = add_bias_metrics_to_agg(
        agg=agg,
        run=run,
        qid_to_labels=qid_to_labels,
        creator_prior_attention=creator_prior_attention,
        ks=ks,
    )

    meta = {}

    agg["run_time"] = meta["run_time"] = run_cfg.get("run_time", "unknown")
    agg["model_name_or_path"] = meta["model_name_or_path"] = run_cfg.get(
        "model_name_or_path",
        args.model,
    )
    agg["template_name"] = meta["template_name"] = template_name
    agg["loss_type"] = meta["loss_type"] = loss_type
    agg["skill_set_version"] = meta["skill_set_version"] = run_cfg.get(
        "skill_set_version",
        "rootdata",
    )

    agg["training_peft_method"] = meta["training_peft_method"] = training_peft_method
    agg["training_lambda_corr"] = meta["training_lambda_corr"] = training_lambda_corr
    agg["training_lambda_prior_attention"] = meta["training_lambda_prior_attention"] = training_lambda_prior_attention

    def safe(s: str) -> str:
        return (
            str(s)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )

    # filenames include loss
    run_id = Path(args.model).parent.name
    safe_model = safe(meta["model_name_or_path"])
    safe_loss = safe(loss_type)

    per_query_csv = eval_dir / (
        f"metrics_per_query_{run_id}_{safe_model}_{safe_loss}_{meta['skill_set_version']}.csv"
    )

    agg_csv = eval_dir / "results.csv"

    save_metrics_csv(
        per_query=per_query,
        model_name=args.model,
        output_csv=str(per_query_csv),
        ks=ks,
        meta=meta,
    )

    append_agg_metrics_csv(
        agg=agg,
        ks=ks,
        output_csv=str(agg_csv),
    )

    if args.verbose:
        print("\n📊 Aggregated metrics (this run):")
        print(json.dumps(agg, indent=2))
        print(f"\n✅ Saved per-query CSV: {per_query_csv}")
        print(f"✅ Appended agg CSV   : {agg_csv}")


if __name__ == "__main__":
    main()
