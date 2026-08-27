import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
try:
    import pytrec_eval
except ImportError:
    pytrec_eval = None
import math
import yaml

# -----------------------------
# Measures
# -----------------------------
def make_measures_for_ks(ks=(2, 5, 10)) -> set:
    """
    Measures that match your current reporting needs.
    make_measures_for_ks defines TREC-native metrics that:

    are computed by pytrec_eval

    must be known to pytrec_eval.RelevanceEvaluator
    """
    measures = set()
    for k in ks:
        measures |= {f"P_{k}", f"recall_{k}", f"ndcg_cut_{k}", f"map_cut_{k}",}
    return measures


def evaluate_once(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    ks=(2, 5, 10),
    extra_measures: Optional[List[str]] = None,
) -> Tuple[set, Dict[str, Dict[str, float]]]:
    if pytrec_eval is None:
        raise ImportError("pytrec_eval is required for TREC metric evaluation")
    measures = make_measures_for_ks(ks)
    if extra_measures:
        measures |= set(extra_measures)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_query = evaluator.evaluate(run)
    return measures, per_query

def evaluate_binary_and_graded(
    qrels_binary: Dict[str, Dict[str, int]],
    qrels_graded: Dict[str, Dict[str, float]],
    run: Dict[str, Dict[str, float]],
    ks=(2, 5, 10),
    extra_measures: Optional[List[str]] = None,
):
    """
    Evaluate binary and graded metrics separately and keep separate names.

    Binary side:
      - P@k
      - Recall@k
      - MAP@k
      - binary NDCG@k

    Graded side:
      - graded NDCG@k

    Note:
      P / Recall are inherently binary-style metrics.
      MAP is usually interpreted in a binary way too.
      The main metric that meaningfully differs here is NDCG.
    """
    if pytrec_eval is None:
        raise ImportError("pytrec_eval is required for TREC metric evaluation")

    binary_measures = set()
    graded_measures = set()

    for k in ks:
        binary_measures |= {
            f"P_{k}",
            f"recall_{k}",
            f"map_cut_{k}",
            f"ndcg_cut_{k}",   # binary NDCG
        }
        graded_measures |= {
            f"ndcg_cut_{k}",   # graded NDCG
        }

    if extra_measures:
        binary_measures |= set(extra_measures)

    per_query_binary_raw = pytrec_eval.RelevanceEvaluator(
        qrels_binary, binary_measures
    ).evaluate(run)

    per_query_graded_raw = pytrec_eval.RelevanceEvaluator(
        qrels_graded, graded_measures
    ).evaluate(run)

    # Merge with explicit names
    per_query = {}
    all_qids = set(per_query_binary_raw) | set(per_query_graded_raw)

    for qid in all_qids:
        per_query[qid] = {}

        if qid in per_query_binary_raw:
            for metric_name, value in per_query_binary_raw[qid].items():
                if metric_name.startswith("ndcg_cut_"):
                    per_query[qid][f"binary_{metric_name}"] = value
                else:
                    per_query[qid][metric_name] = value

        if qid in per_query_graded_raw:
            for metric_name, value in per_query_graded_raw[qid].items():
                if metric_name.startswith("ndcg_cut_"):
                    per_query[qid][f"graded_{metric_name}"] = value
                else:
                    per_query[qid][f"graded_{metric_name}"] = value

    measures = set()

    for k in ks:
        measures |= {
            f"P_{k}",
            f"recall_{k}",
            f"map_cut_{k}",
            f"binary_ndcg_cut_{k}",
            f"graded_ndcg_cut_{k}",
        }

    if extra_measures:
        measures |= set(extra_measures)

    return measures, per_query
    
def aggregate_metrics(per_query: Dict[str, Dict[str, float]], measures: set) -> dict:
    agg = {}
    for m in measures:
        vals = [per_query[qid].get(m) for qid in per_query if m in per_query[qid]]
        agg[m] = float(sum(vals) / len(vals)) if vals else 0.0
    agg["num_q"] = float(len(per_query))
    return agg

def compute_exposure_metrics(run, creator_prior_attention, ks):
    """
    Paper-consistent exposure metrics using g(x)=log(1+x):

      Exp@k  := log(1 + (1/k) * sum_{i=1..k} a_i)
      DExp@k := log(1 + sum_{i=1..k} a_i / log2(i+1))

    We store them into existing columns:
      Exp@k  (as Exp@k)
      DExp@k (as DExp@k)
    """
    avg = {}
    disc = {}
    if creator_prior_attention is None:
        return {}, {}

    for k in ks:
        exp_vals = []
        dexp_vals = []

        for qid, docs in run.items():
            ranked = sorted(docs.items(), key=lambda x: x[1], reverse=True)[:k]
            a_vals = [float(creator_prior_attention.get(docid, 0.0)) for docid, _ in ranked]

            # If fewer than k docs exist, pad with zeros to keep 1/k semantics
            if len(a_vals) < k:
                a_vals += [0.0] * (k - len(a_vals))

            # Exp@k = g( (1/k) * sum a_i )
            mean_a = sum(a_vals) / k
            exp_vals.append(math.log1p(mean_a))

            # DExp@k = g( sum a_i / log2(i+1) )
            disc_sum = sum(
                a / math.log2(i + 2) for i, a in enumerate(a_vals)  # i=0 -> log2(2)=1
            )
            dexp_vals.append(math.log1p(disc_sum))

        avg[f"Exp@{k}"] = sum(exp_vals) / len(exp_vals) if exp_vals else 0.0
        disc[f"DExp@{k}"] = sum(dexp_vals) / len(dexp_vals) if dexp_vals else 0.0

    return avg, disc


def add_exposure_metrics_to_agg(agg, run, creator_prior_attention, ks):
    """
    Compute prior-attention exposure metrics and merge them into agg.
    """
    if creator_prior_attention is None:
        return agg

    average, discounted = compute_exposure_metrics(run, creator_prior_attention, ks)
    agg.update(average)
    agg.update(discounted)
    return agg

def compute_exposure_bias_at_k(
    run: Dict[str, Dict[str, float]],
    qid_to_labels: Dict[str, Dict[str, float]],
    creator_prior_attention: Dict[str, float],
    ks: Tuple[int, ...],
    log_attr: bool = True,
) -> Dict[str, float]:
    """
    ExposureBias@k = mean_q [S_model_k(q) - S_gt_k(q)]
    where
      S_k(q) = sum_{i=1..k} f(a_i) / log(i+1)

    Ground truth ranking is label-descending.
    Model ranking is score-descending.
    """
    out = {}

    def f_attr(docid: str) -> float:
        y = float(creator_prior_attention.get(docid, 0.0))
        return math.log1p(y) if log_attr else y

    for k in ks:
        per_query_vals = []

        for qid, docscores in run.items():
            if qid not in qid_to_labels:
                continue

            label_map = qid_to_labels[qid]

            # model ranking
            ranked_model = sorted(docscores.items(), key=lambda x: x[1], reverse=True)
            ranked_model_docids = [docid for docid, _ in ranked_model[:k]]

            # ground-truth ranking from graded labels
            ranked_gt = sorted(label_map.items(), key=lambda x: x[1], reverse=True)
            ranked_gt_docids = [docid for docid, lab in ranked_gt[:k]]

            s_model = 0.0
            for i, docid in enumerate(ranked_model_docids, start=1):
                s_model += f_attr(docid) / math.log(i + 1)

            s_gt = 0.0
            for i, docid in enumerate(ranked_gt_docids, start=1):
                s_gt += f_attr(docid) / math.log(i + 1)

            per_query_vals.append(s_model - s_gt)

        out[f"ExposureBias@{k}"] = (
            sum(per_query_vals) / len(per_query_vals) if per_query_vals else 0.0
        )

    return out


def compute_exposure_top1_bias(
    run: Dict[str, Dict[str, float]],
    qid_to_labels: Dict[str, Dict[str, float]],
    creator_prior_attention: Dict[str, float],
    log_attr: bool = True,
) -> Dict[str, float]:
    """
    ExposureTop1Bias:
      among misclassified queries, fraction where predicted top-1
      has higher prior attention than the label-best top-1.

    Ground-truth top-1 = highest graded label.
    Predicted top-1    = highest model score.
    """
    wrong_count = 0
    biased_wrong_count = 0

    def f_attr(docid: str) -> float:
        y = float(creator_prior_attention.get(docid, 0.0))
        return math.log1p(y) if log_attr else y

    for qid, docscores in run.items():
        if qid not in qid_to_labels:
            continue

        label_map = qid_to_labels[qid]
        if not docscores or not label_map:
            continue

        pred_top = max(docscores.items(), key=lambda x: x[1])[0]
        gt_top = max(label_map.items(), key=lambda x: x[1])[0]

        if pred_top != gt_top:
            wrong_count += 1
            if f_attr(pred_top) > f_attr(gt_top):
                biased_wrong_count += 1

    bis_val = (biased_wrong_count / wrong_count) if wrong_count > 0 else 0.0

    return {
        "ExposureTop1Bias": bis_val,
        "ExposureTop1BiasNumWrong": float(wrong_count),
    }


def add_bias_metrics_to_agg(
    agg: dict,
    run: Dict[str, Dict[str, float]],
    qid_to_labels: Dict[str, Dict[str, float]],
    creator_prior_attention: Optional[Dict[str, float]],
    ks: Tuple[int, ...],
) -> dict:
    """
    Add prior-attention bias metrics into agg.
    """
    if creator_prior_attention is None:
        return agg

    ibs = compute_exposure_bias_at_k(
        run=run,
        qid_to_labels=qid_to_labels,
        creator_prior_attention=creator_prior_attention,
        ks=ks,
        log_attr=True,
    )
    bis = compute_exposure_top1_bias(
        run=run,
        qid_to_labels=qid_to_labels,
        creator_prior_attention=creator_prior_attention,
        log_attr=True,
    )

    agg.update(ibs)
    agg.update(bis)
    return agg

# -----------------------------
# Saving CSVs
# -----------------------------
def save_metrics_csv(
    per_query: Dict[str, Dict[str, float]],
    model_name: str,
    meta: Optional[dict],
    output_csv: str,
    ks=(2, 5, 10),
) -> None:
    """
    Save per-query metrics table.
    """
    rows = []
    for qid, metrics in per_query.items():
        row = {
            "query_id": qid,
            "model": model_name,
        }
        if meta:
            row.update({
                "template_version": meta.get("template_name", "grouped"),
                "has_neg": meta.get("has_neg", False),
                "has_score": meta.get("has_score", True),
        
                "loss_type": meta.get("loss_type", "unknown"),
                "training_peft_method": meta.get("training_peft_method", "none"),
                "training_lambda_corr": meta.get("training_lambda_corr", 0.0),
                "training_lambda_prior_attention": meta.get("training_lambda_prior_attention", 0.0),
            })
        else:
            row.update({
                "template_version": "grouped",
                "has_neg": False,
                "has_score": True,
            })

        # Add metrics in your preferred format: P@k, R@k, NDCG@k, MAP@
        for k in ks:
            row[f"P@{k}"] = metrics.get(f"P_{k}", 0.0)
            row[f"R@{k}"] = metrics.get(f"recall_{k}", 0.0)
            row[f"BinaryNDCG@{k}"] = metrics.get(f"binary_ndcg_cut_{k}", 0.0)
            row[f"GradedNDCG@{k}"] = metrics.get(f"graded_ndcg_cut_{k}", 0.0)
            row[f"MAP@{k}"] = metrics.get(f"map_cut_{k}", 0.0)
            row[f"SkillCoverage@{k}"] = metrics.get(f"skill_coverage_{k}", 0.0)
            


        # If you include extra measures like map/recip_rank, store them too
        #if "map" in metrics:
        #    row["map"] = metrics.get("map", 0.0)
        if "recip_rank" in metrics:
            row["mrr"] = metrics.get("recip_rank", 0.0)

        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Per-query metrics saved to {output_csv}")


def append_agg_metrics_csv(
    agg: dict,
    ks: tuple,
    output_csv: str,
) -> None:
    """
    Append one aggregated row per run to output_csv.
    
    """
   

    row = {
        "run_time": agg.get("run_time", "unknown"),
        "model_name": agg.get("model_name_or_path", "unknown"),
        "loss_type": agg.get("loss_type", "unknown"),
        "skill_set_version": agg.get("skill_set_version", "unknown"),
        "template_version": agg.get("template_name", "unknown"),
        "ks": ",".join(map(str, ks)),
        "num_q": agg.get("num_q", 0.0),
    
        # training-time metadata
        "training_peft_method": agg.get("training_peft_method", "none"),
        "training_lambda_corr": agg.get("training_lambda_corr", 0.0),
        "training_lambda_prior_attention": agg.get("training_lambda_prior_attention", 0.0),
    }



    # metrics in strict order
    for k in ks:
        row[f"P_{k}"] = agg.get(f"P_{k}", 0.0)
    for k in ks:
        row[f"recall_{k}"] = agg.get(f"recall_{k}", 0.0)
    for k in ks:
        row[f"binary_ndcg_cut_{k}"] = agg.get(f"binary_ndcg_cut_{k}", 0.0)
    for k in ks:
        row[f"graded_ndcg_cut_{k}"] = agg.get(f"graded_ndcg_cut_{k}", 0.0)
    for k in ks:
        row[f"map_cut_{k}"] = agg.get(f"map_cut_{k}", 0.0)  
    for k in ks:
        row[f"skill_coverage_{k}"] = agg.get(f"skill_coverage_{k}", 0.0)
    for k in ks:
        row[f"Exp@{k}"] = agg.get(f"Exp@{k}", 0.0)
        row[f"DExp@{k}"] = agg.get(f"DExp@{k}", 0.0)
        row[f"ExposureBias@{k}"] = agg.get(f"ExposureBias@{k}", 0.0)

    row["ExposureTop1Bias"] = agg.get("ExposureTop1Bias", 0.0)
    row["ExposureTop1BiasNumWrong"] = agg.get("ExposureTop1BiasNumWrong", 0.0)

    
    if "recip_rank" in agg:
        row["recip_rank"] = agg.get("recip_rank", 0.0)

    df_new = pd.DataFrame([row])

    if os.path.exists(output_csv):
        df_old = pd.read_csv(output_csv)
    
        # preserve old column order
        old_cols = list(df_old.columns)
        new_cols = [c for c in df_new.columns if c not in old_cols]
    
        # append new columns at the END
        df = pd.concat([df_old, df_new], ignore_index=True)
    
        # explicitly enforce column order
        df = df[old_cols + new_cols]
    else:
        df = df_new


    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"📈 Aggregated metrics appended to {output_csv}")

# -----------------------------
# Debug / sanity helpers
# -----------------------------
def analyze_run_binary(qrels: dict, run: dict) -> None:
    """
    Works with binary qrels (label>0 -> 1).
    Helpful sanity checks.
    """
    num_q = len(run)
    if num_q == 0:
        print("⚠️ No queries in run.")
        return

    docs_per_q = []
    rel_in_run = []
    nonrel_in_run = []
    zero_hit = 0

    for qid in run:
        retrieved = set(run[qid].keys())
        relevant = set(qrels.get(qid, {}).keys())

        docs_per_q.append(len(retrieved))
        rel_count = len(retrieved & relevant)
        nonrel_count = len(retrieved - relevant)

        rel_in_run.append(rel_count)
        nonrel_in_run.append(nonrel_count)

        if rel_count == 0:
            zero_hit += 1

    print("⚙️ RUN ANALYSIS (binary qrels)")
    print(f"  Queries evaluated         : {num_q}")
    print(f"  Avg docs / query          : {sum(docs_per_q)/num_q:.2f}")
    print(f"  Avg relevant retrieved    : {sum(rel_in_run)/num_q:.2f}")
    print(f"  Avg non-relevant retrieved: {sum(nonrel_in_run)/num_q:.2f}")
    print(f"  Queries with 0 relevant   : {zero_hit} ({100*zero_hit/num_q:.1f}%)")
    print("")
    

def _load_config_file(path: Path) -> Optional[dict]:
    try:
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if path.suffix in {".yml", ".yaml"}:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
    except Exception:
        return None
    return None


def _candidate_run_metadata_paths(model_path: Path) -> List[Path]:
    candidates: List[Path] = []
    seen = set()

    for base in [model_path] + list(model_path.parents):
        for candidate in (
            base / "run_config.json",
            base / "model" / "run_config.json",
        ):
            if candidate.exists() and candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

        ranker_dir = base / "ranker"
        if ranker_dir.exists():
            hparams_paths = sorted(
                ranker_dir.glob("*/hparams.yml"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for candidate in hparams_paths:
                if candidate.exists() and candidate not in seen:
                    candidates.append(candidate)
                    seen.add(candidate)

    return candidates


def load_run_config_or_default(model_arg: str) -> dict:
    """
    Resolve run metadata for either a final exported model directory or an
    intermediate checkpoint directory.

    Lookup order:
    1) <model_arg>/run_config.json
    2) ancestor/model/run_config.json
    3) ancestor/ranker/*/hparams.yml

    If nothing usable is found, return defaults.
    """
    defaults = {
        "template_name": "v2",
        "loss_type": "unknown",
        "model_name_or_path": model_arg,
        "model_type": "cross-encoder",
        "run_time": "unknown",
        "skill_set_version": "rootdata",
        "lambda_corr": 0.0,
        "lambda_prior_attention": 0.0,
    }

    model_path = Path(model_arg)

    if not model_path.exists():
        return defaults

    for cfg_path in _candidate_run_metadata_paths(model_path):
        cfg = _load_config_file(cfg_path)
        if isinstance(cfg, dict):
            return {**defaults, **cfg}

    return defaults
