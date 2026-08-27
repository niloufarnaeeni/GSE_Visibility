import json
from pathlib import Path
from typing import Union, List, Tuple, Optional
import numpy as np
import torch
import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from .ranker import BaseRanker
from .result import RankedResults, Result
from .utils import get_device, get_dtype, vprint


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _resolve_head_out_dim(model):
    candidate_paths = [
        "score",
        "classifier",
        "base_model.score",
        "base_model.classifier",
        "base_model.model.score",
        "base_model.model.classifier",
    ]

    for path in candidate_paths:
        module = model
        found = True
        for attr in path.split("."):
            if not hasattr(module, attr):
                found = False
                break
            module = getattr(module, attr)
        if found and hasattr(module, "weight"):
            return int(module.weight.shape[0])

    return None


def _normalize_device_map(device_map: Optional[str]):
    if device_map is None:
        return None
    normalized = str(device_map).strip()
    if not normalized:
        return None
    if normalized.lower() == "none":
        return None
    return normalized


def _first_cuda_device_from_map(hf_device_map) -> Optional[str]:
    if not hf_device_map:
        return None

    for target in hf_device_map.values():
        if isinstance(target, int):
            return f"cuda:{target}"
        if isinstance(target, str) and target.startswith("cuda"):
            return target
    return None


def _resolve_input_device(model, fallback_device: Optional[str]) -> torch.device:
    mapped = _first_cuda_device_from_map(getattr(model, "hf_device_map", None))
    if mapped is not None:
        return torch.device(mapped)

    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)

    if fallback_device is not None:
        return torch.device(fallback_device)

    return torch.device("cpu")


class LLMDecoderRanker(BaseRanker):
    def __init__(
        self,
        model_name_or_path: str,
        dtype: str = None,
        device: str = None,
        verbose: int = 1,
        query_format: str = "query: {}",
        document_format: str = "document: {}",
        seq: str = "\n",
        special_token: str = "\nrelevance",
        device_map: str = None,
    ):
        self.verbose = verbose
        self.model_name_or_path = model_name_or_path
        self.device_map = _normalize_device_map(device_map)
        self.device = get_device(device, verbose=self.verbose)
        self.dtype = get_dtype(dtype, device=self.device, verbose=self.verbose)

        self.query_format = query_format
        self.document_format = document_format
        self.seq = seq
        self.special_token = special_token

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )

        adapter_config_path = Path(model_name_or_path) / "adapter_config.json"
        run_config_path = Path(model_name_or_path) / "run_config.json"
        peft_num_labels = 1
        if run_config_path.exists():
            try:
                run_config = json.loads(run_config_path.read_text())
                peft_num_labels = int(run_config.get("num_labels", 1))
            except Exception:
                peft_num_labels = 1

        if adapter_config_path.exists():
            if PeftModel is None:
                raise ImportError("peft is required to load LoRA/QLoRA adapter checkpoints.")

            adapter_config = json.loads(adapter_config_path.read_text())
            base_model_name_or_path = adapter_config.get("base_model_name_or_path")
            if not base_model_name_or_path:
                raise ValueError(
                    f"adapter_config.json at {adapter_config_path} is missing "
                    "'base_model_name_or_path'."
                )

            base_config = AutoConfig.from_pretrained(
                base_model_name_or_path,
                trust_remote_code=True,
            )
            base_config.num_labels = peft_num_labels

            base_model_kwargs = {
                "config": base_config,
                "torch_dtype": self.dtype,
                "trust_remote_code": True,
            }
            if self.device_map is not None:
                base_model_kwargs["device_map"] = self.device_map
                base_model_kwargs["low_cpu_mem_usage"] = True

            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name_or_path,
                **base_model_kwargs,
            )

            peft_kwargs = {}
            if self.device_map is not None:
                peft_kwargs["device_map"] = self.device_map

            self.model = PeftModel.from_pretrained(
                base_model,
                model_name_or_path,
                **peft_kwargs,
            )
            if self.device_map is None:
                self.model = self.model.to(self.device)
        else:
            base_config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
            base_config.num_labels = peft_num_labels
            base_model_kwargs = {
                "config": base_config,
                "torch_dtype": self.dtype,
                "trust_remote_code": True,
            }
            if self.device_map is not None:
                base_model_kwargs["device_map"] = self.device_map
                base_model_kwargs["low_cpu_mem_usage"] = True

            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                **base_model_kwargs,
            )
            if self.device_map is None:
                self.model = self.model.to(self.device)

        if hasattr(self.tokenizer, "deprecation_warnings"):
            self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.model.resize_token_embeddings(len(self.tokenizer))

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.tokenizer.padding_side = "right"

        self.special_token_ids = self.tokenizer.encode(
            self.special_token,
            add_special_tokens=False,
        )
        self.sep_token_ids = self.tokenizer.encode(
            self.seq,
            add_special_tokens=False,
        )

        model_dtype = next(self.model.parameters()).dtype
        self.input_device = _resolve_input_device(self.model, self.device)
        vprint(f"Loaded model {self.model_name_or_path}", self.verbose)
        vprint(f"model_dtype is {model_dtype}", self.verbose)
        vprint(f"input_device is {self.input_device}", self.verbose)
        if self.device_map is not None:
            vprint(f"device_map is {self.device_map}", self.verbose)

        output_dim = getattr(self.model.config, "num_labels", None)
        if output_dim != 1:
            raise ValueError(
                f"Expected evaluation model with num_labels=1, got num_labels={output_dim} "
                f"for {self.model_name_or_path}."
            )

        score_out_dim = _resolve_head_out_dim(self.model)
        if score_out_dim is not None and score_out_dim != 1:
            raise ValueError(
                f"Expected score head output dim 1, got {score_out_dim} "
                f"for {self.model_name_or_path}."
            )

        self.model.eval()

    def preprocess(self, sentence_pairs, max_len):
        features = []

        for query, document in sentence_pairs:
            query_ids = self.tokenizer.encode(
                self.query_format.format(query.strip()),
                add_special_tokens=False,
            )

            document_max_len = (
                max_len
                - len(query_ids)
                - len(self.sep_token_ids)
                - len(self.special_token_ids)
            )

            if document_max_len <= 0:
                raise ValueError(
                    f"max_len={max_len} is too small for the formatted query and special tokens."
                )

            document_ids = self.tokenizer.encode(
                self.document_format.format(document.strip()),
                add_special_tokens=False,
                truncation=True,
                max_length=document_max_len,
            )

            input_ids = (
                query_ids
                + self.sep_token_ids
                + document_ids
                + self.special_token_ids
            )

            features.append({"input_ids": input_ids})

        tokens = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        return tokens

    @torch.no_grad()
    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 16,
        max_length: int = 320,
        normalize: bool = False,
        enable_tqdm: bool = True,
    ):
        if isinstance(sentence_pairs, tuple):
            sentence_pairs = [sentence_pairs]

        all_scores = []

        for start_index in tqdm.tqdm(
            range(0, len(sentence_pairs), batch_size),
            desc="Compute Scores",
            disable=not enable_tqdm,
        ):
            batch_pairs = sentence_pairs[start_index:start_index + batch_size]
            inputs = self.preprocess(batch_pairs, max_length)
            inputs = {k: v.to(self.input_device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            scores = outputs.logits.view(-1).detach().cpu().float().tolist()
            all_scores.extend(scores)

        if normalize:
            all_scores = [sigmoid(x) for x in all_scores]

        if len(all_scores) == 1:
            return all_scores[0]

        return all_scores

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        docs: Union[List[str], str],
        batch_size: int = 16,
        normalize: bool = False,
        max_length: int = 256,
    ):
        docs = [doc for doc in docs if isinstance(doc, str) and len(doc) > 0]

        if not query or not docs:
            return RankedResults(results=[], query=query, has_scores=True)

        sentence_pairs = [[query, doc] for doc in docs]
        scores = self.compute_score(
            sentence_pairs,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
            enable_tqdm=False,
        )

        ranked_results = [
            Result(doc_id=i, text=doc, score=score, rank=rank + 1)
            for rank, (i, doc, score) in enumerate(
                sorted(
                    zip(range(len(docs)), docs, scores),
                    key=lambda x: x[2],
                    reverse=True,
                )
            )
        ]

        return RankedResults(results=ranked_results, query=query, has_scores=True)
