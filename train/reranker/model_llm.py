import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import tqdm
from pathlib import Path

from rag_retrieval.train.reranker import ranking_loss

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


def _resolve_torch_dtype(dtype_name):
    if dtype_name is None:
        return None

    if isinstance(dtype_name, torch.dtype):
        return dtype_name

    normalized = str(dtype_name).strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[normalized]


def _normalize_target_modules(target_modules):
    if target_modules is None:
        return None

    if isinstance(target_modules, str):
        modules = [item.strip() for item in target_modules.split(",") if item.strip()]
        return modules or None

    if isinstance(target_modules, (list, tuple)):
        modules = [str(item).strip() for item in target_modules if str(item).strip()]
        return modules or None

    raise TypeError("lora_target_modules must be None, a comma-separated string, or a list/tuple")


def _default_lora_target_modules():
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]


def _maybe_enable_gradient_checkpointing(hf_model):
    if hasattr(hf_model, "gradient_checkpointing_enable"):
        hf_model.gradient_checkpointing_enable()
    if hasattr(hf_model, "enable_input_require_grads"):
        hf_model.enable_input_require_grads()


def _build_lora_model(
    hf_model,
    *,
    peft_method,
    lora_r,
    lora_alpha,
    lora_dropout,
    lora_target_modules,
    gradient_checkpointing,
):
    if LoraConfig is None or get_peft_model is None or TaskType is None:
        raise ImportError("peft is required for LoRA/QLoRA training but is not installed.")

    if gradient_checkpointing:
        _maybe_enable_gradient_checkpointing(hf_model)

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=lora_target_modules or _default_lora_target_modules(),
        bias="none",
    )
    hf_model = get_peft_model(hf_model, config)
    print(
        f"[PEFT] Enabled {peft_method.upper()} with final target_modules="
        f"{list(config.target_modules)}"
    )
    hf_model.print_trainable_parameters()
    return hf_model


class LLMDecoder(nn.Module):
    def __init__(
        self,
        hf_model=None,
        tokenizer=None,
        loss_type="ranknet",
        query_format="{}",
        document_format="{}",
        seq="",
        special_token="",
    ):
        super().__init__()

        self.model = hf_model
        self.tokenizer = tokenizer
        self.loss_type = loss_type
        self.train_group_size = None  # set in train_reranker.py
        self.lambda_prior_attention = 0.0
        self.lambda_corr = 0.05

        self.query_format = query_format
        self.document_format = document_format
        self.seq = seq
        self.special_token = special_token

        self.special_token_ids = self.tokenizer.encode(
            self.special_token,
            add_special_tokens=False,
        )
        self.sep_token_ids = self.tokenizer.encode(
            self.seq,
            add_special_tokens=False,
        )

    def forward(self, batch, labels=None):
        output = self.model(**batch)

        # keep shape stable for batch_size=1
        logits = output.logits.view(-1)

        # inference mode
        if labels is None:
            return output

        # Grouped prior-attention dataset support.
        prior_attention = None
        if isinstance(labels, dict):
            prior_attention = labels.get("prior_attention", None)
            labels = labels.get("labels", None)

        if labels is None:
            raise ValueError(
                "labels is None after unpacking. Check your dataset/collate_fn."
            )

        if self.loss_type == "ranknet":
            loss = ranking_loss.ranknet(logits, labels, self.train_group_size)
        elif self.loss_type in {"ear", "ear_sym", "pairwise_reg"}:
            if prior_attention is None:
                raise ValueError(f"{self.loss_type} requires prior_attention.")
            if self.loss_type == "ear":
                loss = ranking_loss.ear(logits, labels, prior_attention, self.train_group_size, self.lambda_prior_attention)
            elif self.loss_type == "ear_sym":
                loss = ranking_loss.ear_sym(logits, labels, prior_attention, self.train_group_size, self.lambda_prior_attention)
            else:
                loss = ranking_loss.pairwise_reg(logits, labels, prior_attention, self.train_group_size, self.lambda_corr)
        else:
            raise ValueError(f"Unknown ranking objective: {self.loss_type}")

        output["loss"] = loss
        return output

    @torch.no_grad()
    def compute_score(
        self,
        sentences_pairs,
        batch_size=256,
        max_length=512,
        normalize=False,
    ):
        """
        sentences_pairs = [[query, document], [query1, document1], ...]
        """
        all_logits = []

        for start_index in tqdm.tqdm(range(0, len(sentences_pairs), batch_size)):
            sentences_batch = sentences_pairs[start_index : start_index + batch_size]
            batch_data = self.preprocess(sentences_batch, max_length).to(
                self.model.device
            )

            output = self.forward(batch_data)
            logits = output.logits.detach().cpu().view(-1).tolist()
            all_logits.extend(logits)

        if normalize:
            all_logits = torch.sigmoid(torch.tensor(all_logits)).cpu().tolist()

        return all_logits

    def preprocess(self, sentences_pairs, max_len):
        features = []

        for query, document in sentences_pairs:
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

        # safer than prepare_for_model(...) for Qwen tokenizer
        tokens = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        return tokens

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        loss_type="ranknet",
        num_labels=1,
        query_format="query: {}",
        document_format="document: {}",
        seq=" ",
        special_token="</s>",
        peft_method="none",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None,
        gradient_checkpointing=False,
        qlora_compute_dtype="bfloat16",
        qlora_quant_type="nf4",
        qlora_use_double_quant=True,
    ):
        peft_method = str(peft_method).strip().lower()
        if peft_method not in {"none", "lora", "qlora"}:
            raise ValueError(f"Unsupported peft_method: {peft_method}")

        print(f"Gradient checkpointing: {gradient_checkpointing}")

        lora_target_modules = _normalize_target_modules(lora_target_modules)
        final_lora_target_modules = lora_target_modules or _default_lora_target_modules()

        model_kwargs = {
            "num_labels": num_labels,
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }
        if peft_method == "qlora":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_resolve_torch_dtype(qlora_compute_dtype),
                bnb_4bit_quant_type=str(qlora_quant_type),
                bnb_4bit_use_double_quant=bool(qlora_use_double_quant),
            )
            model_kwargs["device_map"] = None

        hf_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
        )

        # avoid tokenizer warning crash on some versions
        if hasattr(tokenizer, "deprecation_warnings"):
            tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                hf_model.resize_token_embeddings(len(tokenizer))

        if hf_model.config.pad_token_id is None:
            hf_model.config.pad_token_id = tokenizer.pad_token_id

        if hasattr(hf_model.config, "use_cache"):
            hf_model.config.use_cache = False

        if peft_method == "qlora":
            if prepare_model_for_kbit_training is None:
                raise ImportError("peft is required for QLoRA training but is not installed.")
            print(
                f"[PEFT] Final LoRA target modules for training: "
                f"{final_lora_target_modules}"
            )
            hf_model = prepare_model_for_kbit_training(
                hf_model,
                use_gradient_checkpointing=bool(gradient_checkpointing),
            )
            hf_model = _build_lora_model(
                hf_model,
                peft_method=peft_method,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=final_lora_target_modules,
                gradient_checkpointing=gradient_checkpointing,
            )
        elif peft_method == "lora":
            print(
                f"[PEFT] Final LoRA target modules for training: "
                f"{final_lora_target_modules}"
            )
            hf_model = _build_lora_model(
                hf_model,
                peft_method=peft_method,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=final_lora_target_modules,
                gradient_checkpointing=gradient_checkpointing,
            )
        elif gradient_checkpointing:
            _maybe_enable_gradient_checkpointing(hf_model)

        # right padding is important for decoder-only models
        tokenizer.padding_side = "right"

        reranker = cls(
            hf_model=hf_model,
            tokenizer=tokenizer,
            loss_type=loss_type,
            query_format=query_format,
            document_format=document_format,
            seq=seq,
            special_token=special_token,
        )
        return reranker

    def save_pretrained(self, save_dir, state_dict=None, safe_serialization=True):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()

        if hasattr(self.model, "peft_config"):
            # For PEFT/QLoRA models, let PEFT collect adapter weights from the
            # wrapped model directly. Passing the accelerator-gathered
            # full-model state_dict here can break PEFT's modules_to_save lookup
            # during adapter export.
            self.model.save_pretrained(
                str(save_dir),
                safe_serialization=safe_serialization,
            )
            return

        def _trans_state_dict(raw_state_dict):
            return type(raw_state_dict)(
                {k: v.detach().clone().cpu() for k, v in raw_state_dict.items()}
            )

        self.model.save_pretrained(
            str(save_dir),
            state_dict=_trans_state_dict(state_dict),
            safe_serialization=safe_serialization,
        )


def test_LLMDecoder():
    ckpt_path = "./Qwen2-1.5B-Instruct"

    reranker = LLMDecoder.from_pretrained(
        model_name_or_path=ckpt_path,
        num_labels=1,
        query_format="query: {}",
        document_format="document: {}",
        seq=" ",
        special_token="</s>",
    )

    reranker.model.to("cuda:0")
    reranker.eval()

    input_lst = [
        ["我喜欢中国", "我喜欢中国"],
        ["我喜欢美国", "我一点都不喜欢美国"],
        [
            "泰山要多长时间爬上去",
            "爬上泰山需要1-8个小时，具体的时间需要看个人的身体素质。专业登山运动员可能只需要1个多小时就可以登顶，有些身体素质比较低的，爬得慢的就需要5个多小时了。",
        ],
    ]

    res = reranker.compute_score(input_lst)

    print(torch.sigmoid(torch.tensor(res[0])))
    print(torch.sigmoid(torch.tensor(res[1])))
    print(torch.sigmoid(torch.tensor(res[2])))


if __name__ == "__main__":
    test_LLMDecoder()
