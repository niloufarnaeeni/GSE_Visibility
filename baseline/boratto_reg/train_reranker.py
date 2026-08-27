"""Isolated Boratto-reg training entrypoint."""

from rag_retrieval.baseline.boratto_reg.loss import pairwise_boratto_reg_prior_attention
from rag_retrieval.train.reranker import train_reranker as core_train
from rag_retrieval.train.reranker.data import GroupedRankerDatasetWithPriorAttention
from rag_retrieval.train.reranker.model_bert import CrossEncoder
from rag_retrieval.train.reranker.model_llm import LLMDecoder


BASELINE_CONFIG = None

_ORIGINAL_PARSE_ARGS = core_train.parse_args
_ORIGINAL_CE_FORWARD = CrossEncoder.forward
_ORIGINAL_LLM_FORWARD = LLMDecoder.forward
_ORIGINAL_CE_FROM_PRETRAINED = CrossEncoder.from_pretrained
_ORIGINAL_LLM_FROM_PRETRAINED = LLMDecoder.from_pretrained


def _parse_args_for_boratto():
    global BASELINE_CONFIG
    args = _ORIGINAL_PARSE_ARGS()
    if args.loss_type != "boratto_reg":
        raise ValueError(
            "The isolated Boratto-reg trainer expects runtime config "
            "loss_type='boratto_reg'."
        )
    if not isinstance(getattr(args, "baseline", None), dict):
        raise ValueError("The isolated Boratto-reg trainer requires a baseline config.")
    BASELINE_CONFIG = args.baseline
    return args


def _attach_boratto_config(model):
    model.baseline_config = BASELINE_CONFIG
    model.loss_type = "boratto_reg"
    return model


def _ce_from_pretrained(cls, *args, **kwargs):
    return _attach_boratto_config(_ORIGINAL_CE_FROM_PRETRAINED(*args, **kwargs))


def _llm_from_pretrained(cls, *args, **kwargs):
    return _attach_boratto_config(_ORIGINAL_LLM_FROM_PRETRAINED(*args, **kwargs))


def _boratto_forward(self, batch, labels=None):
    if self.loss_type != "boratto_reg":
        original_forward = (
            _ORIGINAL_CE_FORWARD
            if type(self) is CrossEncoder
            else _ORIGINAL_LLM_FORWARD
        )
        return original_forward(self, batch, labels)

    output = self.model(**batch)
    logits = output.logits.view(-1)
    if labels is None:
        return output

    prior_attention = None
    if isinstance(labels, dict):
        prior_attention = labels.get("prior_attention")
        labels = labels.get("labels")

    if labels is None or prior_attention is None:
        raise ValueError("Boratto-reg requires grouped labels and prior_attention.")

    output["loss"] = pairwise_boratto_reg_prior_attention(
        logits=logits,
        labels=labels,
        prior_attention=prior_attention,
        group_size=self.train_group_size,
        lambda_corr=float(self.baseline_config["lambda_corr"]),
    )
    return output


def main():
    core_train.parse_args = _parse_args_for_boratto
    core_train.GroupedRankerDataset = GroupedRankerDatasetWithPriorAttention
    CrossEncoder.forward = _boratto_forward
    LLMDecoder.forward = _boratto_forward
    CrossEncoder.from_pretrained = classmethod(_ce_from_pretrained)
    LLMDecoder.from_pretrained = classmethod(_llm_from_pretrained)
    core_train.main()


if __name__ == "__main__":
    main()
