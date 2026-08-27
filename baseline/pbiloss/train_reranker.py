"""Isolated PBiLoss training entrypoint."""

from rag_retrieval.baseline.pbiloss.loss import pbiloss_popneg_ft
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


def _parse_args_for_pbiloss():
    global BASELINE_CONFIG
    args = _ORIGINAL_PARSE_ARGS()
    if args.loss_type != "pbiloss_popneg_ft":
        raise ValueError(
            "The isolated PBiLoss trainer expects runtime config "
            "loss_type='pbiloss_popneg_ft'."
        )
    if not isinstance(getattr(args, "baseline", None), dict):
        raise ValueError("The isolated PBiLoss trainer requires a baseline config.")
    BASELINE_CONFIG = args.baseline
    return args


def _attach_pbiloss_config(model):
    model.baseline_config = BASELINE_CONFIG
    model.loss_type = "pbiloss_popneg_ft"
    return model


def _ce_from_pretrained(cls, *args, **kwargs):
    return _attach_pbiloss_config(_ORIGINAL_CE_FROM_PRETRAINED(*args, **kwargs))


def _llm_from_pretrained(cls, *args, **kwargs):
    return _attach_pbiloss_config(_ORIGINAL_LLM_FROM_PRETRAINED(*args, **kwargs))


def _pbiloss_forward(self, batch, labels=None):
    if self.loss_type != "pbiloss_popneg_ft":
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
        raise ValueError("PBiLoss requires grouped labels and prior_attention.")

    output["loss"] = pbiloss_popneg_ft(
        logits=logits,
        labels=labels,
        prior_attention=prior_attention,
        group_size=self.train_group_size,
        baseline_config=self.baseline_config,
    )
    return output


def main():
    core_train.parse_args = _parse_args_for_pbiloss
    core_train.GroupedRankerDataset = GroupedRankerDatasetWithPriorAttention
    CrossEncoder.forward = _pbiloss_forward
    LLMDecoder.forward = _pbiloss_forward
    CrossEncoder.from_pretrained = classmethod(_ce_from_pretrained)
    LLMDecoder.from_pretrained = classmethod(_llm_from_pretrained)
    core_train.main()


if __name__ == "__main__":
    main()
