import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tqdm

from rag_retrieval.train.reranker import ranking_loss


class CrossEncoder(nn.Module):
    def __init__(self, hf_model=None, tokenizer=None, loss_type="ranknet", query_format="{}", document_format="{}"):
        super().__init__()
        self.model, self.tokenizer, self.loss_type = hf_model, tokenizer, loss_type
        self.query_format, self.document_format = query_format, document_format
        self.train_group_size = None
        self.lambda_prior_attention = 0.0
        self.lambda_corr = 0.05

    def forward(self, batch, labels=None):
        output = self.model(**batch)
        logits = output.logits.view(-1)
        if labels is None:
            return output
        prior_attention = None
        if isinstance(labels, dict):
            prior_attention, labels = labels.get("prior_attention"), labels.get("labels")
        if labels is None:
            raise ValueError("labels is None after unpacking. Check your dataset/collate_fn.")
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
    def compute_score(self, sentences_pairs, batch_size=256, max_length=512, normalize=False):
        all_logits = []
        for start_index in tqdm.tqdm(range(0, len(sentences_pairs), batch_size)):
            batch_data = self.preprocess(sentences_pairs[start_index:start_index + batch_size], max_length).to(self.model.device)
            all_logits.extend(self.forward(batch_data).logits.detach().cpu().view(-1).tolist())
        return torch.sigmoid(torch.tensor(all_logits)).tolist() if normalize else all_logits

    def preprocess(self, sentences_pairs, max_len):
        pairs = [[self.query_format.format(q.strip()), self.document_format.format(d.strip())] for q, d in sentences_pairs]
        return self.tokenizer(pairs, add_special_tokens=True, padding="longest", max_length=max_len, truncation="only_second", return_tensors="pt")

    @classmethod
    def from_pretrained(cls, model_name_or_path, loss_type="ranknet", num_labels=1, query_format="{}", document_format="{}"):
        return cls(AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels, trust_remote_code=True), AutoTokenizer.from_pretrained(model_name_or_path), loss_type, query_format, document_format)

    def save_pretrained(self, save_dir, state_dict=None, safe_serialization=False):
        if state_dict is None:
            state_dict = self.model.state_dict()
        state_dict = type(state_dict)({key: value.detach().clone().cpu() for key, value in state_dict.items()})
        self.model.save_pretrained(save_dir, state_dict=state_dict, safe_serialization=safe_serialization)
