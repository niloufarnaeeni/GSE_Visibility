# Training

Core training supports only these ranking objectives:

- `ranknet` — RankNet
- `ear` — EAR
- `ear_sym` — EAR-Sym
- `pairwise_reg` — Pairwise Reg

The grouped training schema uses `prior_attention` for the creator prior-attention signal.

Example:

```bash
python train_eval.py --help
python -m rag_retrieval.train.reranker.train_reranker --help
```

Baseline methods are intentionally not exposed as core training objectives.

