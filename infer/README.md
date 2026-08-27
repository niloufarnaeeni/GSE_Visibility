# Inference and Evaluation

Main evaluator:

```bash
python -m rag_retrieval.infer.eval.evaluate_reranker --help
```

Supported reranker types:

- `cross-encoder`
- `llm`
- `llm-decoder`

Creator prior attention is loaded from `prior_attention_scores.csv` and used to compute exposure metrics:

- `Exp@k`
- `DExp@k`
