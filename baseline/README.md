# Baselines

Baselines are isolated from the normal core training objective list.

Supported baseline IDs:

- `boratto_reg` — Boratto-reg
- `pal` — PAL
- `pbiloss_popneg_ft` — PBiLoss

Entrypoints:

```bash
python -m rag_retrieval.baseline.boratto_reg.train_eval --help
python -m rag_retrieval.baseline.pal.train_eval --help
python -m rag_retrieval.baseline.pbiloss.train_eval --help
```

All baseline implementations use the shared grouped trainer infrastructure where appropriate and consume `prior_attention`.

