# GSE Visibility

Code for **Who Gets Seen? Creator Visibility from Ranking to Generation**.

This repository contains the implementation for reranker training, first stage retrieval, generative search experiments, evaluation, and result analysis.

## Installation

```bash
git clone <repository-url>
cd GSE_Visibility
pip install -r requirements.txt
```

The generative search experiments use Ollama for local LLM inference. Ollama should be installed and running before executing Study A or Study B.

## Repository Structure

```text
rag_retrieval/
├── train/reranker/          # RankNet, EAR, EAR-Sym, Pairwise Reg
├── baseline/                # Boratto-reg, PAL, PBiLoss
├── infer/                   # Reranker inference and evaluation
├── generative_search/
│   ├── retriever/           # BM25 and dense retrieval
│   ├── study_a/             # Ranking to generation evaluation
│   └── study_b/             # Controlled position analysis
└── plots/                   # Figure generation
```

## Ranking Methods

The main training framework supports:

```text
ranknet
ear
ear_sym
pairwise_reg
```

Additional baselines are implemented separately:

```text
Boratto-reg
PAL
PBiLoss
```

## Training

```bash
accelerate launch -m rag_retrieval.train.reranker.train_reranker \
  --config <training_config.yaml>
```

Set `loss_type` in the training configuration to the desired objective.

## Reranker Evaluation

```bash
python -m rag_retrieval.infer.eval.evaluate_reranker \
  --jsonl <test.jsonl> \
  --model <model_directory> \
  --output_dir <output_directory> \
  --raw-data-dir <raw_data_directory>
```

## First Stage Retrieval

BM25 retrieval:

```bash
python -m rag_retrieval.generative_search.retriever.prepare \
  --input_jsonl <test.jsonl> \
  --output_root <retrieval_output_directory> \
  --mode prepare \
  --method bm25 \
  --retrieval_k 50
```

## Study A

Study A evaluates the ranking to generation pipeline. The retrieved candidates are reranked, the top 10 are provided to the generator, and the generated top 5 are evaluated.

```bash
python -m rag_retrieval.generative_search.study_a.run \
  --test_jsonl <retrieval_output>/bm25/test_topK.jsonl \
  --model_path <reranker_model> \
  --candidate_source preselected \
  --generator_backend ollama \
  --generator_model <ollama_model> \
  --input_k 10 \
  --output_k 5 \
  --raw_data_dir <raw_data_directory> \
  --output_dir <output_directory>
```

For the **No Reranking** condition, remove `--model_path` and add:

```text
--no_reranking
```

## Study B

Study B keeps the same 10 creators for each query and changes only their ordering before generation.

Prepare the fixed candidate sets:

```bash
python -m rag_retrieval.generative_search.study_b.prepare_dataset \
  --input_jsonl <test.jsonl> \
  --output_jsonl <study_b.jsonl>
```

Run the experiment:

```bash
python -m rag_retrieval.generative_search.study_b.run \
  --test_jsonl <study_b.jsonl> \
  --model_path <reranker_model> \
  --generator_backend ollama \
  --generator_model <ollama_model> \
  --raw_data_dir <raw_data_directory> \
  --output_dir <output_directory>
```

## Evaluation

The repository includes evaluation for:

- graded nDCG
- SkillCov
- Exp and DExp
- creator coverage
- attention group representation
- textual visibility
- generator selection
- position sensitivity

`prior_attention` denotes the historical creator attention signal. `Exp` and `DExp` are evaluation metrics derived from this signal.

## Plotting

Scripts for reproducing the experimental figures are available under:

```text
rag_retrieval/plots/
```