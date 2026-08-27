# Generative Search

Study A pipeline:

```text
query -> first stage retrieval -> Top K -> reranker -> Top 10 -> generator -> Top 5 -> evaluation
```

First-stage retrieval is prepared separately:

```bash
python -m rag_retrieval.generative_search.retriever.prepare \
  --input_jsonl path/to/test.jsonl \
  --mode prepare \
  --method bm25 \
  --retrieval_k 50

python -m rag_retrieval.generative_search.retriever.prepare \
  --input_jsonl path/to/test.jsonl \
  --mode prepare \
  --method dense \
  --embedding_model BAAI/bge-m3 \
  --retrieval_k 50
```

Study A consumes implemented candidate sources only:

- `full`
- `preselected`

Study A entrypoint:

```bash
python -m rag_retrieval.generative_search.study_a.run \
  --test_jsonl path/to/retrieval/test_topK.jsonl \
  --candidate_source preselected \
  --model_path path/to/reranker/checkpoint \
  --generator_backend echo \
  --output_dir outputs/study_a_reranked
```

Use `--no_reranking` for the No Reranking baseline. It preserves the first-stage candidate order and sends the first `input_k` candidates directly to the generator.

No Reranking does not require `--model_path`:

```bash
python -m rag_retrieval.generative_search.study_a.run \
  --test_jsonl path/to/test_topK.jsonl \
  --candidate_source preselected \
  --no_reranking \
  --generator_backend echo \
  --output_dir outputs/study_a_no_reranking
```

Study B entrypoint:

```bash
python -m rag_retrieval.generative_search.study_b.run --help
```
