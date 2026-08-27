AVAILABLE_RANKERS = {}

try:
    from rag_retrieval.infer.reranker_models.cross_encoder_ranker import CrossEncoderRanker
    AVAILABLE_RANKERS["CrossEncoderRanker"] = CrossEncoderRanker
except Exception as e:
    print(f"[WARN] Failed to load CrossEncoderRanker: {e}")

try:
    from rag_retrieval.infer.reranker_models.llm_rankers import LLMRanker
    AVAILABLE_RANKERS["LLMRanker"] = LLMRanker
except Exception as e:
    print(f"[WARN] Failed to load LLMRanker: {e}")

try:
    from rag_retrieval.infer.reranker_models.llm_decoder_ranker import LLMDecoderRanker
    AVAILABLE_RANKERS["LLMDecoderRanker"] = LLMDecoderRanker
except Exception as e:
    print(f"[WARN] Failed to load LLMDecoderRanker: {e}")
