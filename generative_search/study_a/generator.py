import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

from .data import stable_document_id


DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "mistral_nemo_study_a.txt"
JSON_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?[ \t\r]*\n(?P<body>.*?)\n?```\s*$", re.IGNORECASE | re.DOTALL)


def load_prompt_template(prompt_path: Optional[str] = None) -> str:
    path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
    return path.read_text(encoding="utf-8")


def format_profiles(profiles: List[dict]) -> str:
    clean_profiles = [
        {
            "creator_id": str(profile["creator_id"]),
            "document_id": str(profile.get("document_id") or stable_document_id(profile["creator_id"])),
            "content": str(profile["content"]),
        }
        for profile in profiles
    ]
    return "\n".join(
        f"creator_id: {profile['creator_id']}\n"
        f"document_id: {profile['document_id']}\n"
        f"profile: {profile['content']}"
        for profile in clean_profiles
    )


def build_prompt(query: str, profiles: List[dict], output_k: int, prompt_path: Optional[str] = None) -> str:
    prompt = load_prompt_template(prompt_path)
    replacements = {"{query}": query, "{profiles}": format_profiles(profiles), "{output_k}": str(output_k)}
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    unreplaced = [placeholder for placeholder in replacements if placeholder in prompt]
    if unreplaced:
        raise ValueError(f"Generator prompt has unreplaced placeholders: {unreplaced}")
    return prompt


def assert_prompt_is_clean(prompt: str) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Generator prompt is empty")


def ollama_schema(output_k: int, candidate_creator_ids: Optional[List[str]] = None, candidate_document_ids: Optional[List[str]] = None) -> dict:
    creator_ids = [str(cid) for cid in candidate_creator_ids] if candidate_creator_ids else None
    document_ids = [str(did) for did in candidate_document_ids] if candidate_document_ids else ([stable_document_id(cid) for cid in creator_ids] if creator_ids else None)
    creator_schema = {"type": "string"}
    document_schema = {"type": "string"}
    if creator_ids:
        creator_schema["enum"] = creator_ids
    if document_ids:
        document_schema["enum"] = document_ids
    return {
        "type": "object", "additionalProperties": False,
        "required": ["recommendations"],
        "properties": {
            "recommendations": {
                "type": "array", "minItems": output_k, "maxItems": output_k,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["rank", "creator_id", "document_id", "reason"],
                    "properties": {
                        "rank": {"type": "integer", "minimum": 1, "maximum": output_k},
                        "creator_id": creator_schema,
                        "document_id": document_schema,
                        "reason": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    }


def is_gpt_oss_model(model: str) -> bool:
    return str(model).strip().lower().startswith("gpt-oss")


def call_ollama(base_url: str, model: str, prompt: str, temperature: float, output_k: int = 5, seed: int = 42, candidate_creator_ids: Optional[List[str]] = None, candidate_document_ids: Optional[List[str]] = None) -> str:
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "seed": seed},
    }
    if is_gpt_oss_model(model):
        payload["think"] = "low"
    else:
        payload["format"] = ollama_schema(output_k, candidate_creator_ids, candidate_document_ids)
    req = urllib.request.Request(base_url.rstrip("/") + "/api/generate", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            response_text = body.get("response", "")
            if not isinstance(response_text, str) or not response_text.strip():
                raise RuntimeError(
                    "Ollama returned an empty response "
                    f"(model={model!r}, done={body.get('done')!r}, done_reason={body.get('done_reason')!r}, "
                    f"thinking_empty={not bool(str(body.get('thinking', '')).strip())})"
                )
            return response_text
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc


def parse_json_output(raw: str) -> dict:
    text = str(raw).strip()
    fence_match = JSON_CODE_FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group("body").strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Generator output root is not an object.")
    return parsed


def validate_generation(parsed: dict, supplied_profiles: List[dict], output_k: int, query_id: str = "unknown") -> Tuple[bool, List[str], dict]:
    errors = []
    if supplied_profiles and isinstance(supplied_profiles[0], str):
        supplied_profiles = [{"creator_id": cid, "document_id": stable_document_id(cid)} for cid in supplied_profiles]
    supplied = {str(item["creator_id"]) for item in supplied_profiles}
    supplied_documents = {str(item["document_id"]) for item in supplied_profiles}
    document_to_creator = {str(item["document_id"]): str(item["creator_id"]) for item in supplied_profiles}
    recs = parsed.get("recommendations") if isinstance(parsed, dict) else None
    if not isinstance(recs, list):
        placeholders = [f"__INVALID_{query_id}_{idx}" for idx in range(1, output_k + 1)]
        return False, ["recommendations is not a list"], {"recommendations": [], "valid_recommendations": [], "position_preserving_creator_ids": placeholders, "parsed_recommendation_count": 0, "valid_citation_count": 0, "creator_document_match_count": 0, "hallucinated_count": 0, "duplicate_count": 0, "citation_id_validity_rate": 0.0, "valid_citation_rate": 0.0, "creator_document_match_rate": 0.0, "hallucinated_creator_rate": 0.0, "duplicate_creator_rate": 0.0, "output_completeness": 0.0, "exact_output_validity": 0.0}
    if len(recs) != output_k:
        errors.append(f"expected {output_k} recommendations, got {len(recs)}")
    seen = set(); duplicate_count = 0; hallucinated_count = 0; valid_citation_count = 0; creator_document_match_count = 0
    sanitized = []; valid_recommendations = []; position_preserving_creator_ids = []
    for idx, rec in enumerate(recs, start=1):
        if not isinstance(rec, dict):
            errors.append(f"recommendation {idx} is not an object")
            position_preserving_creator_ids.append(f"__INVALID_{query_id}_{idx}")
            continue
        creator_id = rec.get("creator_id") if isinstance(rec.get("creator_id"), str) else ""
        document_id = rec.get("document_id") if isinstance(rec.get("document_id"), str) else ""
        reason = rec.get("reason") if isinstance(rec.get("reason"), str) else ""
        rank_ok = rec.get("rank") == idx
        creator_in_pool = creator_id in supplied
        document_in_pool = document_id in supplied_documents
        document_matches_creator = document_to_creator.get(document_id) == creator_id
        duplicate = creator_id in seen
        if duplicate: duplicate_count += 1; errors.append(f"duplicate creator_id {creator_id}")
        seen.add(creator_id)
        if not creator_in_pool: hallucinated_count += 1; errors.append(f"creator_id {creator_id} is not in supplied candidates")
        if not document_in_pool: errors.append(f"document_id {document_id} is not in supplied documents")
        if document_in_pool and not document_matches_creator: errors.append(f"document_id {document_id} does not belong to creator_id {creator_id}")
        if not rank_ok: errors.append(f"rank at position {idx} is {rec.get('rank')}")
        if document_in_pool:
            valid_citation_count += 1
        if document_matches_creator:
            creator_document_match_count += 1
        if not reason.strip(): errors.append(f"empty reason at position {idx}")
        item = {"rank": rec.get("rank"), "creator_id": creator_id, "document_id": document_id, "reason": reason}
        sanitized.append(item)
        if rank_ok and creator_in_pool and document_in_pool and document_matches_creator and not duplicate and reason.strip():
            valid_recommendations.append(item)
            position_preserving_creator_ids.append(creator_id)
        else:
            position_preserving_creator_ids.append(f"__INVALID_{query_id}_{idx}")
    while len(position_preserving_creator_ids) < output_k:
        position_preserving_creator_ids.append(f"__INVALID_{query_id}_{len(position_preserving_creator_ids) + 1}")
    position_preserving_creator_ids = position_preserving_creator_ids[:output_k]
    parsed_count = len(recs)
    citation_id_validity_rate = valid_citation_count / parsed_count if parsed_count else 0.0
    rates = {
        "recommendation_count": parsed_count, "parsed_recommendation_count": parsed_count,
        "valid_citation_count": valid_citation_count, "creator_document_match_count": creator_document_match_count, "duplicate_count": duplicate_count, "hallucinated_count": hallucinated_count,
        "citation_id_validity_rate": citation_id_validity_rate,
        "valid_citation_rate": citation_id_validity_rate,
        "creator_document_match_rate": creator_document_match_count / parsed_count if parsed_count else 0.0,
        "hallucinated_creator_rate": hallucinated_count / parsed_count if parsed_count else 0.0,
        "duplicate_creator_rate": duplicate_count / parsed_count if parsed_count else 0.0,
        "output_completeness": min(len(valid_recommendations), output_k) / output_k if output_k else 0.0,
        "exact_output_validity": 1.0 if not errors and len(valid_recommendations) == output_k else 0.0,
        "recommendations": sanitized, "valid_recommendations": valid_recommendations, "position_preserving_creator_ids": position_preserving_creator_ids,
    }
    return not errors, errors, rates
