from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .models import EntityType, RawEntity

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

EXTRACTION_MODEL = "gpt-4o-mini"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


EXTRACTION_SYSTEM_PROMPT = """You are a clinical entity extraction system. Given a medical transcription, extract every mention of a clinical entity and classify it into exactly one of these categories:

- condition: a diagnosis or medical condition, current or being ruled in/out
- past_medical_history: a prior diagnosis, surgery, or health event mentioned as history rather than the current presenting problem
- drug: a medication, including dose/route/frequency if stated
- symptom: something the patient reports or exhibits (complaint, sign observed on exam)
- phenotype: an observable clinical finding, especially a lab/imaging abnormality or physical finding, as distinct from a subjective complaint
- allergy: a stated allergy or adverse reaction to a substance
- procedure: a medical or surgical procedure performed, planned, or recommended

Rules:
1. Optimize for recall, not precision. Extract every plausible mention, even if ambiguous or hedged ("possible", "rule out", "questionable"). A downstream verification step filters incorrect extractions — do not filter here.
2. Extract `text` exactly as it appears in the transcription. Do not normalize, expand abbreviations, correct spelling, or paraphrase.
3. `context` is the surrounding sentence(s), quoted verbatim, with enough surrounding text to disambiguate negation, laterality, and timing (e.g. "denies chest pain" must keep "denies").
4. For drug entities only, also extract `dose`, `route` (e.g. "PO", "IV", "topical"), and `frequency` (e.g. "BID", "q6h", "once daily") exactly as written when stated. Use null for any that are not mentioned — never infer a value.
5. Extract each occurrence separately. Do not deduplicate repeated mentions of the same entity.
6. A negated or denied entity is still an extraction — preserve the negation in `context` rather than omitting the entity.
7. Output ONLY a JSON array, no prose, no markdown code fences. Each element has exactly this shape:

{
  "entity_type": "condition" | "past_medical_history" | "drug" | "symptom" | "phenotype" | "allergy" | "procedure",
  "text": string,
  "context": string,
  "dose": string | null,
  "route": string | null,
  "frequency": string | null
}

If no entities are found, output []."""


def extract_entities(transcription: str) -> list[RawEntity]:
    """Stage 1: open-ended LLM extraction of raw entity spans.

    Recall-optimized, no ontology constraint. Calls a regular LLM (OpenAI)
    with EXTRACTION_SYSTEM_PROMPT and parses the response into RawEntity
    objects, one per (type, span).
    """
    response = _get_client().chat.completions.create(
        model=EXTRACTION_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": transcription},
        ],
    )
    content = response.choices[0].message.content or "[]"
    return [_parse_raw_entity(item) for item in _parse_json_array(content)]


def _parse_json_array(content: str) -> list[dict]:
    """Parse the model's JSON array, tolerating an accidental ```json fence."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array, got {type(data).__name__}")
    return data


def _parse_raw_entity(item: dict) -> RawEntity:
    return RawEntity(
        entity_type=EntityType(item["entity_type"]),
        text=item["text"],
        context=item["context"],
        dose=item.get("dose"),
        route=item.get("route"),
        frequency=item.get("frequency"),
    )
