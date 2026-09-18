from __future__ import annotations

import os
from pathlib import Path

from cooksafe import JsonCache
from dotenv import load_dotenv
from typesafe_sdk import Choice, TypeSafeClient

from .models import NO_MATCH, GroundedEntity, OntologyCandidate, RawEntity

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TYPESAFE_MODEL = "jev-latest"

_client: TypeSafeClient | None = None


def _get_client() -> TypeSafeClient:
    global _client
    if _client is None:
        _client = TypeSafeClient(
            api_key=os.environ["TYPESAFE_API_KEY"],
            base_url=os.environ.get("TYPESAFE_BASE_URL"),
            timeout=30.0,
        )
    return _client


# Cached to grounding_cache.json so repeated runs during development replay
# instead of re-hitting the paid API. Delete it to re-run live.
_cache = JsonCache(Path(__file__).resolve().parent.parent / "grounding_cache.json")


def _describe(candidate: OntologyCandidate) -> str:
    if not candidate.synonyms:
        return f"{candidate.label} ({candidate.ontology})"
    return f"{candidate.label} ({candidate.ontology}); synonyms: {', '.join(candidate.synonyms[:5])}"


@_cache
def _call_typesafe(context: str, instructions: str, criteria: dict[str, str]) -> dict:
    """Single System One call: one Choice question, criteria IS the closed
    output set (Stage 2's shortlist + NO_MATCH) -- Typesafe never sees the
    ontology itself.
    """
    answer = _get_client().system_one(
        state=context,
        questions={"grounding": Choice(instructions=instructions, criteria=criteria)},
        model=TYPESAFE_MODEL,
    ).choices["grounding"]
    return {"choice": answer.choice, "confidence": answer.confidence}


def ground_entity(
    raw_entity: RawEntity,
    candidates: list[OntologyCandidate],
) -> GroundedEntity:
    """Stage 3: Typesafe classifies raw_entity against the Stage 2 shortlist.

    The shortlist (candidates + an explicit NO_MATCH option) is Typesafe's
    entire closed output set for this call — it never sees the full
    ontology. Uses TYPESAFE_API_KEY.
    """
    if not candidates:
        return GroundedEntity(
            raw=raw_entity,
            selected_label=NO_MATCH,
            ontology=None,
            code=None,
            label=None,
            confidence=0.0,
        )

    by_code = {c.code: c for c in candidates}
    criteria = {c.code: _describe(c) for c in candidates}
    criteria[NO_MATCH] = "None of the listed concepts correctly identify this entity."

    instructions = (
        f'The text "{raw_entity.text}" ({raw_entity.entity_type.value}) appears in the '
        "context below. Which listed concept, if any, does it refer to? Use "
        f"'{NO_MATCH}' if none of them are correct — do not pick the closest match."
    )

    result = _call_typesafe(raw_entity.context, instructions, criteria)
    picked = by_code.get(result["choice"])

    return GroundedEntity(
        raw=raw_entity,
        selected_label=result["choice"],
        ontology=picked.ontology if picked else None,
        code=picked.code if picked else None,
        label=picked.label if picked else None,
        confidence=result["confidence"],
    )
