from __future__ import annotations

from .dosing import validate_dose
from .extraction import extract_entities
from .grounding import ground_entity
from .models import EntityType, TranscriptionResult
from .ontology import search_candidates


def run_pipeline(transcription_id: str, transcription: str) -> TranscriptionResult:
    """End to end: Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 (drugs only)."""
    raw_entities = extract_entities(transcription)

    result = TranscriptionResult(transcription_id=transcription_id)
    for raw_entity in raw_entities:
        candidates = search_candidates(raw_entity.entity_type, raw_entity.text)
        grounded = ground_entity(raw_entity, candidates)

        if raw_entity.entity_type is EntityType.DRUG:
            grounded.dose_verdict = validate_dose(grounded)

        result.entities.append(grounded)

    return result
