from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

NO_MATCH = "NO_MATCH"


class EntityType(str, Enum):
    CONDITION = "condition"
    PAST_MEDICAL_HISTORY = "past_medical_history"
    DRUG = "drug"
    SYMPTOM = "symptom"
    PHENOTYPE = "phenotype"
    ALLERGY = "allergy"
    PROCEDURE = "procedure"


class DoseVerdict(str, Enum):
    WITHIN_RANGE = "within_range"
    ABOVE_RANGE = "above_range"
    BELOW_RANGE = "below_range"
    UNKNOWN_REFERENCE = "unknown_reference"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class RawEntity:
    """Stage 1 output: a free-text span pulled from the transcription."""

    entity_type: EntityType
    text: str
    context: str
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None


@dataclass
class OntologyCandidate:
    """A single Stage 2 shortlist entry."""

    ontology: str
    code: str
    label: str
    synonyms: list[str] = field(default_factory=list)


@dataclass
class GroundedEntity:
    """Stage 3 (+4) output: raw_entity resolved to an ontology code, or NO_MATCH."""

    raw: RawEntity
    selected_label: str
    ontology: str | None
    code: str | None
    label: str | None
    confidence: float
    dose_verdict: DoseVerdict = DoseVerdict.NOT_APPLICABLE


@dataclass
class TranscriptionResult:
    """Final structured record for one transcription."""

    transcription_id: str
    entities: list[GroundedEntity] = field(default_factory=list)
