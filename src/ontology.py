from __future__ import annotations

import httpx

from .models import EntityType, OntologyCandidate

OLS4_SEARCH_URL = "https://www.ebi.ac.uk/ols4/api/search"

# No API key required. TODO: confirm ALLERGY split (substance vs. reaction).
ONTOLOGY_MAP: dict[EntityType, list[str]] = {
    EntityType.CONDITION: ["mondo"],
    EntityType.PAST_MEDICAL_HISTORY: ["mondo"],
    EntityType.SYMPTOM: ["hp"],
    EntityType.PHENOTYPE: ["hp"],
    EntityType.ALLERGY: ["chebi", "mondo"],
    EntityType.DRUG: ["chebi", "ncit"],
    EntityType.PROCEDURE: ["ncit"],
}


def search_candidates(
    entity_type: EntityType,
    query: str,
    top_k: int = 8,
) -> list[OntologyCandidate]:
    """Stage 2: retrieve a bounded shortlist of ontology candidates for one entity.

    Queries OLS4 over the ontologies mapped for entity_type and merges/dedupes
    the results into a top_k shortlist. Must filter each hit on its own
    `ontology_prefix` field — OLS4's cross-ontology linking means the
    `ontology=` request param alone doesn't reliably scope results (verified:
    a `mondo`-scoped query can surface an `HP:` hit).
    """
    candidates: list[OntologyCandidate] = []
    seen: set[tuple[str, str]] = set()

    for ontology in ONTOLOGY_MAP.get(entity_type, []):
        for candidate in _query_ols4(ontology, query, rows=top_k):
            key = (candidate.ontology, candidate.code)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)

    return candidates[:top_k]


def _query_ols4(ontology: str, query: str, rows: int) -> list[OntologyCandidate]:
    """Single OLS4 request against one ontology, filtered to that ontology's hits."""
    response = httpx.get(
        OLS4_SEARCH_URL,
        params={"q": query, "ontology": ontology, "rows": rows},
        timeout=10.0,
    )
    response.raise_for_status()
    docs = response.json().get("response", {}).get("docs", [])

    candidates = []
    for doc in docs:
        if doc.get("ontology_prefix", "").lower() != ontology.lower():
            continue
        obo_id = doc.get("obo_id")
        label = doc.get("label")
        if not obo_id or not label:
            continue
        synonyms = [
            *doc.get("exact_synonyms", []),
            *doc.get("related_synonyms", []),
        ]
        candidates.append(
            OntologyCandidate(ontology=ontology, code=obo_id, label=label, synonyms=synonyms)
        )
    return candidates
