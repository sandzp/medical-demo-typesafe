from __future__ import annotations

from typing import Tuple

from .models import DoseVerdict, GroundedEntity

NormalizedDose = Tuple[float, str]  # (amount, unit)
ReferenceRange = Tuple[float, float, str]  # (min, max, unit)


def normalize_dose(raw_dose: str | None, route: str | None) -> NormalizedDose | None:
    """Parse the extracted dose text (as written) into structured (amount, unit)."""
    raise NotImplementedError


def lookup_reference_range(ontology: str, code: str) -> ReferenceRange | None:
    """Look up a (min, max, unit) reference dosing range for a grounded drug concept.

    Source TBD (formulary dataset / RxNorm dosing fields / DailyMed label data).
    """
    raise NotImplementedError


def validate_dose(entity: GroundedEntity) -> DoseVerdict:
    """Stage 4: classify a grounded drug's dose into a closed bucket set.

    Normalizes the dose, looks up the reference range, and asks Typesafe to
    pick from {within_range, above_range, below_range, unknown_reference}.
    """
    raise NotImplementedError
