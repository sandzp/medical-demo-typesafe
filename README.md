# Validated Structured Medical Extraction with TypeSafe's Jev

## Problem

Medical encounters are increasingly captured as audio and turned into text
via transcription, and AI is then used to extract structured entities from
that transcription — conditions, past medical history, medications and
doses, symptoms, phenotypes, allergies, procedures — because coding every
note by hand doesn't scale.

Traditional LLMs are good at this kind of named entity recognition, but
they're free-text generators: they can hallucinate a drug that was never
said, misread a dose, or assert a diagnosis the transcript doesn't actually
support. In a medical record, that's not a cosmetic bug — an unverified
entity is a patient-safety and liability risk, so an extraction pipeline
needs a way to check that what the LLM pulled out actually corresponds to
something real before it's trusted.

One way to do that is to ground each extracted entity against a medical
ontology (SNOMED CT, RxNorm, HPO, MONDO, etc.) instead of trusting the raw
free-text span: turn "the model said X" into "X matches concept Y in a
controlled clinical vocabulary," which is a checkable claim instead of an
assertion. This demo grounds and validates entities using Typesafe's Jev
model. Typesafe requires a fixed, enumerable list of possible outputs — it
classifies/verifies, it doesn't freely generate — but an ontology is
anything but small (SNOMED CT, RxNorm, and HPO each have 100k+ concepts), so
we can't hand Typesafe "the whole ontology" as its output list. The
architecture below splits the problem so each stage plays to the right
tool: a regular LLM does open-ended extraction, and Typesafe/Jev is only
ever asked to pick from a small, per-entity candidate set.

### Typesafe's System One and the Jev model

Typesafe's **[System One](https://docs.typesafe.ai/concepts/system-one)**
is its closed-set judgment mode: instead of generating free text, you give
it a piece of [state](https://docs.typesafe.ai/concepts/state) (here, the
transcription context around an entity) plus one or more explicitly
enumerated questions — a **[Choice](https://docs.typesafe.ai/primitives/choice)**
between named alternatives, a **[Noul](https://docs.typesafe.ai/primitives/noul)**
yes/no question, or a **[Score](https://docs.typesafe.ai/primitives/score)**
against an ordered rubric — and it answers each one from *only* the options
you defined. For every question it returns the selected answer, a
confidence score, and the full probability distribution over every listed
option.

**Jev** is the model this demo calls through System One to actually perform
the grounding judgment: "given this entity and its context, which of these
ontology candidates (if any) does it refer to?" Because Jev can only select
from the candidate set it's handed, it structurally cannot hallucinate a
concept that isn't one of the listed options — the worst it can do is pick
the wrong real candidate, or (correctly) say `NO_MATCH`, and either way that
comes back as an explicit, checkable answer with a confidence score attached
rather than an unverifiable free-text guess.

## Setup

`data/mtsamples.csv` is gitignored and is not checked into this repo.
Anyone cloning this project needs to download it separately before running
the notebook or pipeline:

1. Download the "Medical Transcriptions" dataset from Kaggle (uploaded by
   tboyle10): https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
2. Place the CSV at `data/mtsamples.csv`.

You'll also need an `.env` file at the project root with `OPENAI_API_KEY`
and `TYPESAFE_API_KEY` set, then run `uv sync` to install dependencies.

## Pipeline

![Entity grounding pipeline: a transcription is extracted by an open-ended LLM, each entity is narrowed from a large ontology down to a small candidate shortlist, Typesafe classifies against that shortlist, and drugs additionally go through dose validation against a formulary range.](docs/architecture.svg)

Stage 2 is the step that makes this architecture compatible with Typesafe.
It narrows an unbounded ontology (SNOMED CT, RxNorm, and HPO each have over
100,000 concepts) down to a small, per-entity candidate list before Typesafe
evaluates it. Typesafe never classifies against the ontology directly; it
only ever selects from that shortlist.

### Stage 1: Extraction

A general-purpose LLM reads the transcription and extracts candidate
entities as free-text spans, grouped into seven categories:

- Conditions and diagnoses
- Past medical history
- Drugs, including dose, route, and frequency as written
- Symptoms
- Phenotypes
- Allergies
- Procedures

This stage is optimized for recall rather than precision: no vocabulary
constraint is applied, and ambiguous or hedged mentions are extracted rather
than filtered out. Its output is a list of raw entities, each carrying its
type, source text, surrounding context, and, for drugs, any stated
dose/route/frequency.

### Stage 2: Ontology candidate retrieval

For each raw entity, this stage retrieves a small shortlist of plausible
ontology matches rather than searching the full ontology directly:

| Entity type | Ontology |
|---|---|
| Conditions, past medical history, symptoms, procedures | SNOMED CT (or ICD-10 for billing-oriented coding) |
| Drugs | RxNorm |
| Phenotypes | HPO (Human Phenotype Ontology) |
| Allergies | SNOMED CT or RxNorm, substance branch |

Retrieval uses embedding similarity, with fuzzy or lexical matching as a
fallback, over each ontology's preferred terms and synonyms, returning the
top-k candidates (k = 5-10). The shortlist always includes an explicit
`NO_MATCH` option, so Typesafe is never forced to select an incorrect code
when retrieval fails to surface a usable match.

### Stage 3: Typesafe classification

Typesafe receives the raw entity's text and surrounding context, along with
the Stage 2 shortlist as its complete set of possible outputs, and selects
the correct grounding, or `NO_MATCH`. The result is a verified,
ontology-grounded entity rather than an unchecked free-text guess.

### Stage 4: Dose validation (drugs only)

Once a drug entity is grounded to an RxNorm concept, its dose is validated
in three steps:

1. Normalize the extracted dose, route, and frequency to structured units.
2. Look up a reference dosing range for that concept, sourced from a
   formulary dataset, RxNorm's dosing fields, or DailyMed label data.
3. Have Typesafe classify the normalized dose into one of a fixed set of
   buckets: `within_range`, `above_range`, `below_range`, or
   `unknown_reference`.

### Output

The pipeline produces one structured, ontology-grounded record per
transcription. Each entity in that record carries its type, source text
span, ontology code, confidence score, and, for drugs, a dose validation
verdict.

## Open questions

- **Ontology licensing.** Which ontology sources are actually available.
  SNOMED CT and RxNorm require UMLS access; MONDO, HPO, Uberon, ChEBI, and
  NCIt do not.
- **Retrieval implementation.** Whether Stage 2 should use a precomputed
  embedding index, or a lexical/fuzzy search (e.g. OLS4) as an initial
  version.
- **Dosing reference data.** The source for Stage 4's reference dosing
  ranges.
- **Shortlist size.** The right value of k: too small risks omitting the
  correct code from the shortlist; too large stops being a small,
  enumerable list Typesafe can classify against.

## Project layout

- `src/` — pipeline implementation: `extraction.py` (Stage 1),
  `ontology.py` (Stage 2), `grounding.py` (Stage 3), `dosing.py` (Stage 4),
  `pipeline.py` (orchestration), `models.py` (shared data types)
- `data/` — sample transcriptions and ontology reference data (gitignored)
- `docs/architecture.svg` — the architecture diagram referenced above
- `explore.ipynb` — notebook for interactive development and testing
- `pyproject.toml` — project dependencies (`typesafe-sdk`, `cooksafe`,
  `openai`, `pandas`, etc.)
