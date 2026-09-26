"""TypeSafeExtractor's own logic — dedup, batching, direction, filtering.

All of this runs against `FakeTypeSafeClient` (see conftest.py): zero real
TypeSafe calls, so it's safe to run constantly while iterating. The one real
GLiNER test below is local and free (no API key involved).

The one test that hits the real TypeSafe API lives in test_typesafe.py, gated
behind `--run-typesafe` — see that file's docstring.
"""

from __future__ import annotations

import pytest

from conftest import FakeTypeSafeClient
from typesafe_extractor import RELATION_TYPES, TypeSafeExtractor, gliner_candidate_extractor

SAMPLE = (
    "Maria Chen is the CTO of Acme Robotics. "
    "She opened the company's new office in Berlin in March and hired Tom Okafor to run it."
)
TRUTH = {
    "maria chen": "PERSON",
    "tom okafor": "PERSON",
    "acme robotics": "ORGANIZATION",
    "berlin": "LOCATION",
    ("maria chen", "acme robotics"): "WORKS_FOR",
}


def make_extractor(**kwargs) -> tuple[TypeSafeExtractor, FakeTypeSafeClient]:
    fake = FakeTypeSafeClient(TRUTH)
    extractor = TypeSafeExtractor(
        fake, candidate_extractor=gliner_candidate_extractor(threshold=0.3), **kwargs
    )
    return extractor, fake


# --------------------------------------------------------------------- GLiNER
# Real GLiNER, no TypeSafe involved — local ONNX/torch model, no API key.
async def test_gliner_finds_the_expected_candidates() -> None:
    candidates = gliner_candidate_extractor(threshold=0.3)
    result = await candidates.extract(SAMPLE, extract_relations=False, extract_preferences=False)
    names = {e.name for e in result.entities}
    assert {"Maria Chen", "Acme Robotics", "Berlin", "Tom Okafor"} <= names


# ------------------------------------------------------------- typing/filtering (fake TypeSafe)
async def test_typing_keeps_real_entities_and_drops_noise() -> None:
    extractor, fake = make_extractor()
    result = await extractor.extract(SAMPLE)
    found = {e.name: e.type for e in result.entities}
    assert found == {
        "Maria Chen": "PERSON",
        "Tom Okafor": "PERSON",
        "Acme Robotics": "ORGANIZATION",
        "Berlin": "LOCATION",
    }
    assert fake.calls, "extractor never called system_one"


async def test_relations_are_directional_and_typed() -> None:
    extractor, _ = make_extractor()
    result = await extractor.extract(SAMPLE)
    triples = {r.as_triple for r in result.relations}
    assert ("Maria Chen", "WORKS_FOR", "Acme Robotics") in triples
    # The fake truth table has no entry for the reverse direction -> NONE -> dropped.
    assert ("Acme Robotics", "WORKS_FOR", "Maria Chen") not in triples


async def test_low_confidence_entities_are_dropped() -> None:
    fake = FakeTypeSafeClient(TRUTH, confidence=0.3)  # below default min_entity_confidence=0.5
    extractor = TypeSafeExtractor(
        fake, candidate_extractor=gliner_candidate_extractor(threshold=0.3)
    )
    result = await extractor.extract(SAMPLE, extract_relations=False)
    assert result.entities == []


async def test_min_entity_confidence_is_configurable() -> None:
    fake = FakeTypeSafeClient(TRUTH, confidence=0.3)
    extractor = TypeSafeExtractor(
        fake,
        candidate_extractor=gliner_candidate_extractor(threshold=0.3),
        min_entity_confidence=0.1,
    )
    result = await extractor.extract(SAMPLE, extract_relations=False)
    assert len(result.entities) == 4


async def test_entity_types_filter_narrows_the_result() -> None:
    extractor, _ = make_extractor()
    result = await extractor.extract(SAMPLE, entity_types=["PERSON"])
    assert {e.type for e in result.entities} == {"PERSON"}


async def test_extract_relations_false_skips_the_relation_pass() -> None:
    extractor, fake = make_extractor()
    result = await extractor.extract(SAMPLE, extract_relations=False)
    assert result.entities  # typing still happens
    assert result.relations == []
    # Every question asked was an entity-typing question ("e0", "e1", ...), never "r*".
    asked = [name for call in fake.calls for name in call["names"]]
    assert asked and all(name.startswith("e") for name in asked)


async def test_empty_text_returns_an_empty_result() -> None:
    extractor, fake = make_extractor()
    result = await extractor.extract("")
    assert result.entities == []
    assert result.relations == []
    assert fake.calls == []  # no wasted API calls on empty input


async def test_no_candidates_returns_an_empty_result_without_calling_typesafe() -> None:
    from neo4j_agent_memory.extraction import ExtractionResult

    class NoCandidates:
        async def extract(self, text: str, **kwargs: object) -> ExtractionResult:
            return ExtractionResult(source_text=text)

    fake = FakeTypeSafeClient(TRUTH)
    extractor = TypeSafeExtractor(fake, candidate_extractor=NoCandidates())
    result = await extractor.extract("text with nothing GLiNER-worthy in it")
    assert result.entities == []
    assert fake.calls == []


# --------------------------------------------------------------------- batching
async def test_questions_are_chunked_at_max_questions_per_call() -> None:
    extractor, fake = make_extractor(max_questions_per_call=2)
    await extractor.extract(SAMPLE, extract_relations=False)
    # 4 candidates, 2 per call -> 2 calls, each with at most 2 questions.
    assert len(fake.calls) == 2
    assert all(call["questions"] <= 2 for call in fake.calls)


async def test_max_pairs_caps_relation_questions() -> None:
    # SAMPLE's 4 entities co-occur in enough sentences to produce more than one
    # ordered pair; max_pairs=1 must cap the relation questions asked to 1.
    extractor, fake = make_extractor(max_pairs=1, max_questions_per_call=100)
    await extractor.extract(SAMPLE)
    relation_questions = [name for call in fake.calls for name in call["names"] if name.startswith("r")]
    assert len(relation_questions) <= 1


# --------------------------------------------------------------------- validation
def test_entity_labels_without_none_is_rejected() -> None:
    fake = FakeTypeSafeClient(TRUTH)
    with pytest.raises(ValueError, match="NONE"):
        TypeSafeExtractor(fake, entity_labels={"PERSON": "x"}, relation_labels=RELATION_TYPES)


def test_relation_labels_without_none_is_rejected() -> None:
    from typesafe_extractor import POLEO_TYPES

    fake = FakeTypeSafeClient(TRUTH)
    with pytest.raises(ValueError, match="NONE"):
        TypeSafeExtractor(fake, entity_labels=POLEO_TYPES, relation_labels={"KNOWS": "x"})


# --------------------------------------------------------------------- symmetric relations
async def test_symmetric_relations_are_deduplicated_across_direction() -> None:
    text = "Alice met Bob at the office. Bob met Alice for lunch the next day."
    truth = {
        "alice": "PERSON",
        "bob": "PERSON",
        ("alice", "bob"): "KNOWS",
        ("bob", "alice"): "KNOWS",
    }
    fake = FakeTypeSafeClient(truth)
    extractor = TypeSafeExtractor(fake, candidate_extractor=gliner_candidate_extractor(threshold=0.3))
    result = await extractor.extract(text)
    knows = [r for r in result.relations if r.relation_type == "KNOWS"]
    assert len(knows) == 1, f"expected exactly one deduplicated KNOWS relation, got {knows}"
