"""TypeSafe-powered entity and relation extractor for neo4j-agent-memory.

Two stages, each doing what it is good at:

    1. Candidate spans (recall)    — GLiNER, via the library's ``ExtractorBuilder``.
                                     Run at a low threshold so little is missed.
    2. Typing + relations (precision) — TypeSafe ``system_one`` ``Choice`` questions:
                                     "what POLE+O type is this mention?" (or NONE = noise)
                                     and "what relationship does the text state from A to B?"

``TypeSafeExtractor`` implements the library's ``EntityExtractor`` protocol, so it
plugs straight into ``MemoryClient(settings, extractor=...)``. From then on every
``short_term.add_message(..., extract_entities=True)`` — including the messages the
LangChain ``Neo4jMemoryMiddleware`` persists — builds the graph through TypeSafe:
memory writes the ``:Entity`` nodes, links them to the message, and stores the
relations by entity name.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from itertools import permutations

from neo4j_agent_memory.extraction import (
    EntityExtractor,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    ExtractorBuilder,
)
from typesafe_sdk import AsyncTypeSafeClient, Choice, ChoiceAnswer, RetryPolicy

NONE = "NONE"

#: POLE+O entity types. Descriptions go to TypeSafe as the choice criteria.
POLEO_TYPES: dict[str, str] = {
    "PERSON": "A specific, named human being",
    "ORGANIZATION": "A company, institution, team, restaurant, brand, or other group",
    "LOCATION": "A place: city, country, region, address, venue, or building",
    "EVENT": "Something that happens at a point in time: meeting, launch, conference, incident",
    "OBJECT": "A product, tool, document, vehicle, software, or other thing",
    NONE: "Not a real entity: a generic word, pronoun, date, number, or extraction noise",
}

#: Relationship types, always read as `source` -> `target`.
RELATION_TYPES: dict[str, str] = {
    "WORKS_FOR": "`source` works for, leads, or founded the organization `target`",
    "LOCATED_IN": "`source` is based in, lives in, or takes place in the location `target`",
    "PART_OF": "`source` is a part, team, subsidiary, or member of `target`",
    "KNOWS": "`source` and `target` are people who know, met, or work with each other",
    "PARTICIPATED_IN": "`source` took part in, organized, or attended the event `target`",
    "OWNS": "`source` owns, built, or is responsible for `target`",
    "USES": "`source` uses or depends on `target`",
    "PREFERS": "`source` likes, prefers, or recommends `target`",
    NONE: "The document states no direct relationship from `source` to `target`",
}

#: Relations that read the same in both directions — stored once per pair.
SYMMETRIC_RELATIONS = frozenset({"KNOWS"})

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def typesafe_client() -> AsyncTypeSafeClient:
    """A TypeSafe client that is patient with 429s.

    The SDK default is 2 retries with waits of at most 5s. Behind the Vercel AI
    Gateway the upstream provider intermittently answers 429 "high demand", so
    retry harder: up to 8 retries, backing off to 20s (and honoring `retry-after`).
    Configuration (key, base URL, model) still comes from the TYPESAFE_* env vars.
    """
    return AsyncTypeSafeClient(
        retry=RetryPolicy(max_retries=8, backoff_initial=1.0, backoff_max=20.0),
        timeout=60.0,
    )


def gliner_candidate_extractor(
    model: str | None = None, threshold: float = 0.3, device: str = "cpu"
) -> EntityExtractor:
    """GLiNER span finder. Low threshold on purpose: TypeSafe filters the noise."""
    return ExtractorBuilder().with_gliner(model=model, threshold=threshold, device=device).build()


class TypeSafeExtractor:
    """Find spans with a candidate extractor, then type and relate them with TypeSafe.

    Questions are batched: one ``system_one`` call answers up to
    ``max_questions_per_call`` questions, and calls run ``max_concurrency`` at a time.
    """

    def __init__(
        self,
        client: AsyncTypeSafeClient,
        *,
        candidate_extractor: EntityExtractor | None = None,
        entity_labels: Mapping[str, str] = POLEO_TYPES,
        relation_labels: Mapping[str, str] = RELATION_TYPES,
        min_entity_confidence: float = 0.5,
        min_relation_confidence: float = 0.6,
        max_questions_per_call: int = 25,
        max_pairs: int = 40,
        max_concurrency: int = 4,
        model: str | None = None,
    ) -> None:
        if NONE not in entity_labels or NONE not in relation_labels:
            raise ValueError(f"entity_labels and relation_labels must include a {NONE!r} label")
        self.client = client
        self.candidates = candidate_extractor or gliner_candidate_extractor()
        self.entity_labels = dict(entity_labels)
        self.relation_labels = dict(relation_labels)
        self.min_entity_confidence = min_entity_confidence
        self.min_relation_confidence = min_relation_confidence
        self.max_questions_per_call = max_questions_per_call
        self.max_pairs = max_pairs
        self.model = model
        self._semaphore = asyncio.Semaphore(max_concurrency)
        #: The most recent result — memory calls ``extract`` internally, so this is
        #: how callers see what TypeSafe decided for the message they just stored.
        self.last_result: ExtractionResult | None = None

    # ------------------------------------------------------------------ protocol
    async def extract(
        self,
        text: str,
        *,
        entity_types: list[str] | None = None,
        extract_relations: bool = True,
        extract_preferences: bool = True,
    ) -> ExtractionResult:
        """``EntityExtractor.extract``. Preferences are not extracted (always empty)."""
        result = await self._extract(
            text, entity_types=entity_types, extract_relations=extract_relations
        )
        self.last_result = result
        return result

    async def extract_batch(
        self, texts: Sequence[str], **kwargs: object
    ) -> list[ExtractionResult]:
        return list(await asyncio.gather(*(self._extract(t, **kwargs) for t in texts)))  # type: ignore[arg-type]

    # ------------------------------------------------------------------ stages
    async def _extract(
        self,
        text: str,
        *,
        entity_types: list[str] | None = None,
        extract_relations: bool = True,
    ) -> ExtractionResult:
        if not text or not text.strip():
            return ExtractionResult(source_text=text)

        found = await self.candidates.extract(
            text, extract_relations=False, extract_preferences=False
        )
        candidates = _dedupe(found.entities)
        if not candidates:
            return ExtractionResult(source_text=text)

        entities = await self._type_entities(text, candidates)
        if entity_types:
            allowed = {t.upper() for t in entity_types}
            entities = [e for e in entities if e.type in allowed]

        relations: list[ExtractedRelation] = []
        if extract_relations and len(entities) > 1:
            relations = await self._relate(text, entities)

        return ExtractionResult(entities=entities, relations=relations, source_text=text)

    async def _type_entities(
        self, text: str, candidates: list[ExtractedEntity]
    ) -> list[ExtractedEntity]:
        questions = {
            f"e{i}": Choice(
                # Structured instructions: the mention travels as data, so quotes or
                # odd characters in a name cannot garble the question.
                instructions={
                    "task": "Classify this mention as it is used in the document.",
                    "mention": entity.name,
                },
                criteria=self.entity_labels,
            )
            for i, entity in enumerate(candidates)
        }
        answers = await self._ask(text, questions)

        typed: list[ExtractedEntity] = []
        for i, entity in enumerate(candidates):
            answer = answers.get(f"e{i}")
            if answer is None or answer.choice == NONE:
                continue
            if answer.confidence < self.min_entity_confidence:
                continue
            typed.append(
                ExtractedEntity(
                    name=entity.name,
                    type=answer.choice,
                    # A candidate subtype only survives if TypeSafe agreed on the type.
                    subtype=entity.subtype if entity.type == answer.choice else None,
                    start_pos=entity.start_pos,
                    end_pos=entity.end_pos,
                    confidence=answer.confidence,
                    context=entity.context,
                    attributes={
                        **entity.attributes,
                        "typed_by": "typesafe",
                        "candidate_type": entity.type,
                        "candidate_confidence": entity.confidence,
                    },
                    extractor="typesafe",
                )
            )
        return typed

    async def _relate(
        self, text: str, entities: list[ExtractedEntity]
    ) -> list[ExtractedRelation]:
        pairs = _co_occurring_pairs(text, entities)[: self.max_pairs]
        if not pairs:
            return []

        questions = {
            f"r{i}": Choice(
                instructions={
                    "task": "Which relationship does the document state from source to "
                    "target? Direction matters; answer NONE unless the text says so.",
                    "source": source.name,
                    "target": target.name,
                },
                criteria=self.relation_labels,
            )
            for i, (source, target) in enumerate(pairs)
        }
        answers = await self._ask(text, questions)

        relations: list[ExtractedRelation] = []
        seen: set[tuple[str, str, str]] = set()
        for i, (source, target) in enumerate(pairs):
            answer = answers.get(f"r{i}")
            if answer is None or answer.choice == NONE:
                continue
            if answer.confidence < self.min_relation_confidence:
                continue
            key = (source.name, answer.choice, target.name)
            if answer.choice in SYMMETRIC_RELATIONS:
                key = (*sorted((source.name, target.name)), answer.choice)  # type: ignore[assignment]
            if key in seen:
                continue
            seen.add(key)
            relations.append(
                ExtractedRelation(
                    source=source.name,
                    target=target.name,
                    relation_type=answer.choice,
                    confidence=answer.confidence,
                )
            )
        return relations

    # ------------------------------------------------------------------ transport
    async def _ask(self, text: str, questions: dict[str, Choice]) -> dict[str, ChoiceAnswer]:
        """Answer ``questions`` about ``text``, chunked and run concurrently."""
        names = list(questions)
        size = self.max_questions_per_call
        chunks = [names[i : i + size] for i in range(0, len(names), size)]

        async def one(chunk: list[str]) -> dict[str, ChoiceAnswer]:
            async with self._semaphore:
                response = await self.client.system_one(
                    state={"document": text},
                    questions={name: questions[name] for name in chunk},
                    model=self.model,
                )
            return response.choices

        answers: dict[str, ChoiceAnswer] = {}
        for part in await asyncio.gather(*(one(chunk) for chunk in chunks)):
            answers.update(part)
        return answers


def _dedupe(entities: Sequence[ExtractedEntity]) -> list[ExtractedEntity]:
    """One candidate per normalized name, keeping the most confident span."""
    best: dict[str, ExtractedEntity] = {}
    for entity in entities:
        key = entity.normalized_name
        if len(key) < 2:
            continue
        if key not in best or entity.confidence > best[key].confidence:
            best[key] = entity
    return list(best.values())


def _co_occurring_pairs(
    text: str, entities: list[ExtractedEntity]
) -> list[tuple[ExtractedEntity, ExtractedEntity]]:
    """Ordered pairs of entities mentioned in the same sentence.

    Pairwise questions grow as n², so only entities that share a sentence are
    asked about. Both directions are asked; SYMMETRIC_RELATIONS are deduped later.
    """
    pairs: list[tuple[ExtractedEntity, ExtractedEntity]] = []
    seen: set[tuple[str, str]] = set()
    for sentence in _SENTENCE_SPLIT.split(text):
        lowered = sentence.lower()
        present = [e for e in entities if e.normalized_name in lowered]
        for source, target in permutations(present, 2):
            key = (source.normalized_name, target.normalized_name)
            if key not in seen:
                seen.add(key)
                pairs.append((source, target))
    return pairs


__all__ = [
    "POLEO_TYPES",
    "RELATION_TYPES",
    "TypeSafeExtractor",
    "gliner_candidate_extractor",
    "typesafe_client",
]
