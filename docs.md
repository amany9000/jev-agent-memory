# Reference

Supporting detail split out of [README.md](README.md): what each file does, and how to
tune the extractor. Start with the README for setup and running; come here once that's
working and you want to change how it behaves.

## Files

| File | What it is |
|---|---|
| `langchain_agent_memory.py` | Memory wiring (Aura + the default spaCy/GLiNER extraction pipeline), document ingestion, and the LangChain agent. |
| `typesafe_extractor.py` | The TypeSafe extractor: GLiNER candidates → TypeSafe typing + relations. Used when `USE_SPACY=false` — see **Extractor options** below. No Neo4j or LangChain code. |
| `utils.py` | Operational commands: `graph`, `fix-vector-indexes`, `wipe-test-data`, `wipe-entities` — see **Keeping debug data out of the real graph** below. |
| `fastembed_embedder.py` | FastEmbed adapter for the memory library's `EmbeddingProvider` protocol. |
| `_env.py` | Loads `.env`. |
| `tests/` | pytest suite — see **Testing** in the README. |
| `data/docs/` | Sample documents for `make ingest`. |

## Extractor options

`USE_SPACY` picks the extractor `open_memory()` connects `MemoryClient` with:

- `USE_SPACY=true` (default): the standard neo4j-agent-memory pipeline (spaCy, then
  GLiNER; both local, no API key), via `ExtractionConfig` — no `extractor=` override.
- `USE_SPACY=false`: `typesafe_extractor.TypeSafeExtractor` — GLiNER candidates typed
  and related by a real TypeSafe API call, passed as `MemoryClient(..., extractor=...)`.
  This is the extractor the project showcases; the pipeline above is the free fallback.

`tests/test_extractor.py` exercises `TypeSafeExtractor` against a fake TypeSafe client
(free); `make test-typesafe` makes one real call.

## A library bug that affects repeated entity names

`neo4j_agent_memory.memory.short_term._extract_and_link_entities` generates a new UUID
for each extracted entity and passes it to a `MERGE (e:Entity {name, type}) ... SET e.id
= $id` query. When an entity with that exact name+type already exists anywhere in the
graph (from an earlier message, or an earlier ingest run), the `MERGE` matches the
*existing* node — with its own, different, real id — but the surrounding code still uses
the UUID it generated locally for the follow-up `MENTIONS` link. That link silently
matches no node and is never created. The entity's properties (confidence, `updated_at`)
still update correctly; only the message → entity link is lost. The same pattern affects
`RELATED_TO` relations built from the same local id map.

In practice: the *first* time a document mentions "Maria Chen", she links correctly.
Documents that use her name again — including later passages in the same ingest run —
silently produce an unlinked (but perfectly real) `Maria Chen` node. This affects every
extractor (GLiNER, spaCy, LLM, `TypeSafeExtractor`), since the bug is in the shared
`neo4j-agent-memory` write path, not in this project's code. It hasn't been patched here;
it's a library issue to report upstream or work around (e.g. re-resolving the entity's
real id by name before linking) if it matters for your use of the graph.

This is also why `entities_mentioned_by()` in `langchain_agent_memory.py` can print
"(no entities)" for a passage that clearly mentions known names — it isn't a bug in that
helper; it's this one.

## Keeping debug data out of the real graph

Aura's Free tier is one database per instance, so ad hoc debugging and the real demo
data share the same graph — there's no free per-run database to isolate into.

- **Convention:** give any debug/exploration `add_message(...)` call a session id
  starting with `debug-` (e.g. `debug-my-experiment`).
- **`make wipe-test-data`** deletes every conversation (and its messages) whose session
  id starts with `debug-`. Fully automatic and safe — session ids are explicit, so
  there's no ambiguity about what gets deleted.
- **Entities are not touched by that command, on purpose.** They're global nodes, not
  scoped to a session, and the bug above means a real, legitimately-ingested entity can
  end up with zero incoming `MENTIONS` edges too — indistinguishable, by that signal
  alone, from a leftover debug entity. An earlier version of this cleanup used "delete
  entities with no incoming MENTIONS" as the rule; it matched almost every entity in the
  database (because of the bug above) and deleted real demo data, including "Maria
  Chen". Don't reintroduce that heuristic.
- **`make wipe-entities NAMES="Name One,Name Two"`** deletes specific entities by exact
  name instead — deliberate, run once you know exactly which names your debug run
  created (e.g. by printing them before wiping, the way you'd inspect any destructive
  command's target first).

## Tuning

`GLINER_MODEL`/`GLINER_DEVICE` and `GLINER_THRESHOLD` are shared `.env` settings, read
by both extractors — but `GLINER_THRESHOLD` means something different in each:

- **`USE_SPACY=true`:** GLiNER's output goes straight into the graph, unfiltered.
  `GLINER_THRESHOLD=0.5` (the default) is GLiNER's own library default — a plain
  confidence cutoff on what becomes a real entity.
- **`USE_SPACY=false`:** GLiNER is just `TypeSafeExtractor`'s candidate finder; TypeSafe
  judges and can reject anything GLiNER proposes. `TypeSafeExtractor` was built assuming
  this, so its `gliner_candidate_extractor(threshold=...)` call passes `GLINER_THRESHOLD`
  too, deliberately low (`0.3` is the project's original tuning for that path) to favor
  recall, since the TypeSafe pass downstream is what rejects the noise.

Raising the shared default to `0.5` (GLiNER's own default, up from `0.3`) trades a bit of
recall on the `USE_SPACY=false` path for cleaner output on `USE_SPACY=true`, where nothing
else filters GLiNER's guesses. If you're running `USE_SPACY=false` and want the original
high-recall behavior back, set `GLINER_THRESHOLD=0.3` in `.env`.

`SPACY_MODEL` defaults to `en_core_web_trf` (transformer-based), not `en_core_web_sm`.
The small model has no per-entity confidence (the library stamps every spaCy entity with
a fixed `0.85`, real or not) and mistyped names it has no domain knowledge of — e.g. it
tagged "Neo4j" and "Pathfinder" as `PERSON`/`ORGANIZATION`, and grabbed "Tom Okafor's"
(with the possessive) as its own entity instead of resolving to "Tom Okafor". The
transformer model uses contextual embeddings rather than surface patterns, so it makes
these particular mistakes less often — it doesn't add real per-entity confidence either,
and it doesn't fix name normalization (possessives, case) since nothing in this pipeline
does that post-processing. It's a ~440MB extra dependency (`spacy-transformers` +
`en_core_web_trf`) and slower per-passage than `en_core_web_sm`.

The rest are `TypeSafeExtractor` constructor arguments (`build_typesafe_extractor`):

| Setting | Default | Effect |
|---|---|---|
| `min_entity_confidence` | `0.5` | TypeSafe choice confidence needed to keep an entity |
| `min_relation_confidence` | `0.6` | Same, for relations |
| `max_questions_per_call` | `25` | Questions batched into one `system_one` request |
| `max_pairs` | `40` | Cap on relation questions per passage |
| `entity_labels` / `relation_labels` | POLE+O / 8 relation types | Your own taxonomy. Both must keep a `NONE` label. |

**Cost model:** each passage costs one question per candidate entity, plus two per pair of
entities that share a sentence (both directions). Questions are batched, so a typical
passage takes 1–3 API calls.

## Known extraction quality gaps (neither extractor fixes these)

- **No cross-passage entity resolution.** "Tom Okafor" and "Tom Okafor's" (or any two
  spellings of the same thing) become two permanent, disconnected nodes. Nothing in this
  project or `neo4j-agent-memory` normalizes or merges near-duplicate names.
- **`USE_SPACY=true` never produces relationships.** GLiNER doesn't extract relations at
  all (by design), and spaCy's NER doesn't either — so that path only ever writes
  `:Entity` nodes and `MENTIONS` edges, never `RELATED_TO`. Only `TypeSafeExtractor`
  (`USE_SPACY=false`) produces relations.
