# Reference

Supporting detail split out of [README.md](README.md): what each file does, and how to
tune the extractor. Start with the README for setup and running; come here once that's
working and you want to change how it behaves.

## Files

| File | What it is |
|---|---|
| `typesafe_extractor.py` | The extractor: GLiNER candidates → TypeSafe typing + relations. No Neo4j or LangChain code. |
| `langchain_agent_memory.py` | Memory wiring (Aura + TypeSafe extractor), document ingestion, and the LangChain agent. |
| `utils.py` | Operational commands: `graph` (show what's in Aura), `fix-vector-indexes`. |
| `fastembed_embedder.py` | FastEmbed adapter for the memory library's `EmbeddingProvider` protocol. |
| `_env.py` | Loads `.env`. |
| `tests/` | pytest suite — see **Testing** in the README. |
| `data/docs/` | Sample documents for `make ingest`. |

## Tuning

These are constructor arguments of `TypeSafeExtractor` (set them in `build_extractor`) or `.env` values:

| Setting | Default | Effect |
|---|---|---|
| `GLINER_THRESHOLD` | `0.3` | Lower gives more candidates for TypeSafe to judge. Higher gives fewer TypeSafe questions. |
| `min_entity_confidence` | `0.5` | TypeSafe choice confidence needed to keep an entity |
| `min_relation_confidence` | `0.6` | Same, for relations |
| `max_questions_per_call` | `25` | Questions batched into one `system_one` request |
| `max_pairs` | `40` | Cap on relation questions per passage |
| `entity_labels` / `relation_labels` | POLE+O / 8 relation types | Your own taxonomy. Both must keep a `NONE` label. |

**Cost model:** each passage costs one question per candidate entity, plus two per pair of
entities that share a sentence (both directions). Questions are batched, so a typical
passage takes 1–3 API calls.
