# TypeSafe × Neo4j Agent Memory — POC

A LangChain-based agent whose memory graph on **Neo4j Aura**  with the ingestion pipeline using **TypeSafe's Jev** instead of **SpaCy**.

Here, extraction is swapped for a two-stage extractor:

```
text ──► GLiNER (local, low threshold)  ──► candidate spans         "recall"
     ──► TypeSafe system_one            ──► POLE+O type or NONE     "precision"
     ──► TypeSafe system_one (pairs)    ──► WORKS_FOR / KNOWS / …   "relations"
     ──► neo4j-agent-memory             ──► (:Entity) nodes, [:MENTIONS], [:RELATED_TO] in Aura
```

`TypeSafeExtractor` implements the library's `EntityExtractor` protocol and is passed as
`MemoryClient(settings, extractor=...)`. So **every** message stored with
`extract_entities=True` goes through TypeSafe. That covers ingested documents, and it also
covers the chat turns that `Neo4jMemoryMiddleware` persists while the agent runs.

## Files

| File | What it is |
|---|---|
| `typesafe_extractor.py` | The extractor: GLiNER candidates → TypeSafe typing + relations. No Neo4j or LangChain code. |
| `langchain_agent_memory.py` | Memory wiring (Aura + TypeSafe extractor), document ingestion, and the LangChain agent. |
| `checks.py` | Health checks behind the `make check-*` targets. |
| `_env.py` | Loads `.env`. |
| `data/docs/` | Sample documents for `make ingest`. |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) and Python 3.11–3.13
- A **Neo4j Aura** instance (the free tier works). Keep the credentials `.txt` Aura gives you on creation.
- A **TypeSafe API key** ([typesafe.ai](https://typesafe.ai))
- An **OpenAI API key**. It's used for embeddings (`openai/text-embedding-3-small` by default).
  After `make install-openai` it also gives the agent a real chat model.
  To run without it, set `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` for local embeddings.
- About 2 GB of disk for torch and GLiNER (~800 MB). GLiNER downloads on first use.

## Run it

```bash
# 1. Install
make install            # or `make install-openai` to also get langchain-openai

# 2. Configure
make env                # creates .env from .env.example
#    edit .env: NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD / NEO4J_DATABASE
#               from the Aura credentials file, TYPESAFE_API_KEY, OPENAI_API_KEY

# 3. Verify each piece (run individually, or all at once with `make check`)
make check-env          # required values present
make check-neo4j        # Aura reachable, credentials valid, can write
make check-typesafe     # API key valid, one question answered
make check-extractor    # GLiNER + TypeSafe find the expected entities (no Neo4j)
make check-memory       # neo4j-agent-memory connects to Aura, round-trips a message

# 4. Build the graph and talk to it
make ingest             # data/docs → TypeSafe → Aura
make graph              # what was written
make ask Q="Who leads Acme Robotics' Berlin office?"
make demo               # ingest + one question in a single run
```

`make help` lists every target. To ingest your own files, run `make ingest DOCS="path/to/dir_or_file.md"`.
`.md` and `.txt` files are split into passages on blank lines, and lines starting with `#` are skipped.

## What each check proves

| Target | Passes when | Typical failures |
|---|---|---|
| `check-env` | `NEO4J_URI`, `NEO4J_PASSWORD`, `TYPESAFE_API_KEY` are set, plus `OPENAI_API_KEY` when using OpenAI embeddings | `.env` missing → `make env` |
| `check-neo4j` | Driver connects, authenticates, reads server version, **creates and deletes a test node**, and counts nodes | Wrong password (`AuthError`); paused free instance or wrong URI (`ServiceUnavailable`); `bolt://` or `neo4j://` instead of `neo4j+s://`; wrong `NEO4J_DATABASE` |
| `check-typesafe` | Lists your models and classifies "Berlin" as `LOCATION` | Bad or missing API key |
| `check-extractor` | GLiNER finds candidates in a sample sentence, and TypeSafe types Maria Chen / Acme Robotics / Berlin correctly. Relations are reported but don't fail the check. | GLiNER model download failed; TypeSafe errors |
| `check-memory` | `MemoryClient` connects, creates its schema and indexes, stores a message with an OpenAI embedding, reads it back, and cleans up | Bad `OPENAI_API_KEY`; vector-index dimension mismatch (see below); Aura permissions |

`check-neo4j` uses the raw driver and `check-memory` uses the memory library. When one passes and the other fails, that tells you which layer is broken.

## Viewing the graph in Aura

Open your instance in the [Aura console](https://console.neo4j.io) and run this in the Query tab:

```cypher
// Entities and the relations TypeSafe found
MATCH p = (:Entity)-[:RELATED_TO]->(:Entity) RETURN p;

// Which passages mention which entities
MATCH p = (:Message)-[:MENTIONS]->(:Entity) RETURN p LIMIT 100;
```

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

## Known limitations

- **Relations only between entities in the same sentence.** Pronouns aren't resolved.
  "She hired Tom" doesn't link Maria to Tom. Keep passages short and explicit, or widen `_co_occurring_pairs`.
- **One relationship per ordered pair.** The memory library stores relations as `(a)-[:RELATED_TO {relation_type}]->(b)` and merges on the pair.
  A second relation type between the same two entities updates confidence and keeps the first type.
- **No preference extraction.** `extract_preferences` is accepted and ignored. The agent's `save_preference` tool writes preferences explicitly.
- **Don't change `EMBEDDING_MODEL` after the first ingest.** The Aura vector index is sized to the model:
  1536 dims for `text-embedding-3-small`, 3072 for `-large`, 384 for MiniLM.
  Switching means dropping the vector indexes (`SHOW INDEXES` → `DROP INDEX <name>`) or using a fresh database.
- **Offline agent.** Without `OPENAI_API_KEY` the agent answers with a scripted fake model and has no tools.
  Memory injection, message persistence and TypeSafe extraction still run.
- **Sessions.** Documents go to `INGEST_SESSION_ID` and chat to `CHAT_SESSION_ID`. The `search_memory` tool searches across both.
  Nothing here deletes data. To start over, clear the Aura database from the console.
