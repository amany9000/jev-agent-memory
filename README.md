# TypeSafe × Neo4j Agent Memory 

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

See [docs.md](docs.md) for what each file does and how to tune the extractor.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) and Python 3.11–3.13
- A **Neo4j Aura** instance (the free tier works). Keep the credentials `.txt` Aura gives you on creation.
- A **TypeSafe API key** ([typesafe.ai](https://typesafe.ai))
- Optional: a **DeepSeek API key** ([platform.deepseek.com](https://platform.deepseek.com)) for the agent's chat model.
  Without it the agent runs on a scripted fake model.
- Embeddings need no key. They come from **FastEmbed** (`BAAI/bge-small-en-v1.5`, 384 dims), which runs locally and downloads ~70 MB on first use.
- About 2 GB of disk for torch and GLiNER (~800 MB). GLiNER downloads on first use.

## Run it

```bash
# 1. Install
make install

# 2. Configure
make env                # creates .env from .env.example
#    edit .env: NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD / NEO4J_DATABASE
#               from the Aura credentials file, TYPESAFE_API_KEY, and optionally DEEPSEEK_API_KEY

# 3. Test each piece — see "Testing" below for what runs and what it costs
make test               # everything except the real TypeSafe call: free, safe to run often
make test-typesafe      # the one test that spends TypeSafe credits — run deliberately

# 4. Build the graph and talk to it
make ingest             # data/docs → TypeSafe → Aura
make graph              # what was written
make ask Q="Who leads Acme Robotics' Berlin office?"
make demo               # ingest + one question in a single run
```

`make help` lists every target. To ingest your own files, run `make ingest DOCS="path/to/dir_or_file.md"`.
`.md` and `.txt` files are split into passages on blank lines, and lines starting with `#` are skipped.

## Testing

The suite is pytest, under `tests/`. It's split so that iterating on the extractor's
logic — the part you're most likely to change — never touches a paid API, and the
one test that does is never run by accident.

| Command | Runs | Cost |
|---|---|---|
| `make test-offline` | `tests/test_config.py`, `tests/test_extractor.py` | Free. Local only (GLiNER runs, but no network). |
| `make test` | Everything **except** `tests/test_typesafe.py` | Free. Neo4j/DeepSeek tests **skip** (not fail) if their credentials aren't set. |
| `make test-typesafe` | The one test in `tests/test_typesafe.py` | **Spends TypeSafe credits.** One `system_one` call, one question. |

`pytest` on its own behaves like `make test`: `tests/test_typesafe.py` is marked
`@pytest.mark.typesafe`, and `tests/conftest.py` skips every test with that marker
unless you pass `--run-typesafe` — so nothing in a normal test run, an IDE's "run
all tests", or CI calls TypeSafe. `make test-typesafe` is the only path that does.

| File | What it covers | Needs |
|---|---|---|
| `test_config.py` | `.env` names exist; `EMBEDDING_MODEL` resolves to a real FastEmbed model | nothing |
| `test_extractor.py` | `TypeSafeExtractor`'s own logic — dedup, batching, direction, confidence filtering, symmetric-relation dedup — against `FakeTypeSafeClient`, a stub that answers from a lookup table. Also runs real GLiNER (local, free) to confirm it finds the sample entities. | nothing (GLiNER model download on first run) |
| `test_neo4j.py` | Aura connects, authenticates, can write, reports a version | `NEO4J_URI` / `NEO4J_PASSWORD` — **skips** without them |
| `test_memory.py` | `MemoryClient` + FastEmbed round-trip a message on Aura | same as above |
| `test_deepseek.py` | One short `ChatDeepSeek` call | `DEEPSEEK_API_KEY` — **skips** without it |
| `test_typesafe.py` | One real `system_one` call classifies "Berlin" as `LOCATION` | `TYPESAFE_API_KEY`, and `--run-typesafe` |

If `test_memory.py` fails with a dimension mismatch, run `make fix-vector-indexes`
(see below) and retry — that's a leftover index from a previous embedding model, not
a code problem.

To debug one layer in isolation: `uv run pytest tests/test_neo4j.py -v` (raw driver)
vs `tests/test_memory.py -v` (through the memory library) tells you which layer broke
if one passes and the other doesn't.

Extending the extractor's logic? Add a case to `test_extractor.py` against
`FakeTypeSafeClient` — it's free and instant. Reserve `test_typesafe.py` for
confirming the real API contract hasn't changed, not for logic you can test offline.

## Viewing the graph in Aura

Open your instance in the [Aura console](https://console.neo4j.io) and run this in the Query tab:

```cypher
// Entities and the relations TypeSafe found
MATCH p = (:Entity)-[:RELATED_TO]->(:Entity) RETURN p;

// Which passages mention which entities
MATCH p = (:Message)-[:MENTIONS]->(:Entity) RETURN p LIMIT 100;
```

See [docs.md](docs.md) for tuning `TypeSafeExtractor` (thresholds, batching, your own
entity/relation taxonomy) and the per-passage cost model.
