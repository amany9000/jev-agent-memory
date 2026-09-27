.DEFAULT_GOAL := help
RUN := uv run --no-sync python
PYTEST := uv run --no-sync pytest
Q ?= Who leads Acme Robotics' Berlin office, and what do they use?

.PHONY: help install env \
        test test-offline test-typesafe \
        fix-vector-indexes wipe-test-data wipe-entities \
        ingest ask demo graph clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# --- setup -------------------------------------------------------------------
install: ## Install dependencies (incl. dev/test deps) into .venv (uv)
	uv sync

env: ## Create .env from .env.example (never overwrites)
	@if [ -f .env ]; then echo ".env already exists — edit it directly"; \
	else cp .env.example .env && echo "Created .env — fill in NEO4J_* and TYPESAFE_API_KEY"; fi

# --- tests ---------------------------------------------------------------
# Neo4j/DeepSeek tests skip cleanly (not fail) without credentials; TypeSafe
# tests are excluded unless explicitly requested (see test-typesafe) — nothing
# here spends TypeSafe credits by accident.
test: ## Run the suite (offline logic + GLiNER + Neo4j/DeepSeek if configured). No TypeSafe calls.
	$(PYTEST) -v

test-offline: ## Run only what needs no network at all (extractor logic + GLiNER + config)
	$(PYTEST) -v tests/test_config.py tests/test_extractor.py

test-typesafe: ## Run the ONE real TypeSafe API test (spends credits) — use sparingly
	$(PYTEST) -v --run-typesafe tests/test_typesafe.py

fix-vector-indexes: ## Drop Aura vector indexes sized for a different embedding model
	$(RUN) utils.py fix-vector-indexes

# Debug/exploration data pollutes the same graph as the demo data (see utils.py's
# module docstring — Aura Free has no per-run database to isolate into). Use a
# session id prefixed "debug-" in ad hoc scripts, then:
wipe-test-data: ## Delete debug-prefixed conversations/messages (entities untouched — see utils.py)
	$(RUN) utils.py wipe-test-data

wipe-entities: ## Delete specific entities by exact name (NAMES="Name One,Name Two")
	$(RUN) utils.py wipe-entities "$(NAMES)"

# --- run ---------------------------------------------------------------------
ingest: ## Ingest data/docs into the memory graph (spaCy + GLiNER; DOCS=path to override)
	$(RUN) langchain_agent_memory.py ingest $(DOCS)

ask: ## Ask the memory-backed agent (Q="your question")
	$(RUN) langchain_agent_memory.py ask "$(Q)"

demo: ## Ingest data/docs, then ask one question
	$(RUN) langchain_agent_memory.py demo --question "$(Q)"

graph: ## Show the entities and relationships written to Aura
	@$(RUN) utils.py graph

clean: ## Remove caches (keeps .venv and .env)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
