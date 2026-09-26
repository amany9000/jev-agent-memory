.DEFAULT_GOAL := help
RUN := uv run --no-sync python
Q ?= Who leads Acme Robotics' Berlin office, and what do they use?

.PHONY: help install install-openai env \
        check check-env check-neo4j check-memory check-typesafe check-extractor \
        ingest ask demo graph clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# --- setup -------------------------------------------------------------------
install: ## Install dependencies into .venv (uv)
	uv sync

install-openai: ## Install deps + langchain-openai for a real chat model
	uv sync --extra openai

env: ## Create .env from .env.example (never overwrites)
	@if [ -f .env ]; then echo ".env already exists — edit it directly"; \
	else cp .env.example .env && echo "Created .env — fill in NEO4J_* and TYPESAFE_API_KEY"; fi

# --- checks ------------------------------------------------------------------
check-env: ## Required settings are present in .env
	@$(RUN) checks.py env

check-neo4j: ## Aura reachable, credentials valid, user can write
	@$(RUN) checks.py neo4j

check-memory: ## neo4j-agent-memory connects to Aura and round-trips a message
	@$(RUN) checks.py memory

check-typesafe: ## TypeSafe API key works and answers a question
	@$(RUN) checks.py typesafe

check-extractor: ## GLiNER + TypeSafe extract the expected entities from a sample (no Neo4j)
	@$(RUN) checks.py extractor

check: check-env check-neo4j check-typesafe check-extractor check-memory ## Run every check, in dependency order

# --- run ---------------------------------------------------------------------
ingest: ## Ingest data/docs into the memory graph via TypeSafe (DOCS=path to override)
	$(RUN) langchain_agent_memory.py ingest $(DOCS)

ask: ## Ask the memory-backed agent (Q="your question")
	$(RUN) langchain_agent_memory.py ask "$(Q)"

demo: ## Ingest data/docs, then ask one question
	$(RUN) langchain_agent_memory.py demo --question "$(Q)"

graph: ## Show the entities and relationships TypeSafe has written
	@$(RUN) checks.py graph

clean: ## Remove caches (keeps .venv and .env)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
