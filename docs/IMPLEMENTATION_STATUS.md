# Implementation Status and Reproducibility

This repository contains an **engineering prototype**, not a production-deployed system.

## Available features

Patent demonstration corpus and manually written source notes; claims parser and lexical comparison; BM25 and TF-IDF retrieval; optional dense retrieval/reranking interfaces; API routes, browser demo, MCP tools and regression tests.

## Validation caveats

Evaluation sets are small and partly synthetic or manually constructed. Metrics are not evidence of general business accuracy. Optional provider-backed LLM calls, real patent APIs, end-to-end model inference, external MCP clients and Docker deployment must be verified separately in the user environment. Patent comparisons are **not legal novelty or freedom-to-operate opinions**.

## Quick checks

```bash
python -m pip install -e ".[dev]"
pytest -q
```
