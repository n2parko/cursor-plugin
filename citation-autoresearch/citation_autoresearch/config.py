"""Fixed configuration for the citation-prediction autoresearch loop.

Mirrors the role of `prepare.py` in karpathy/autoresearch: constants and knobs
that the autoresearch agent is NOT meant to edit. The strategy that the agent
*does* edit lives in `predictor.py`.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Model -------------------------------------------------------------------
# Default to the most capable Claude model. Override with CITATION_AR_MODEL.
MODEL = os.environ.get("CITATION_AR_MODEL", "claude-opus-4-8")

# Effort for the prediction/strategy calls (low|medium|high|xhigh|max).
EFFORT = os.environ.get("CITATION_AR_EFFORT", "high")

# --- Paths -------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
SEED_DATASET = DATA_DIR / "seed_papers.json"
CACHE_DIR = PROJECT_DIR / ".cache"
RUN_LOG = PROJECT_DIR / "autoresearch_log.jsonl"

# --- Data sources ------------------------------------------------------------
S2_API = "https://api.semanticscholar.org/graph/v1/paper"
S2_FIELDS = (
    "title,abstract,year,venue,citationCount,referenceCount,"
    "fieldsOfStudy,authors.name,authors.hIndex,authors.citationCount"
)
ARXIV_API = "http://export.arxiv.org/api/query"
HTTP_TIMEOUT = float(os.environ.get("CITATION_AR_HTTP_TIMEOUT", "10"))

# --- Loop --------------------------------------------------------------------
# Fraction of the dataset held out for evaluation (rest is used by the agent to
# analyze its errors and revise the strategy).
EVAL_FRACTION = 0.4
RANDOM_SEED = 13

# A paper is "highly cited" (positive class) if its citation count is at or
# above this percentile within the dataset. The proxy-for-impact target.
HIGHLY_CITED_PERCENTILE = 75

CACHE_DIR.mkdir(exist_ok=True)
