"""Fetch paper metadata + ground-truth citation counts.

Primary source: Semantic Scholar Graph API (free, no key) — gives title,
abstract, year, venue, fields, references, authors *and* the current
citationCount we use as the impact label. arXiv is a fallback for metadata.

All network calls degrade gracefully: if a host is unreachable (restricted
sandbox, offline), we fall back to the bundled seed dataset so the loop still
runs. Responses are cached under ``.cache/``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import requests

from . import config
from .paper import Paper


def _cache_path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return config.CACHE_DIR / f"{safe}.json"


def _cached_get(url: str, params: dict, key: str) -> Optional[dict]:
    cp = _cache_path(key)
    if cp.exists():
        return json.loads(cp.read_text())
    try:
        resp = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        cp.write_text(json.dumps(data))
        return data
    except Exception:
        return None


def _paper_from_s2(s2: dict, fallback_id: str) -> Paper:
    authors = s2.get("authors") or []
    prior = [a.get("citationCount") for a in authors if a.get("citationCount") is not None]
    hidx = [a.get("hIndex") for a in authors if a.get("hIndex") is not None]
    return Paper(
        paper_id=s2.get("paperId") or fallback_id,
        title=s2.get("title") or "",
        abstract=s2.get("abstract") or "",
        year=s2.get("year"),
        venue=s2.get("venue") or "",
        fields=s2.get("fieldsOfStudy") or [],
        author_names=[a.get("name", "") for a in authors],
        reference_count=s2.get("referenceCount"),
        author_prior_citations=max(prior) if prior else None,
        author_max_h_index=max(hidx) if hidx else None,
        citation_count=s2.get("citationCount"),
    )


def fetch_paper(identifier: str) -> Optional[Paper]:
    """Fetch a single paper by Semantic Scholar id, ``arXiv:<id>``, or DOI."""
    data = _cached_get(
        f"{config.S2_API}/{identifier}",
        {"fields": config.S2_FIELDS},
        key=f"s2_{identifier}",
    )
    if data and not data.get("error"):
        return _paper_from_s2(data, identifier)
    return None


def load_seed_papers() -> list[Paper]:
    raw = json.loads(config.SEED_DATASET.read_text())
    return [Paper.from_dict(d) for d in raw]


def build_dataset(enrich: bool = True) -> tuple[list[Paper], str]:
    """Return (papers, source_note).

    Starts from the bundled seed set (which carries snapshot citation counts so
    the loop is reproducible offline). When ``enrich`` and the network is
    reachable, refresh metadata + citation counts from Semantic Scholar.
    """
    papers = load_seed_papers()
    if not enrich:
        return papers, "bundled seed dataset (offline)"

    refreshed = 0
    for p in papers:
        ident = None
        # seed records may carry an "arxiv_id" or "s2_id" hint in paper_id
        if p.paper_id.startswith("arXiv:") or p.paper_id.startswith("10."):
            ident = p.paper_id
        if ident:
            live = fetch_paper(ident)
            if live and live.citation_count is not None:
                live.paper_id = p.paper_id
                # keep seed abstract if S2 omitted it
                if not live.abstract:
                    live.abstract = p.abstract
                papers[papers.index(p)] = live
                refreshed += 1

    if refreshed:
        return papers, f"seed dataset, {refreshed}/{len(papers)} refreshed from Semantic Scholar"
    return papers, "bundled seed dataset (network unreachable; using snapshot counts)"
