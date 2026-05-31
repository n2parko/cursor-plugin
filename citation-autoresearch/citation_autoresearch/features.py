"""Publication-time feature extraction.

The arXiv paper (2601.13627) shows that LLMs can read early signals of impact
from title/abstract/metadata available when a paper is published. We extract a
small, interpretable set of such signals. They are deliberately normalized to
roughly [0, 1] so both the Claude predictor (as emphasis hints) and the offline
heuristic fallback can consume them.
"""
from __future__ import annotations

import re
from typing import Dict

from .paper import Paper

# Lexical cues. Crude on purpose — the autoresearch loop is what tunes how much
# weight each cue deserves.
_NOVELTY_TERMS = re.compile(
    r"\b(novel|new|first|propose|introduce|state[- ]of[- ]the[- ]art|outperform|"
    r"breakthrough|unprecedented|significantly)\b",
    re.I,
)
_METHOD_TERMS = re.compile(
    r"\b(transformer|attention|convolutional|residual|recurrent|embedding|"
    r"pre[- ]?train|fine[- ]?tun|self[- ]?supervis|generative|diffusion|"
    r"reinforcement|gradient|optimizer|architecture|network)\b",
    re.I,
)
_BENCHMARK_TERMS = re.compile(
    r"\b(benchmark|dataset|imagenet|glue|coco|wmt|accuracy|f1|bleu|leaderboard|"
    r"evaluat)\b",
    re.I,
)
_HOT_FIELDS = {
    "computer science",
    "machine learning",
    "artificial intelligence",
    "mathematics",
}

# Order is fixed: the heuristic predictor relies on a stable feature vector.
FEATURE_NAMES = [
    "year_recency",
    "n_authors",
    "reference_count",
    "abstract_len",
    "title_len",
    "novelty",
    "method_terms",
    "has_benchmark",
    "author_prior",
    "hot_field",
]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def extract_features(paper: Paper) -> Dict[str, float]:
    abstract = paper.abstract or ""
    title = paper.title or ""
    n_abs_words = len(abstract.split())
    n_title_words = len(title.split())

    year = paper.year or 2015
    fields_lc = {f.lower() for f in (paper.fields or [])}

    # author_prior is neutral (0.5) when unknown so it neither helps nor hurts.
    if paper.author_prior_citations is not None:
        author_prior = _clip01(paper.author_prior_citations / 50000.0)
    elif paper.author_max_h_index is not None:
        author_prior = _clip01(paper.author_max_h_index / 100.0)
    else:
        author_prior = 0.5

    return {
        "year_recency": _clip01((year - 2010) / 15.0),
        "n_authors": _clip01(paper.n_authors / 10.0),
        "reference_count": _clip01((paper.reference_count or 25) / 60.0),
        "abstract_len": _clip01(n_abs_words / 300.0),
        "title_len": _clip01(n_title_words / 20.0),
        "novelty": _clip01(len(_NOVELTY_TERMS.findall(f"{title} {abstract}")) / 5.0),
        "method_terms": _clip01(len(_METHOD_TERMS.findall(f"{title} {abstract}")) / 5.0),
        "has_benchmark": 1.0 if _BENCHMARK_TERMS.search(f"{title} {abstract}") else 0.0,
        "author_prior": author_prior,
        "hot_field": 1.0 if fields_lc & _HOT_FIELDS else 0.0,
    }


def feature_vector(paper: Paper) -> list[float]:
    f = extract_features(paper)
    return [f[name] for name in FEATURE_NAMES]


def format_features_text(paper: Paper) -> str:
    """Human/LLM-readable rendering of the publication-time signals."""
    f = extract_features(paper)
    lines = [
        f"Title: {paper.title}",
        f"Year: {paper.year}",
        f"Venue: {paper.venue or 'unknown'}",
        f"Fields: {', '.join(paper.fields) or 'unknown'}",
        f"# authors: {paper.n_authors}",
        f"# references: {paper.reference_count if paper.reference_count is not None else 'unknown'}",
        (
            "Author track record: "
            + (
                f"~{paper.author_prior_citations} prior citations"
                if paper.author_prior_citations is not None
                else (
                    f"max h-index {paper.author_max_h_index}"
                    if paper.author_max_h_index is not None
                    else "unknown"
                )
            )
        ),
        "",
        "Abstract:",
        (paper.abstract or "(no abstract)").strip(),
        "",
        "Derived signals (normalized 0-1): "
        + ", ".join(f"{k}={v:.2f}" for k, v in f.items()),
    ]
    return "\n".join(lines)
