"""The prediction strategy — the part the autoresearch agent improves.

Analogous to `train.py` in karpathy/autoresearch: this is the file/object the
loop edits in search of a better metric. A ``Strategy`` bundles:

  - ``guidance``: natural-language instructions handed to Claude
  - ``base`` + ``weights``: a per-feature emphasis vector. Claude receives these
    as hints; the offline heuristic uses them numerically. Either way, the loop
    tunes them to improve held-out ranking of impact.

``predict_paper`` produces, for one paper, a predicted log-citation count and a
probability that the paper becomes "highly cited" (the impact proxy).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict

from .features import FEATURE_NAMES, extract_features, format_features_text
from .llm import LLMClient
from .paper import Paper

# JSON schema Claude must return for a single prediction.
_PRED_SCHEMA = {
    "type": "object",
    "properties": {
        "log_citations": {"type": "number"},
        "highly_cited_probability": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["log_citations", "highly_cited_probability", "reasoning"],
    "additionalProperties": False,
}

_SYSTEM = """You are a scientometrics expert predicting the future citation impact \
of a research paper using ONLY information available at publication time \
(title, abstract, venue, fields, author track record, reference list).

You are forbidden from using any knowledge of the paper's actual citation count.
Reason from early signals of impact: novelty and clarity of the contribution, \
breadth of applicability, venue and field citation norms, author track record, \
and how much the abstract reads like a foundational vs. incremental result.

Output strict JSON:
  - log_citations: your prediction of ln(1 + total citations), a float \
(typical range 2-13; a landmark ML paper can exceed 11).
  - highly_cited_probability: probability in [0,1] that this paper lands in the \
top quartile of impact for its cohort.
  - reasoning: one or two sentences."""


@dataclass
class Strategy:
    guidance: str = (
        "Weight foundational, broadly-applicable contributions and strong author "
        "track records most heavily. Treat hype words alone as weak evidence."
    )
    base: float = 6.0
    weights: Dict[str, float] = field(
        default_factory=lambda: {name: 0.5 for name in FEATURE_NAMES}
    )
    version: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def copy(self) -> "Strategy":
        return Strategy(
            guidance=self.guidance,
            base=self.base,
            weights=dict(self.weights),
            version=self.version,
        )


@dataclass
class Prediction:
    paper_id: str
    log_citations: float
    highly_cited_probability: float
    reasoning: str


def _heuristic(paper: Paper, strategy: Strategy) -> Prediction:
    """Deterministic offline predictor used when Claude is unavailable.

    A transparent linear model over the normalized features, parameterized by
    the strategy's ``base`` + ``weights`` so the loop can genuinely improve it.
    """
    feats = extract_features(paper)
    score = strategy.base + sum(strategy.weights.get(k, 0.0) * v for k, v in feats.items())
    log_cit = max(0.0, score)
    # Squash the same score into a highly-cited probability.
    centered = (log_cit - 9.0) / 2.0
    prob = 1.0 / (1.0 + pow(2.718281828, -centered))
    return Prediction(
        paper_id=paper.paper_id,
        log_citations=log_cit,
        highly_cited_probability=prob,
        reasoning="offline linear heuristic over publication-time signals",
    )


def _llm_user_prompt(paper: Paper, strategy: Strategy) -> str:
    weights = ", ".join(f"{k}:{strategy.weights[k]:+.2f}" for k in FEATURE_NAMES)
    return (
        f"{format_features_text(paper)}\n\n"
        f"Current strategy guidance: {strategy.guidance}\n"
        f"Current per-signal emphasis (hints, not hard rules): {weights}\n\n"
        "Predict this paper's citation impact as instructed."
    )


def predict_paper(paper: Paper, strategy: Strategy, llm: LLMClient) -> Prediction:
    if llm.mode != "live":
        return _heuristic(paper, strategy)
    try:
        out = llm.complete_json(
            system=_SYSTEM,
            user=_llm_user_prompt(paper, strategy),
            schema=_PRED_SCHEMA,
            cache_key="predict-system-v1",
        )
    except Exception:
        return _heuristic(paper, strategy)
    return Prediction(
        paper_id=paper.paper_id,
        log_citations=float(out["log_citations"]),
        highly_cited_probability=float(out["highly_cited_probability"]),
        reasoning=str(out.get("reasoning", "")),
    )
