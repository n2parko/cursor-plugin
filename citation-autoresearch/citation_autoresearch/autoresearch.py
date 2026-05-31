"""The autoresearch loop.

karpathy/autoresearch, adapted: instead of editing `train.py` to lower
val_bpb, the agent edits a citation-prediction *strategy* to raise held-out
ranking of impact. One iteration:

  1. predict held-out papers with the current strategy and evaluate
  2. analyze the worst errors on the train split
  3. propose a revised strategy (Claude in live mode; coordinate descent offline)
  4. re-evaluate; KEEP if the eval score improved, else DISCARD
  5. log the experiment

Run N iterations to search for a better strategy.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from . import config
from .evaluate import Metrics, evaluate
from .features import FEATURE_NAMES, extract_features
from .llm import LLMClient
from .paper import Paper
from .predictor import Strategy, predict_paper


def _percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    idx = min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1))))
    return s[idx]


@dataclass
class Dataset:
    train: list[Paper]
    evalset: list[Paper]
    highly_threshold: int  # citation count at/above which a paper is "highly cited"

    def highly_labels(self, papers: list[Paper]) -> list[int]:
        return [1 if (p.citation_count or 0) >= self.highly_threshold else 0 for p in papers]


def make_dataset(papers: list[Paper]) -> Dataset:
    labeled = [p for p in papers if p.citation_count is not None]
    counts = [float(p.citation_count) for p in labeled]
    threshold = int(_percentile(counts, config.HIGHLY_CITED_PERCENTILE))
    rng = random.Random(config.RANDOM_SEED)
    shuffled = labeled[:]
    rng.shuffle(shuffled)
    n_eval = max(1, int(round(len(shuffled) * config.EVAL_FRACTION)))
    return Dataset(train=shuffled[n_eval:], evalset=shuffled[:n_eval], highly_threshold=threshold)


def evaluate_strategy(strategy: Strategy, papers: list[Paper], ds: Dataset, llm: LLMClient):
    preds = [predict_paper(p, strategy, llm) for p in papers]
    metrics = evaluate(
        pred_log=[pr.log_citations for pr in preds],
        pred_highly=[pr.highly_cited_probability for pr in preds],
        actual_citations=[p.citation_count for p in papers],
        highly_labels=ds.highly_labels(papers),
    )
    return metrics, preds


# --- Strategy improvement ----------------------------------------------------

_IMPROVE_SCHEMA = {
    "type": "object",
    "properties": {
        "guidance": {"type": "string"},
        "base": {"type": "number"},
        "weights": {
            "type": "object",
            "properties": {n: {"type": "number"} for n in FEATURE_NAMES},
            "required": FEATURE_NAMES,
            "additionalProperties": False,
        },
        "rationale": {"type": "string"},
    },
    "required": ["guidance", "base", "weights", "rationale"],
    "additionalProperties": False,
}

_IMPROVE_SYSTEM = (
    "You are tuning a citation-impact prediction strategy. You are given the "
    "current strategy and a set of worst-case errors on a training split (each "
    "with the publication-time signals, the prediction, and the truth). Propose "
    "an improved strategy: revised natural-language guidance, a revised base "
    "log-citation level, and revised per-signal emphasis weights. Keep weights "
    "roughly in [-1, 2]. Return strict JSON."
)


def _train_error_report(strategy: Strategy, ds: Dataset, llm: LLMClient) -> str:
    metrics, preds = evaluate_strategy(strategy, ds.train, ds, llm)
    rows = []
    for p, pr in zip(ds.train, preds):
        actual_log = math.log1p(p.citation_count)
        rows.append((abs(pr.log_citations - actual_log), p, pr, actual_log))
    rows.sort(reverse=True, key=lambda r: r[0])
    lines = [f"Current train metrics: {metrics.summary()}", ""]
    for err, p, pr, actual_log in rows[:5]:
        feats = extract_features(p)
        sig = ", ".join(f"{k}={v:.2f}" for k, v in feats.items())
        lines.append(
            f"- '{p.title[:60]}' ({p.year}): predicted log={pr.log_citations:.2f}, "
            f"actual log={actual_log:.2f} (cites={p.citation_count}); signals: {sig}"
        )
    return "\n".join(lines)


def _propose_live(strategy: Strategy, ds: Dataset, llm: LLMClient) -> Optional[Strategy]:
    user = (
        f"Current guidance: {strategy.guidance}\n"
        f"Current base: {strategy.base}\n"
        f"Current weights: {json.dumps(strategy.weights)}\n\n"
        f"{_train_error_report(strategy, ds, llm)}"
    )
    try:
        out = llm.complete_json(_IMPROVE_SYSTEM, user, _IMPROVE_SCHEMA, "improve-system-v1")
    except Exception:
        return None
    new = strategy.copy()
    new.guidance = str(out["guidance"])
    new.base = float(out["base"])
    new.weights = {n: float(out["weights"][n]) for n in FEATURE_NAMES}
    new.version = strategy.version + 1
    return new


def _propose_offline(strategy: Strategy, ds: Dataset, llm: LLMClient) -> Strategy:
    """Coordinate descent on the train MAE — the offline 'agent'."""

    def train_mae(s: Strategy) -> float:
        m, _ = evaluate_strategy(s, ds.train, ds, llm)
        return m.mae_log - 0.5 * m.spearman  # nudge toward better ranking too

    best = strategy.copy()
    best_score = train_mae(best)
    step = 0.25
    for name in FEATURE_NAMES:
        for delta in (step, -step):
            cand = best.copy()
            cand.weights[name] = max(-1.0, min(2.0, cand.weights[name] + delta))
            sc = train_mae(cand)
            if sc < best_score:
                best, best_score = cand, sc
    for delta in (0.5, -0.5):
        cand = best.copy()
        cand.base = cand.base + delta
        sc = train_mae(cand)
        if sc < best_score:
            best, best_score = cand, sc
    best.version = strategy.version + 1
    best.guidance = strategy.guidance + " [auto-tuned weights]"
    return best


def propose_strategy(strategy: Strategy, ds: Dataset, llm: LLMClient) -> Strategy:
    if llm.mode == "live":
        improved = _propose_live(strategy, ds, llm)
        if improved is not None:
            return improved
    return _propose_offline(strategy, ds, llm)


# --- One iteration -----------------------------------------------------------

@dataclass
class IterationResult:
    iteration: int
    before: Metrics
    after: Metrics
    kept: bool
    strategy: Strategy
    proposed: Strategy


def run_iteration(
    iteration: int,
    strategy: Strategy,
    ds: Dataset,
    llm: LLMClient,
) -> IterationResult:
    before, _ = evaluate_strategy(strategy, ds.evalset, ds, llm)
    proposed = propose_strategy(strategy, ds, llm)
    after, _ = evaluate_strategy(proposed, ds.evalset, ds, llm)
    kept = after.score() > before.score()
    chosen = proposed if kept else strategy
    result = IterationResult(iteration, before, after, kept, chosen, proposed)
    _log(result)
    return result


def _log(result: IterationResult) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "iteration": result.iteration,
        "kept": result.kept,
        "before": result.before.__dict__,
        "after": result.after.__dict__,
        "strategy_version": result.strategy.version,
        "weights": result.strategy.weights,
        "base": result.strategy.base,
    }
    with config.RUN_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
