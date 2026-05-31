"""Metrics for citation / impact prediction. Pure-Python (no numpy/scipy)."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Metrics:
    n: int
    mae_log: float          # mean abs error on log1p(citations) — regression
    spearman: float         # rank correlation of predicted vs actual citations
    highly_cited_auc: float # ranking quality for the "highly cited" class
    precision_at_k: float   # precision@k where k = #positives (impact hit-rate)

    def summary(self) -> str:
        return (
            f"n={self.n}  MAE(log)={self.mae_log:.3f}  "
            f"Spearman={self.spearman:+.3f}  "
            f"highly-cited AUC={self.highly_cited_auc:.3f}  "
            f"P@k={self.precision_at_k:.3f}"
        )

    def score(self) -> float:
        """Single scalar the loop maximizes (higher = better)."""
        return self.spearman + self.highly_cited_auc - 0.1 * self.mae_log


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if va == 0 or vb == 0:
        return 0.0
    return cov / (va * vb)


def spearman(a: list[float], b: list[float]) -> float:
    return _pearson(_rank(a), _rank(b))


def _auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def precision_at_k(scores: list[float], labels: list[int]) -> float:
    k = sum(labels)
    if k == 0:
        return 0.0
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return sum(labels[i] for i in top) / k


def evaluate(
    pred_log: list[float],
    pred_highly: list[float],
    actual_citations: list[int],
    highly_labels: list[int],
) -> Metrics:
    actual_log = [math.log1p(c) for c in actual_citations]
    mae = sum(abs(p - a) for p, a in zip(pred_log, actual_log)) / len(pred_log)
    return Metrics(
        n=len(pred_log),
        mae_log=mae,
        spearman=spearman(pred_log, actual_log),
        highly_cited_auc=_auc(pred_highly, highly_labels),
        precision_at_k=precision_at_k(pred_highly, highly_labels),
    )
