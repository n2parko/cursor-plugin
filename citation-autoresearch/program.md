# Autoresearch task: predict citation impact

You are an autoresearch agent. Your objective is to **maximize held-out ranking
of research-paper impact**, where impact is proxied by future citation count.

## The metric

The loop scores a strategy on a held-out eval split with:

- **Spearman** rank correlation between predicted and actual citations
- **highly-cited AUC** — ranking quality for the top-quartile (impact) class
- **MAE(log)** — calibration of the predicted ln(1+citations)

The combined objective is `Spearman + AUC - 0.1*MAE(log)` (higher is better).
A proposed strategy is **kept only if it beats the current one** on this score.

## What you may edit

Only the `Strategy` (in `predictor.py`): its natural-language `guidance`, the
`base` log-citation level, and the per-signal `weights`. You may **not** use the
ground-truth `citation_count` as an input — predictions must rely solely on
publication-time signals (title, abstract, venue, fields, author track record,
references). Using the label is leakage and invalidates the result.

## The loop (per iteration)

1. Predict the eval split with the current strategy; record the score.
2. Inspect the worst errors on the **train** split (never the eval split).
3. Propose a revised strategy that should reduce those errors.
4. Re-evaluate on eval; keep if better, else discard.
5. Repeat under your iteration budget.

## Guidance for good strategies

- Foundational, broadly-applicable contributions cite far more than incremental
  ones — read the abstract for scope, not hype.
- Venue and field have strong citation priors; author track record is an early
  signal but is noisy.
- Calibrate `base` so typical predictions land in ln(1+citations) ≈ 9-12 for
  landmark ML papers and lower for niche work.
