# Citation-Prediction Autoresearch Loop

An [autoresearch](https://github.com/karpathy/autoresearch)-style loop that
predicts a paper's **future citation count as a proxy for impact**, and then
iteratively improves its own prediction strategy.

It combines two ideas:

- **[arXiv:2601.13627](https://arxiv.org/abs/2601.13627)** — *"Are Large Language
  Models able to Predict Highly Cited Papers?"* — uses an LLM with structured
  prompts over **publication-time** signals (title, abstract, keywords,
  bibliographic metadata) to predict whether a paper becomes highly cited.
- **[karpathy/autoresearch](https://github.com/karpathy/autoresearch)** — an
  agent loop that edits a strategy, runs a short experiment, evaluates a metric,
  and **keeps or discards** the change, repeating under a budget.

Here, the "strategy" being optimized is a citation-impact predictor, and the
metric is held-out ranking of impact.

## How it works

```
seed papers (+ Semantic Scholar / arXiv enrichment)
      │
      ▼
publication-time features ──► Strategy (guidance + base + per-signal weights)
      │                              │
      ▼                              ▼
   Claude predictor  ──►  predicted log-citations + highly-cited probability
      │
      ▼
   evaluate on held-out eval split  (Spearman, highly-cited AUC, MAE-log)
      │
      ▼
   analyze worst train errors ──► propose revised strategy ──► keep if better
```

- **Predictor** (`predictor.py`) — the agent-editable part (like `train.py` in
  karpathy/autoresearch). In live mode it prompts **Claude (`claude-opus-4-8`,
  adaptive thinking, prompt caching, structured JSON output)**. With no API key
  it falls back to a transparent linear heuristic so the loop still runs.
- **Data** (`data_sources.py`) — fetches metadata + ground-truth citation counts
  from the Semantic Scholar Graph API (with arXiv fallback), caching responses.
  Ships a bundled snapshot dataset (`data/seed_papers.json`) so everything works
  fully offline.
- **Loop** (`autoresearch.py`) — predict → evaluate → analyze errors → propose →
  keep/discard, logged to `autoresearch_log.jsonl`.
- **No leakage**: the ground-truth `citation_count` is never shown to the
  predictor — only publication-time signals.

## Quickstart

```bash
pip install -r requirements.txt           # optional; loop runs offline without it

# Run one iteration of the autoresearch loop
python -m citation_autoresearch loop --iters 1

# Predict impact for a single paper
python -m citation_autoresearch predict arXiv:1706.03762

# Force fully-offline mode (bundled snapshot data, heuristic predictor)
python -m citation_autoresearch loop --iters 5 --offline
```

To use Claude for the predictions and strategy revisions:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m citation_autoresearch loop --iters 3
```

## Example run (offline heuristic)

```
CITATION-PREDICTION AUTORESEARCH LOOP
------------------------------------------------------------------------
LLM            : mock heuristic (ANTHROPIC_API_KEY not set)
Dataset        : bundled seed dataset (offline)
Papers         : 10 train / 6 eval
Impact label   : 'highly cited' = >= 95000 citations (p75)
------------------------------------------------------------------------

Iteration 1: KEPT  ✅
  before : n=6  MAE(log)=2.643  Spearman=+0.829  highly-cited AUC=1.000  ...
  after  : n=6  MAE(log)=1.080  Spearman=+0.829  highly-cited AUC=1.000  ...
  change : strategy v1 adopted
```

Over more iterations the loop correctly **discards** proposals that lower MAE at
the cost of rank correlation — the keep/discard guard at work.

## Metrics

The loop maximizes `Spearman + highly-cited-AUC − 0.1·MAE(log)`:

- **Spearman** — rank correlation of predicted vs. actual citations
- **highly-cited AUC** — how well predictions rank the top-quartile (impact) class
- **MAE(log)** — calibration of predicted `ln(1 + citations)`

## Layout

| File | Role |
|------|------|
| `config.py` | Fixed knobs (model, paths, loop config) — not agent-edited |
| `paper.py` | The `Paper` record + publication-time view (no label) |
| `features.py` | Publication-time feature extraction |
| `data_sources.py` | Semantic Scholar / arXiv fetch + offline fallback |
| `llm.py` | Claude wrapper (caching, adaptive thinking, JSON) + mock fallback |
| `predictor.py` | The editable `Strategy` and per-paper prediction |
| `evaluate.py` | Ranking/calibration metrics (pure Python) |
| `autoresearch.py` | The iterate → evaluate → keep/discard loop |
| `cli.py` | `loop` and `predict` commands |
| `program.md` | Instructions for the autoresearch agent |

## Notes

- The bundled citation counts are approximate early-2026 snapshots of well-known
  ML papers, included for a reproducible offline demo. With network access and
  `--offline` omitted, the loop refreshes metadata and real citation counts from
  Semantic Scholar.
- This is a research scaffold, not a calibrated impact oracle — citation count is
  an imperfect proxy for impact, and the seed set is small and ML-skewed.
