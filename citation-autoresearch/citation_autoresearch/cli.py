"""Command line entry point.

  python -m citation_autoresearch loop --iters 1     # run the autoresearch loop
  python -m citation_autoresearch predict arXiv:1706.03762
"""
from __future__ import annotations

import argparse
import math
import sys

from . import autoresearch as ar
from . import config
from .data_sources import build_dataset, fetch_paper, load_seed_papers
from .llm import LLMClient
from .predictor import Strategy, predict_paper


def _hr() -> None:
    print("-" * 72)


def cmd_loop(args: argparse.Namespace) -> int:
    llm = LLMClient()
    papers, source = build_dataset(enrich=not args.offline)
    ds = ar.make_dataset(papers)

    print("CITATION-PREDICTION AUTORESEARCH LOOP")
    _hr()
    print(f"LLM            : {llm.status}")
    print(f"Dataset        : {source}")
    print(f"Papers         : {len(ds.train)} train / {len(ds.evalset)} eval")
    print(f"Impact label   : 'highly cited' = >= {ds.highly_threshold} citations "
          f"(p{config.HIGHLY_CITED_PERCENTILE})")
    _hr()

    strategy = Strategy()
    best = strategy
    best_score = ar.evaluate_strategy(strategy, ds.evalset, ds, llm)[0].score()

    for i in range(1, args.iters + 1):
        res = ar.run_iteration(i, best, ds, llm)
        verdict = "KEPT  ✅" if res.kept else "DISCARDED ❌"
        print(f"\nIteration {i}: {verdict}")
        print(f"  before : {res.before.summary()}  (score={res.before.score():+.3f})")
        print(f"  after  : {res.after.summary()}  (score={res.after.score():+.3f})")
        if res.kept:
            print(f"  change : strategy v{res.proposed.version} adopted")
        best = res.strategy
        best_score = max(best_score, res.after.score() if res.kept else res.before.score())

    _hr()
    print("Final held-out predictions with the best strategy:")
    _, preds = ar.evaluate_strategy(best, ds.evalset, ds, llm)
    for p, pr in sorted(zip(ds.evalset, preds), key=lambda t: t[1].log_citations, reverse=True):
        pred_cit = int(math.expm1(pr.log_citations))
        flag = "HIGH" if pr.highly_cited_probability >= 0.5 else "    "
        print(f"  [{flag} p={pr.highly_cited_probability:.2f}] "
              f"pred≈{pred_cit:>7,d}  actual={p.citation_count:>7,d}  {p.title[:48]}")
    _hr()
    print(f"Experiment log appended to {config.RUN_LOG}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    llm = LLMClient()
    paper = None
    if not args.offline:
        paper = fetch_paper(args.identifier)
    if paper is None:
        paper = next(
            (p for p in load_seed_papers()
             if args.identifier.lower() in (p.paper_id.lower(), p.title.lower())),
            None,
        )
    if paper is None:
        print(f"Could not fetch or find '{args.identifier}'.", file=sys.stderr)
        return 1

    pred = predict_paper(paper, Strategy(), llm)
    print(f"LLM        : {llm.status}")
    print(f"Paper      : {paper.title} ({paper.year})")
    _hr()
    print(f"Predicted citations    : ≈ {int(math.expm1(pred.log_citations)):,d} "
          f"(log={pred.log_citations:.2f})")
    print(f"Highly-cited probability: {pred.highly_cited_probability:.2f}")
    print(f"Reasoning  : {pred.reasoning}")
    if paper.citation_count is not None:
        print(f"\nActual citations (label): {paper.citation_count:,d}")
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--offline", action="store_true",
                        help="skip network; use bundled snapshot data")

    parser = argparse.ArgumentParser(prog="citation_autoresearch", parents=[common])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_loop = sub.add_parser("loop", parents=[common], help="run the autoresearch loop")
    p_loop.add_argument("--iters", type=int, default=1)
    p_loop.set_defaults(func=cmd_loop)

    p_pred = sub.add_parser("predict", parents=[common],
                            help="predict citations for one paper")
    p_pred.add_argument("identifier", help="arXiv:<id>, DOI, S2 id, or seed title")
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
