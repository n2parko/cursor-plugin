"""The Paper record — the unit the loop predicts on."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Paper:
    """A paper plus the signals available *at publication time*.

    ``citation_count`` is the ground-truth label used only for evaluation — it
    is never exposed to the predictor (that would leak the answer).
    """

    paper_id: str
    title: str
    abstract: str = ""
    year: Optional[int] = None
    venue: str = ""
    fields: list[str] = field(default_factory=list)
    author_names: list[str] = field(default_factory=list)
    reference_count: Optional[int] = None

    # Author track record known at/just before publication (optional; filled
    # from Semantic Scholar when available). Used as an "early signal".
    author_prior_citations: Optional[int] = None
    author_max_h_index: Optional[int] = None

    # Ground truth (label). May be None at pure prediction time.
    citation_count: Optional[int] = None

    @property
    def n_authors(self) -> int:
        return len(self.author_names)

    @property
    def log_citations(self) -> Optional[float]:
        if self.citation_count is None:
            return None
        return math.log1p(self.citation_count)

    def publication_time_view(self) -> dict:
        """Everything the predictor is allowed to see (no label)."""
        d = asdict(self)
        d.pop("citation_count", None)
        d["n_authors"] = self.n_authors
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        known = {f for f in cls.__dataclass_fields__}  # noqa: F841
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
