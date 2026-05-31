"""Claude wrapper for the loop.

Uses the Anthropic SDK with:
  - claude-opus-4-8 + adaptive thinking + effort
  - prompt caching on the (stable) system prompt
  - structured outputs (output_config.format) so we always get parseable JSON

If no API key is configured, the SDK is missing, or the API is unreachable, the
client transparently falls back to ``mode = "mock"`` and callers use their own
deterministic heuristic. This keeps the autoresearch loop runnable offline.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from . import config


class LLMClient:
    def __init__(self) -> None:
        self.mode = "mock"
        self._client = None
        self._reason = ""
        self._init_live()

    def _init_live(self) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            self._reason = "ANTHROPIC_API_KEY not set"
            return
        try:
            import anthropic  # noqa: F401
        except ImportError:
            self._reason = "anthropic SDK not installed (pip install anthropic)"
            return
        try:
            self._client = __import__("anthropic").Anthropic()
            self.mode = "live"
        except Exception as exc:  # pragma: no cover - depends on env
            self._reason = f"could not init Anthropic client: {exc}"

    @property
    def status(self) -> str:
        if self.mode == "live":
            return f"live (model={config.MODEL}, effort={config.EFFORT})"
        return f"mock heuristic ({self._reason})"

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict,
        cache_key: Optional[str] = None,
    ) -> dict:
        """Call Claude and return a dict validated against ``schema``.

        ``cache_key`` is unused server-side but documents which stable prefix is
        being cached; the system block always carries cache_control.
        """
        if self.mode != "live":
            raise RuntimeError("complete_json called in mock mode")

        import anthropic

        try:
            resp = self._client.messages.create(
                model=config.MODEL,
                max_tokens=4000,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": config.EFFORT,
                    "format": {"type": "json_schema", "schema": schema},
                },
                # Stable instructions first, marked for prompt caching.
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user}],
            )
        except (anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
            # Network/availability problem: degrade to mock for the rest of run.
            self.mode = "mock"
            self._reason = f"API error mid-run: {exc}"
            raise RuntimeError(self._reason) from exc

        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        return json.loads(text)
