"""Groq chat-completions client -- PRD Section 9.9 (provider switched to Groq).

Called over plain REST with `requests` rather than the `groq` SDK: one fewer
dependency, and the request shape is OpenAI-compatible and stable.

**This client is built for a free-tier key**, so three things are non-negotiable:

1. **Disk cache.** Every response is cached on `sha256(model + prompt + params)`.
   Re-running the evaluation, regenerating a report, or restarting the UI costs
   zero API calls. Because generation is pinned to `temperature = 0.0`, caching
   is not merely an optimisation -- it is *semantically correct*, since the same
   prompt is defined to produce the same answer.
2. **Rate limiting.** A minimum interval is enforced between live calls, and a
   429 triggers exponential backoff that honours the `retry-after` header rather
   than hammering a limit that has already been hit.
3. **A hard call budget.** `max_calls` caps live requests per process. Running
   out raises a clear error instead of silently burning the daily quota -- an
   evaluation loop over 51 questions is exactly how a free tier disappears.

The key is read from the environment or a gitignored `.env`, never from source.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from ..utils.logging import get_logger

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqError(RuntimeError):
    """Raised for configuration or unrecoverable API failures."""


def load_env(root: Path) -> None:
    """Minimal .env loader -- avoids a python-dotenv dependency."""
    path = root / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Usage:
    calls_live: int = 0
    calls_cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class GroqClient:
    def __init__(
        self,
        root: Path,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_tokens: int = 512,
        cache_path: Path | None = None,
        min_interval_s: float = 2.0,
        max_calls: int = 120,
        timeout_s: int = 60,
    ) -> None:
        load_env(root)
        self.api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_interval_s = min_interval_s
        self.max_calls = max_calls
        self.timeout_s = timeout_s
        self.usage = Usage()
        self._last_call_at = 0.0

        self.cache_path = cache_path or (root / ".cache" / "groq_responses.json")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                get_logger().warning("groq cache corrupt -- starting fresh")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _key(self, system: str, user: str, model: str) -> str:
        blob = json.dumps(
            {"m": model, "t": self.temperature, "s": system, "u": user},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _save_cache(self) -> None:
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache, indent=0), encoding="utf-8")
        tmp.replace(self.cache_path)

    def complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> str:
        """Return the assistant message, from cache when possible."""
        log = get_logger()
        model = model or self.model
        key = self._key(system, user, model)

        if use_cache and key in self._cache:
            self.usage.calls_cached += 1
            return self._cache[key]

        if not self.api_key:
            raise GroqError(
                "GROQ_API_KEY is not set. Put it in semantic-qa-agent/.env as "
                "GROQ_API_KEY=... (the file is gitignored)."
            )
        if self.usage.calls_live >= self.max_calls:
            raise GroqError(
                f"call budget exhausted ({self.max_calls} live calls). Raise "
                f"--max-calls deliberately if you really want to spend more quota."
            )

        # Rate limit: never issue calls closer together than min_interval_s.
        elapsed = time.perf_counter() - self._last_call_at
        if self._last_call_at and elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": 1,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        backoff = 4.0
        for attempt in range(4):
            self._last_call_at = time.perf_counter()
            try:
                response = requests.post(
                    GROQ_URL, headers=headers, json=payload, timeout=self.timeout_s
                )
            except requests.RequestException as exc:
                if attempt == 3:
                    raise GroqError(f"network error calling Groq: {exc}") from exc
                time.sleep(backoff)
                backoff *= 2
                continue

            if response.status_code == 429:
                # Honour the server's own retry hint rather than guessing.
                wait = float(response.headers.get("retry-after", backoff))
                log.warning("groq rate limited -- waiting %.1fs (attempt %d/4)",
                            wait, attempt + 1)
                time.sleep(min(wait, 60.0))
                backoff *= 2
                continue

            if response.status_code >= 500:
                if attempt == 3:
                    raise GroqError(f"groq server error {response.status_code}")
                time.sleep(backoff)
                backoff *= 2
                continue

            if response.status_code != 200:
                raise GroqError(
                    f"groq returned {response.status_code}: {response.text[:300]}"
                )

            body = response.json()
            text = body["choices"][0]["message"]["content"].strip()
            usage = body.get("usage", {})
            self.usage.calls_live += 1
            self.usage.prompt_tokens += int(usage.get("prompt_tokens", 0))
            self.usage.completion_tokens += int(usage.get("completion_tokens", 0))

            self._cache[key] = text
            self._save_cache()
            return text

        raise GroqError("groq rate limit not cleared after 4 attempts")

    def usage_summary(self) -> dict:
        return {
            "live_calls": self.usage.calls_live,
            "cached_calls": self.usage.calls_cached,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
            "cache_entries": len(self._cache),
        }
