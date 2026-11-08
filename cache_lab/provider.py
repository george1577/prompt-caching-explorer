"""Provider prompt caching with the Anthropic Messages API, plus a simulator that follows the same rules.

The book's anthropic_cache.py sends no `cache_control`, so nothing is ever cached. Here a request either
copies that (mode "none") or marks the system prompt with a cache breakpoint (mode "breakpoint").
"""

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values

from cache_lab.pricing import MIN_CACHEABLE_TOKENS, Usage

PROJECT = Path(__file__).parent.parent
KEY_FILES = [PROJECT.parent / "generative-ai-design-patterns" / "examples" / "keys.env", PROJECT / ".env"]
TTL_SECONDS = {"5m": 300, "1h": 3600}


def looks_like_key(value: Optional[str]) -> bool:
    """Real Anthropic keys start with sk-ant-; anything else (empty, a placeholder comment) is ignored."""
    return bool(value) and value.strip().startswith("sk-ant-")


def find_api_key() -> Optional[str]:
    if looks_like_key(os.environ.get("ANTHROPIC_API_KEY")):
        return os.environ["ANTHROPIC_API_KEY"].strip()
    for f in KEY_FILES:
        if f.exists() and looks_like_key(key := dotenv_values(f).get("ANTHROPIC_API_KEY")):
            return key.strip()
    return None


def build_request(model: str, system: str, question: str, mode: str, ttl: str = "5m",
                  max_tokens: int = 200) -> dict:
    """Keyword arguments for client.messages.create()."""
    block = {"type": "text", "text": system}
    if mode == "breakpoint":
        block["cache_control"] = {"type": "ephemeral"} if ttl == "5m" else {"type": "ephemeral", "ttl": "1h"}
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "system": [block],
        "messages": [{"role": "user", "content": question}],
    }


@dataclass
class CallResult:
    question: str
    usage: Usage
    seconds: float
    text: str
    simulated: bool
    at: float  # wall-clock time the call finished


def usage_from_response(u) -> Usage:
    creation = getattr(u, "cache_creation", None)
    w5 = getattr(creation, "ephemeral_5m_input_tokens", None) if creation else None
    w1 = getattr(creation, "ephemeral_1h_input_tokens", None) if creation else None
    total_write = u.cache_creation_input_tokens or 0
    if w5 is None and w1 is None:  # older responses: no breakdown, assume 5 minutes
        w5, w1 = total_write, 0
    return Usage(u.input_tokens, w5 or 0, w1 or 0, u.cache_read_input_tokens or 0, u.output_tokens)


def call_anthropic(client, request: dict) -> CallResult:
    t0 = time.perf_counter()
    resp = client.messages.create(**request)
    seconds = time.perf_counter() - t0
    return CallResult(request["messages"][0]["content"], usage_from_response(resp.usage), seconds,
                      resp.content[0].text, False, time.time())


def count_prefix_tokens(client, model: str, system: str) -> int:
    """Claude tokens in the system prompt (the part a breakpoint on it would cache)."""
    n_with = client.messages.count_tokens(model=model, system=system,
                                          messages=[{"role": "user", "content": "x"}]).input_tokens
    n_without = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": "x"}]).input_tokens
    return n_with - n_without


# ---- simulation ---------------------------------------------------------------------------------------

def rough_tokens(text: str) -> int:
    """About 4 characters per token for English text: good enough for a simulation."""
    return max(1, round(len(text) / 4))


class Simulator:
    """Plays the provider's caching rules without calling the API. Clearly labelled as simulated in the UI."""

    def __init__(self):
        self.entries: dict[str, float] = {}  # cache entry id -> expiry time

    def call(self, request: dict, now: Optional[float] = None) -> CallResult:
        now = time.time() if now is None else now
        block = request["system"][0]
        question = request["messages"][0]["content"]
        prefix = rough_tokens(block["text"])
        after = rough_tokens(question) + 8  # the question plus chat formatting
        output = min(request["max_tokens"], 150)
        cc = block.get("cache_control")
        ttl = cc.get("ttl", "5m") if cc else "5m"
        usage = Usage(prefix + after, 0, 0, 0, output)
        if cc and prefix >= MIN_CACHEABLE_TOKENS[request["model"]]:
            entry = hashlib.md5(f"{request['model']}|{block['text']}".encode()).hexdigest()
            if self.entries.get(entry, 0) > now:
                usage = Usage(after, 0, 0, prefix, output)
            else:
                usage = Usage(after, prefix if ttl == "5m" else 0, prefix if ttl == "1h" else 0, 0, output)
            self.entries[entry] = now + TTL_SECONDS[ttl]  # every write or read restarts the clock
        # No latency is simulated: real speed-ups depend on the provider's load and can't be faked honestly.
        return CallResult(question, usage, 0.0, "(simulated: no API call was made)", True, now)
