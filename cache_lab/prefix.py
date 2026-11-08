"""Prefix (KV) caching on the local model: reuse the model's work for the leading tokens two prompts share."""

import statistics
from dataclasses import dataclass


def shared_prefix_len(a: list[int], b: list[int]) -> int:
    """How many leading tokens a and b have in common."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def change_one_char(text: str, fraction: float) -> tuple[str, int]:
    """Swap one letter about `fraction` of the way through the text. Returns (new text, char position)."""
    pos = int(fraction * (len(text) - 1))
    while pos < len(text) and not text[pos].isalpha():
        pos += 1
    if pos >= len(text):
        return text, -1
    c = text[pos]
    swapped = "X" if c.lower() != "x" else "Y"
    return text[:pos] + swapped + text[pos + 1:], pos


@dataclass
class PrefixRun:
    cached_ids: list[int]  # the prompt whose KV cache was saved earlier
    new_ids: list[int]  # the prompt we want to answer now
    shared: int  # leading tokens reused from the cache
    cold_seconds: float  # prefill of new_ids from scratch (median)
    warm_seconds: float  # prefill of only the non-shared tokens, reusing the cache (median)
    build_seconds: float  # the one-off cost of creating the cache in the first place
    cold_output: list[int]
    warm_output: list[int]
    max_logit_diff: float

    @property
    def recomputed(self) -> int:
        return len(self.new_ids) - self.reused

    @property
    def reused(self) -> int:
        return min(self.shared, len(self.new_ids) - 1)

    @property
    def outputs_match(self) -> bool:
        return self.cold_output == self.warm_output


def run_prefix_demo(lm, cached_messages: list[dict], new_messages: list[dict], max_new_tokens: int = 40,
                    repeats: int = 3) -> PrefixRun:
    cached_ids = lm.chat_ids(cached_messages)
    new_ids = lm.chat_ids(new_messages)
    shared = shared_prefix_len(cached_ids, new_ids)

    lm.prefill(new_ids[:8])  # warm up the model so the first timing isn't inflated
    saved = lm.prefill(cached_ids)  # "earlier request": this is the cache write
    colds, warms = [], []
    for _ in range(repeats):
        cold = lm.prefill(new_ids)
        warm = lm.prefill(new_ids, reuse=saved, reuse_tokens=shared)
        colds.append(cold.seconds)
        warms.append(warm.seconds)
    return PrefixRun(
        cached_ids, new_ids, shared,
        statistics.median(colds), statistics.median(warms), saved.seconds,
        lm.continue_greedy(cold, max_new_tokens), lm.continue_greedy(warm, max_new_tokens),
        float((cold.last_logits - warm.last_logits).abs().max()),
    )
