"""Two ways to make a response cache tolerate rewordings.

* Paraphrase pre-population (the book's semantic_cache.py): ask an LLM for rewordings of a prompt and
  store the same answer under every rewording's exact key. Lookups are still exact-match.
* Semantic cache: store the prompt's embedding; a new prompt hits when its embedding is similar enough.
"""

import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from cache_lab.response_cache import cache_key

Embedder = Callable[[list[str]], np.ndarray]  # texts -> one unit-length vector per row


def parse_paraphrases(text: str, n: int, original: str) -> list[str]:
    """Pull up to n distinct rewordings out of a model's numbered/bulleted list."""
    found = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(\d+[.):]|[-*•])\s*", "", line).strip().strip('"').strip()
        if len(line) < 5 or line.endswith(":"):
            continue
        if line == original or line in found:
            continue
        found.append(line)
    return found[:n]


@dataclass
class Match:
    hit: bool
    matched_prompt: Optional[str]
    answer: Optional[str]
    similarity: Optional[float] = None


class ParaphraseCache:
    """Exact-match lookups over an original prompt plus its pre-generated rewordings."""

    def __init__(self):
        self.store: dict[str, tuple[str, str]] = {}  # key -> (stored prompt, answer)

    def add(self, prompts: list[str], answer: str) -> None:
        for p in prompts:
            self.store[cache_key(p)] = (p, answer)

    def lookup(self, prompt: str) -> Match:
        found = self.store.get(cache_key(prompt))
        return Match(found is not None, found and found[0], found and found[1])


class SemanticCache:
    """Nearest-neighbour lookup by cosine similarity of prompt embeddings."""

    def __init__(self, embed: Embedder, threshold: float = 0.85):
        self.embed = embed
        self.threshold = threshold
        self.prompts: list[str] = []
        self.answers: list[str] = []
        self.vectors: Optional[np.ndarray] = None

    def add(self, prompt: str, answer: str) -> None:
        vec = self.embed([prompt])
        self.vectors = vec if self.vectors is None else np.vstack([self.vectors, vec])
        self.prompts.append(prompt)
        self.answers.append(answer)

    def similarity(self, prompt: str) -> tuple[int, float]:
        """Index and cosine similarity of the closest stored prompt."""
        sims = self.vectors @ self.embed([prompt])[0]
        best = int(np.argmax(sims))
        return best, float(sims[best])

    def lookup(self, prompt: str) -> Match:
        if self.vectors is None:
            return Match(False, None, None)
        best, sim = self.similarity(prompt)
        if sim >= self.threshold:
            return Match(True, self.prompts[best], self.answers[best], sim)
        return Match(False, self.prompts[best], None, sim)


def verdict(hit: bool, same_meaning: bool) -> str:
    """How a lookup turned out compared with what a person would want."""
    if hit and same_meaning:
        return "✅ correct hit"
    if hit and not same_meaning:
        return "❌ false hit"
    if not hit and same_meaning:
        return "⚠️ missed (could have reused)"
    return "✅ correct miss"
