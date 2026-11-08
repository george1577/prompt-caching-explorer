"""An exact-match response cache on disk: the idea in the book's basic_cache.py.

The prompt is hashed with MD5 and the answer stored as <hash>.json. A later prompt hits only if its
hash is the same, which means the text must be identical down to every space and capital letter.
"""

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def cache_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


@dataclass
class Entry:
    key: str
    prompt: str
    content: str
    model: str
    created_at: float
    generation_seconds: float


class ExactMatchCache:
    def __init__(self, cache_dir: Path | str):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, prompt: str) -> Path:
        return self.dir / f"{cache_key(prompt)}.json"

    def get(self, prompt: str) -> Optional[Entry]:
        path = self.path_for(prompt)
        if not path.exists():
            return None
        return Entry(**json.loads(path.read_text()))

    def put(self, prompt: str, content: str, model: str, generation_seconds: float) -> Entry:
        entry = Entry(cache_key(prompt), prompt, content, model, time.time(), generation_seconds)
        self.path_for(prompt).write_text(json.dumps(entry.__dict__, indent=2))
        return entry

    def entries(self) -> list[Entry]:
        found = [Entry(**json.loads(p.read_text())) for p in self.dir.glob("*.json")]
        return sorted(found, key=lambda e: e.created_at)

    def clear(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Lookup:
    prompt: str
    key: str
    hit: bool
    content: str
    seconds: float  # time to answer: a file read on a hit, a model call on a miss


def get_completion(cache: ExactMatchCache, prompt: str, generate, model: str, use_cache: bool = True) -> Lookup:
    """The book's get_completion(): look in the cache first; on a miss call the model and store the answer."""
    t0 = time.perf_counter()
    if use_cache and (entry := cache.get(prompt)) is not None:
        return Lookup(prompt, entry.key, True, entry.content, time.perf_counter() - t0)
    content = generate(prompt)
    seconds = time.perf_counter() - t0
    cache.put(prompt, content, model, seconds)
    return Lookup(prompt, cache_key(prompt), False, content, seconds)
