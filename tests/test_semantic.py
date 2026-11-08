import numpy as np

from cache_lab.semantic import ParaphraseCache, SemanticCache, parse_paraphrases, verdict


def test_parse_paraphrases_handles_lists_and_drops_junk():
    text = 'Here are some:\n1. "What does prompt caching mean?"\n2) Define prompt caching.\n- What does prompt caching mean?\n3. Q\n4. What is prompt caching?'
    assert parse_paraphrases(text, 5, "What is prompt caching?") == [
        "What does prompt caching mean?", "Define prompt caching."]


def test_paraphrase_cache_is_still_exact_match():
    cache = ParaphraseCache()
    cache.add(["What is X?", "Define X."], "X is ...")
    assert cache.lookup("Define X.").hit
    assert cache.lookup("Define X.").matched_prompt == "Define X."
    assert not cache.lookup("define X.").hit


def fake_embed(texts):
    # deterministic toy embedding: bag of lowercase words over a tiny vocabulary
    vocab = ["what", "is", "x", "y", "not", "cost"]
    rows = []
    for t in texts:
        words = t.lower().replace("?", "").split()
        v = np.array([words.count(w) for w in vocab], dtype=float) + 1e-9
        rows.append(v / np.linalg.norm(v))
    return np.array(rows)


def test_semantic_threshold_decides_hit():
    cache = SemanticCache(fake_embed, threshold=0.9)
    cache.add("What is X?", "ans")
    assert cache.lookup("what is x").hit
    near = cache.lookup("What is not X?")
    assert near.similarity < 1.0
    cache.threshold = near.similarity
    assert cache.lookup("What is not X?").hit  # lowering the threshold creates a false hit
    cache.threshold = 0.99
    assert not cache.lookup("What is Y?").hit


def test_empty_semantic_cache_misses():
    assert not SemanticCache(fake_embed).lookup("anything").hit


def test_verdict():
    assert verdict(True, False) == "❌ false hit"
    assert verdict(False, True).startswith("⚠️")
    assert verdict(True, True).startswith("✅")
    assert verdict(False, False).startswith("✅")
