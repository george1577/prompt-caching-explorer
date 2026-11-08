from cache_lab.prompts import near_miss_variants
from cache_lab.response_cache import ExactMatchCache, cache_key, get_completion


def test_key_is_md5_and_sensitive_to_every_character():
    assert cache_key("hello") == "5d41402abc4b2a76b9719d911017c592"
    assert cache_key("hello") != cache_key("hello ")
    assert cache_key("hello") != cache_key("Hello")


def test_miss_then_hit(tmp_path):
    cache = ExactMatchCache(tmp_path)
    calls = []

    def generate(p):
        calls.append(p)
        return f"answer to {p}"

    first = get_completion(cache, "Q?", generate, "m")
    second = get_completion(cache, "Q?", generate, "m")
    assert (first.hit, second.hit) == (False, True)
    assert first.content == second.content
    assert calls == ["Q?"]
    assert (tmp_path / f"{cache_key('Q?')}.json").exists()


def test_use_cache_false_always_calls_model(tmp_path):
    cache = ExactMatchCache(tmp_path)
    n = []
    gen = lambda p: n.append(1) or "a"
    get_completion(cache, "Q", gen, "m")
    assert not get_completion(cache, "Q", gen, "m", use_cache=False).hit
    assert len(n) == 2


def test_near_misses_all_miss(tmp_path):
    cache = ExactMatchCache(tmp_path)
    prompt = "Explain the concept of prompt caching in AI systems."
    cache.put(prompt, "a", "m", 1.0)
    variants = near_miss_variants(prompt)
    assert variants[0] == ("identical", prompt)
    assert cache.get(prompt) is not None
    assert all(cache.get(v) is None for _, v in variants[1:])
    assert len({v for _, v in variants}) == len(variants)


def test_clear(tmp_path):
    cache = ExactMatchCache(tmp_path)
    cache.put("a", "b", "m", 1.0)
    assert len(cache.entries()) == 1
    cache.clear()
    assert cache.entries() == []
