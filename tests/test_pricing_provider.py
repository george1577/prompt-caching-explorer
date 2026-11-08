import pytest

from cache_lab.pricing import PRICES, Usage, break_even_reads, cost_of
from cache_lab.provider import Simulator, build_request, looks_like_key

HAIKU = "claude-haiku-5-5"


def test_cost_split_and_savings():
    p = PRICES[HAIKU]
    write = cost_of(Usage(20, 1000, 0, 0, 100), p)
    read = cost_of(Usage(20, 0, 0, 1000, 100), p)
    assert write.cache_write == pytest.approx(1000 * 0.125 / 1e6)
    assert read.cache_read == pytest.approx(1000 * 0.01 / 1e6)
    assert write.saved < 0 < read.saved  # a write costs extra, a read saves
    assert write.without_caching == read.without_caching


def test_break_even():
    # 5-minute write costs 0.25x extra, each read saves 0.9x: one read pays it back
    assert break_even_reads(PRICES[HAIKU], "5m") == pytest.approx(0.025 / 0.09)


def test_build_request_modes():
    none = build_request(HAIKU, "sys", "q", "none")
    assert "cache_control" not in none["system"][0]
    bp = build_request(HAIKU, "sys", "q", "breakpoint")
    assert bp["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert build_request(HAIKU, "sys", "q", "breakpoint", ttl="1h")["system"][0]["cache_control"]["ttl"] == "1h"


def test_simulator_write_then_read_then_expire():
    sim = Simulator()
    long_sys = "word " * 1000  # ~1250 rough tokens, above the 512 minimum
    req = build_request(HAIKU, long_sys, "q", "breakpoint")
    a = sim.call(req, now=0).usage
    b = sim.call(req, now=60).usage
    c = sim.call(req, now=60 + 301).usage
    assert a.cache_write > 0 and a.cache_read == 0
    assert b.cache_read == a.cache_write and b.cache_write == 0
    assert c.cache_write > 0  # 5 minutes passed without use: expired


def test_simulator_no_breakpoint_or_too_short_never_caches():
    sim = Simulator()
    for req in [build_request(HAIKU, "word " * 1000, "q", "none"), build_request(HAIKU, "short", "q", "breakpoint")]:
        for t in (0, 1):
            u = sim.call(req, now=t).usage
            assert u.cache_write == u.cache_read == 0


def test_placeholder_keys_are_ignored():
    assert not looks_like_key("#See https://console.anthropic.com")
    assert not looks_like_key("")
    assert looks_like_key("sk-ant-abc")
