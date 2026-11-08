"""Anthropic prices and the cost of one request, split by kind of token.

Prices are US dollars per million tokens, copied from https://platform.claude.com/docs/en/about-claude/pricing
on 2026-11-08. Haiku 5.5 prices are for prompts up to 100,000 tokens (longer prompts cost more).
"""

from dataclasses import dataclass

PRICES_CHECKED = "2026-11-08"


@dataclass
class Price:
    base_input: float
    write_5m: float
    write_1h: float
    cache_read: float
    output: float


PRICES = {
    "claude-haiku-5-5": Price(0.10, 0.125, 0.20, 0.01, 0.50),
    "claude-sonnet-5-5": Price(2.00, 2.50, 4.00, 0.10, 10.00),
    "claude-opus-5-5": Price(4.00, 5.00, 8.00, 0.20, 20.00),
}

# Shorter prompts are silently not cached (no error, both cache counters stay 0).
MIN_CACHEABLE_TOKENS = {"claude-haiku-5-5": 512, "claude-sonnet-5-5": 512, "claude-opus-5-5": 512}


@dataclass
class Usage:
    """The token counts in an Anthropic response's `usage`."""

    input_tokens: int  # tokens after the last cache breakpoint (always billed at the base price)
    cache_write_5m: int
    cache_write_1h: int
    cache_read: int
    output_tokens: int

    @property
    def cache_write(self) -> int:
        return self.cache_write_5m + self.cache_write_1h

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_write + self.cache_read


@dataclass
class Cost:
    input: float
    cache_write: float
    cache_read: float
    output: float
    without_caching: float  # what the same request would cost with every input token at the base price

    @property
    def total(self) -> float:
        return self.input + self.cache_write + self.cache_read + self.output

    @property
    def saved(self) -> float:
        return self.without_caching - self.total


def cost_of(usage: Usage, price: Price) -> Cost:
    m = 1_000_000
    return Cost(
        input=usage.input_tokens * price.base_input / m,
        cache_write=(usage.cache_write_5m * price.write_5m + usage.cache_write_1h * price.write_1h) / m,
        cache_read=usage.cache_read * price.cache_read / m,
        output=usage.output_tokens * price.output / m,
        without_caching=(usage.total_input * price.base_input + usage.output_tokens * price.output) / m,
    )


def break_even_reads(price: Price, ttl: str = "5m") -> float:
    """How many cache reads it takes for one cache write to pay for itself."""
    write = price.write_5m if ttl == "5m" else price.write_1h
    return (write - price.base_input) / (price.base_input - price.cache_read)
