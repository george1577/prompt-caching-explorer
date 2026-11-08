# Local model for tabs ① to ③, Anthropic API (or a simulation) for tab ④

Prefix caching happens inside a provider's servers, so an API call shows only its effects (token counts and a bill), never the mechanism. Tab ③ therefore runs prefix caching on a local Qwen model, where the app can save the KV cache, time the prefill with and without it, and show which tokens were reused. Tab ④ uses the real Anthropic API, because provider pricing, minimum lengths and TTLs can only be seen there. Response caches (tabs ① and ②) also use the local model, so they cost nothing to try. With no API key, tab ④ falls back to a simulator that applies the provider's documented rules and prices to estimated token counts. It shows no latency and is clearly labelled, so it is never mistaken for a real measurement.

## Considered Options

- **API only**: real numbers everywhere, but every click costs money and prefix caching stays a black box.
- **Local only**: free and fully visible, but it can't show what providers actually charge, which is the main reason people use prompt caching.
