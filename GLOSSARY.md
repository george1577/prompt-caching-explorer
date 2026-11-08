# Prompt Caching Explorer

A learning tool for seeing how prompt caching saves time and money: by reusing whole answers, or by reusing the model's work on a repeated prompt prefix.

## Language

### The pattern

**Prompt caching**:
The overall pattern of avoiding repeated LLM work for repeated (or similar) prompts. An umbrella for both response caching and prefix caching.
_Avoid_: Caching (too vague on its own)

**Response cache**:
A store, owned by the application, that maps a prompt to a previously generated answer and returns that answer without calling the model.
_Avoid_: Client-side cache, output cache, prompt cache

**Prefix cache**:
Saved internal model state (the KV cache) for the opening part of a prompt, reused when a later prompt starts with exactly the same tokens. The model still generates a fresh answer.
_Avoid_: Server-side cache, prompt cache, context cache

**Provider prompt caching**:
A prefix cache run by an LLM API provider (e.g. Anthropic, OpenAI) and billed through its pricing.
_Avoid_: Server-side caching

### Response caching

**Cache key**:
The value derived from a prompt that a response cache looks answers up by.
_Avoid_: Hash (a hash is one way to make a key)

**Exact-match cache**:
A response cache that hits only when the new prompt is character-for-character identical to a stored one.
_Avoid_: Basic cache

**Paraphrase pre-population**:
Filling a response cache by asking an LLM for rewordings of a prompt and storing the same answer under each rewording's exact key. Lookups are still exact-match.
_Avoid_: Semantic caching (it isn't lookup by meaning)

**Semantic cache**:
A response cache that hits when a new prompt's embedding is close enough to a stored prompt's embedding, judged by a similarity threshold.
_Avoid_: Fuzzy cache

**False hit**:
A response cache hit that returns an answer meant for a different question.
_Avoid_: Wrong hit, collision

### Prefix caching

**KV cache**:
The per-token key and value tensors the model computes while reading a prompt; the thing a prefix cache saves.
_Avoid_: Attention cache, hidden states

**Prefill**:
The model's pass over the input prompt before it produces the first output token; the work a prefix cache skips.
_Avoid_: Encoding, prompt processing

**Shared prefix**:
The longest run of identical leading tokens between a cached prompt and a new prompt. Only this part can be reused.
_Avoid_: Common prompt, overlap

**Cache breakpoint**:
A marker in a provider request saying "cache everything up to here".
_Avoid_: Cache point, checkpoint

**Cache write**:
A request that creates a prefix cache entry; providers bill these input tokens at a premium.
_Avoid_: Cache creation, cache miss

**Cache read**:
A request that reuses an existing prefix cache entry; providers bill these input tokens at a steep discount.
_Avoid_: Cache hit (reserved for response caches)

**TTL**:
How long a cache entry lives without being used before it's thrown away.
_Avoid_: Expiry, lifetime
