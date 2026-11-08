# Prompt Caching Explorer

An interactive Streamlit app for **seeing** how prompt caching saves time and money. It covers both
families of the pattern: reusing whole answers (response caches) and reusing the model's work on a
repeated start of the prompt (prefix caches). Most of it runs on a small open model (Qwen2.5) on your
laptop, so you can watch the cache hit, miss, or return a wrong answer instead of just reading about it.

It's a companion to the *Prompt Caching* examples in
[Generative AI Design Patterns](https://github.com/lakshmanok/generative-ai-design-patterns)
(`examples/25_prompt_caching`), and covers every script in that folder.

> **New to this?** Read [docs/how-prompt-caching-works.md](docs/how-prompt-caching-works.md) first.
> It explains every idea in plain language, with diagrams and real numbers from this app.
> Terms are defined in [GLOSSARY.md](GLOSSARY.md).

---

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), about 2 GB of free RAM, and an internet connection for
the first run (to download the models, about 1.1 GB in total).

```bash
uv sync                          # installs Python 3.12 + dependencies into .venv
uv run streamlit run app.py      # opens http://localhost:8501
```

The first click in tabs ① to ③ loads the local model (around 10 to 20 seconds). Run the tests:

```bash
uv run pytest                    # includes one slow test that loads the real model
uv run pytest -m "not slow"      # logic only, a few seconds
```

### Configuration

| Setting | How | Default |
|---|---|---|
| Local model | Sidebar dropdown | `Qwen/Qwen2.5-0.5B-Instruct` (also `1.5B`: better answers, about 3x slower) |
| Device | `PCL_DEVICE=mps uv run streamlit run app.py` (or `cuda`) | `cpu` (or `cuda` if available) |
| Anthropic key (tab ④) | Sidebar field, or `ANTHROPIC_API_KEY` in the environment, `../generative-ai-design-patterns/examples/keys.env`, or `.env` | none, so tab ④ runs a labelled **simulation** |
| Prices | Editable table in the sidebar | Anthropic list prices, checked 2026-11-08 |

Keys that don't start with `sk-ant-` (empty values, placeholder comments) are ignored.

> **Why not Apple's MPS by default?** Recent PyTorch builds crash on MPS on older macOS versions (seen on
> macOS 14). The models are small enough for the CPU. On macOS 15+ you can try `PCL_DEVICE=mps`.

---

## How the app works

The four tabs are meant to be followed in order. Each one fixes a weakness of the one before.

### ① Exact-match response cache

**Shows:** the book's `basic_cache.py`. A prompt is hashed with MD5. The answer is saved as
`.prompt_cache/<hash>.json`, and a repeat of the *identical* prompt is answered from disk.

**You provide:** a prompt, a maximum answer length, and whether to use the cache.

**You see:** hit or miss, time to answer (a model call takes seconds, a hit takes about 0.1 ms), the cache key, a
history of every request, a **near-miss table** (a trailing space, lowercase, …), each with a completely
different key, and the cache files themselves.

### ② Paraphrase pre-population vs semantic cache

**Shows:** two ways of tolerating rewordings:
- the book's `semantic_cache.py` (**paraphrase pre-population**): the local model writes N rewordings, and the same
  answer is stored under each one's exact key;
- a real **semantic cache**: a MiniLM embedding of the question, with a hit when cosine similarity ≥ a threshold.

**You provide:** a seed question, N, a similarity threshold, and an editable list of test questions. You mark
which ones mean the same thing as the seed.

**You see:** for each test question, hit or miss for both approaches, the similarity score, and a verdict (correct hit,
**false hit**, missed reuse, correct miss), plus a chart with the threshold line.

### ③ Prefix (KV) cache

**Shows:** what providers do inside their servers, run on the local model. The KV cache of an earlier request
(the book's long system prompt plus a question) is saved. The prefill of a new request is then timed from
scratch and again reusing the cached prefix.

**You provide:** the two questions and what else differs: nothing, a timestamp at the top of the system prompt,
or one changed character at a position you choose.

**You see:** tokens reused vs recomputed, prefill times and speed-up, a colour-coded token strip marking the
first differing token, and proof that both answers are identical.

### ④ Provider prompt caching (Anthropic)

**Shows:** Anthropic's `cache_control` cache breakpoints, with **cache write** and **cache read** token counts and
what each costs.

**You provide:** model, request style (the book's original with no `cache_control`, or a breakpoint on the
system prompt), TTL (5m or 1h), system prompt size (including one below the 512-token minimum), and the questions.

**You see:** per-request token breakdown, latency (real calls only), cost vs cost without caching, savings,
a stacked token chart, and when the cache entry expires.

---

## Project layout

```
app.py                     Streamlit UI (the four tabs)
cache_lab/
  lm.py                    local Qwen model: chat template, prefill (with KV cache reuse), greedy decoding
  embed.py                 local MiniLM sentence embeddings
  prompts.py               the book's prompts, example questions, near-miss variants
  response_cache.py        ExactMatchCache + get_completion (tab ①)
  semantic.py              ParaphraseCache, SemanticCache, verdicts (tab ②)
  prefix.py                shared prefix, run_prefix_demo (tab ③)
  pricing.py               prices, minimum lengths, cost of a request (tab ④)
  provider.py              Anthropic requests, usage parsing, simulator, API key lookup (tab ④)
data/book_system_prompt.txt  the book's long system prompt, verbatim
tests/                     pytest suite
docs/how-prompt-caching-works.md   the beginner's guide
docs/adr/                  architecture decisions
```
