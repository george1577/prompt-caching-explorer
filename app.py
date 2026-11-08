"""Prompt Caching Explorer: a Streamlit app for seeing how prompt caching saves time and money.

Run with:  uv run streamlit run app.py
"""

import html
import json
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from cache_lab.embed import EMBED_MODEL, Embedder
from cache_lab.lm import MODELS, LanguageModel
from cache_lab.prefix import change_one_char, run_prefix_demo
from cache_lab.pricing import MIN_CACHEABLE_TOKENS, PRICES, PRICES_CHECKED, Price, break_even_reads, cost_of
from cache_lab.prompts import (
    BOOK_QUESTION,
    BOOK_SYSTEM_PROMPT,
    SHORT_SYSTEM_PROMPT,
    answer_messages,
    near_miss_variants,
    paraphrase_messages,
)
from cache_lab.provider import TTL_SECONDS, Simulator, build_request, call_anthropic, find_api_key, looks_like_key
from cache_lab.response_cache import ExactMatchCache, cache_key, get_completion
from cache_lab.semantic import ParaphraseCache, SemanticCache, parse_paraphrases, verdict

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / ".prompt_cache"
DOC_LINK = "docs/how-prompt-caching-works.md"

st.set_page_config(page_title="Prompt Caching Explorer", page_icon="🗄️", layout="wide")

# Results kept in session_state are cache_lab objects. When that code changes while the app is running,
# Streamlit reloads it but old results keep their old shape, so drop stored results whenever the code changes.
RESULT_KEYS = ("t1_last", "t1_history", "t2", "t3", "t4_calls", "t4_sim", "t4_last_request")


def code_version() -> str:
    files = sorted((ROOT / "cache_lab").glob("*.py"))
    return "|".join(f"{f.name}:{f.stat().st_mtime_ns}" for f in files)


if st.session_state.get("_code_version") != (version := code_version()):
    for key in RESULT_KEYS:
        st.session_state.pop(key, None)
    st.session_state["_code_version"] = version


@st.cache_resource(show_spinner="Loading the local model (the first run downloads it)…")
def get_model(model_id: str) -> LanguageModel:
    return LanguageModel(model_id)


@st.cache_resource(show_spinner="Loading the embedding model…")
def get_embedder() -> Embedder:
    return Embedder()


def fmt_seconds(s: float) -> str:
    return f"{s * 1000:.1f} ms" if s < 1 else f"{s:.2f} s"


def fmt_usd(x: float) -> str:
    return f"${x:.6f}" if abs(x) < 0.01 else f"${x:.4f}"


# ---- sidebar --------------------------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    model_id = st.selectbox("Local model (tabs ① to ③)", MODELS,
                            help="Runs on your computer. 1.5B writes better answers but is about 3x slower.")
    st.caption(f"Embedding model (tab ②): `{EMBED_MODEL.split('/')[-1]}`")

    st.divider()
    st.subheader("Anthropic API (tab ④)")
    typed_key = st.text_input("API key (optional)", type="password",
                              help="Used only for this browser session. Leave empty to use keys.env / .env, "
                                   "or to run the simulation.")
    api_key = typed_key.strip() if looks_like_key(typed_key) else find_api_key()
    if api_key:
        st.success("Key found: tab ④ makes **real** API calls.")
    else:
        st.info("No key found: tab ④ runs a **simulation** of the provider's caching rules.")

    st.divider()
    st.subheader("Prices ($ per million tokens)")
    st.caption(f"Checked {PRICES_CHECKED}. Edit them if they change.")
    price_df = pd.DataFrame(
        [{"model": m, **p.__dict__} for m, p in PRICES.items()]
    ).set_index("model")
    edited = st.data_editor(price_df, width="stretch")
    prices = {m: Price(**row) for m, row in edited.to_dict(orient="index").items()}

st.title("🗄️ Prompt Caching Explorer")
st.markdown(
    "Four ways to avoid paying for the same LLM work twice. Work through the tabs in order: each one builds on "
    f"the last. **New to this?** Read [the guide]({DOC_LINK}) first."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "① Exact-match response cache",
    "② Paraphrases vs semantic cache",
    "③ Prefix (KV) cache",
    "④ Provider prompt caching",
])

# ---- ① exact-match response cache ----------------------------------------------------------------------

with tab1:
    st.subheader("① Exact-match response cache")
    st.markdown(
        "The application keeps a notebook of **prompt → answer**. Before calling the model, it looks the prompt "
        "up by its **MD5 hash**. Same text → same hash → reuse the stored answer, with no model call at all. "
        "This is the book's `basic_cache.py`."
    )
    cache = ExactMatchCache(CACHE_DIR)
    c1, c2 = st.columns([3, 1])
    prompt = c1.text_input("Prompt", BOOK_QUESTION, key="t1_prompt")
    max_new = c2.slider("Max answer tokens", 20, 300, 120, key="t1_max")
    use_cache = st.checkbox("Use the cache (`use_cache=True`)", value=True,
                            help="Turn off to always call the model, like `get_completion(prompt, use_cache=False)`.")

    if st.button("Ask", type="primary", key="t1_go"):
        lm = get_model(model_id)
        with st.spinner("Looking in the cache, calling the model on a miss…"):
            res = get_completion(cache, prompt, lambda p: lm.generate(answer_messages(p), max_new),
                                 model_id, use_cache)
        st.session_state["t1_last"] = res
        st.session_state.setdefault("t1_history", []).append(res)

    if res := st.session_state.get("t1_last"):
        badge = "🟢 **CACHE HIT**: answer read from disk" if res.hit else "🟠 **CACHE MISS**: the model was called"
        st.markdown(badge)
        m1, m2, m3 = st.columns(3)
        m1.metric("Time to answer", fmt_seconds(res.seconds))
        m2.metric("Cache key (MD5 of the prompt)", res.key[:12] + "…")
        m3.metric("Cache file", f"{res.key[:8]}….json")
        st.text_area("Answer", res.content, height=120, disabled=True, key=f"t1_ans_{len(st.session_state['t1_history'])}")

        hist = st.session_state["t1_history"]
        st.markdown("**Every request so far**")
        st.dataframe(pd.DataFrame([{
            "#": i + 1, "prompt": h.prompt, "result": "🟢 hit" if h.hit else "🟠 miss",
            "time": fmt_seconds(h.seconds), "key": h.key,
        } for i, h in enumerate(hist)]), hide_index=True, width="stretch")
        misses = [h.seconds for h in hist if not h.hit]
        hits = [h.seconds for h in hist if h.hit]
        if misses and hits:
            ratio = (sum(misses) / len(misses)) / max(sum(hits) / len(hits), 1e-6)
            st.success(f"On average a hit was **{ratio:,.0f}× faster** than calling the model.")

    st.markdown("---")
    st.markdown("#### Near misses: what counts as “the same prompt”?")
    st.markdown("Each variant below looks the same to a person. The hash doesn't care: one different character "
                "gives a completely different key, so the lookup misses. (This only looks; it never calls the model.)")
    rows = []
    for desc, v in near_miss_variants(prompt):
        rows.append({"change": desc, "prompt (quotes show spaces)": repr(v), "MD5 key": cache_key(v),
                     "lookup": "🟢 hit" if cache.get(v) else "🔴 miss"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if not cache.get(prompt):
        st.caption("Click **Ask** once first so the identical prompt is in the cache.")

    with st.expander(f"What's in the cache folder (`.prompt_cache/`, {len(cache.entries())} files)"):
        entries = cache.entries()
        if entries:
            st.dataframe(pd.DataFrame([{
                "file": f"{e.key}.json", "prompt": e.prompt, "model": e.model.split("/")[-1],
                "created": datetime.fromtimestamp(e.created_at).strftime("%H:%M:%S"),
                "original generation time": fmt_seconds(e.generation_seconds),
            } for e in entries]), hide_index=True, width="stretch")
            st.markdown("One file, exactly as stored on disk:")
            st.code(json.dumps(entries[-1].__dict__, indent=2), language="json")
        if st.button("Clear the cache", key="t1_clear"):
            cache.clear()
            st.session_state.pop("t1_last", None)
            st.session_state.pop("t1_history", None)
            st.rerun()

# ---- ② paraphrase pre-population vs semantic cache ----------------------------------------------------

DEFAULT_TESTS = [
    ("What is prompt caching?", True),
    ("what is prompt caching", True),
    ("Can you explain what prompt caching is?", True),
    ("Explain the concept of prompt caching in AI systems.", True),
    ("How does prompt caching reduce API costs?", False),
    ("Is prompt caching worth it?", False),
    ("What is not prompt caching?", False),
    ("What is response caching?", False),
    ("What is the capital of France?", False),
]

with tab2:
    st.subheader("② Paraphrase pre-population vs semantic cache")
    st.markdown(
        "Tab ① misses whenever the wording changes. Two fixes:\n"
        "- **Paraphrase pre-population** (the book's `semantic_cache.py`): ask an LLM for N rewordings and store "
        "the same answer under **each rewording's exact key**. Lookups are still exact-match.\n"
        "- **Semantic cache**: turn each prompt into an **embedding** (a list of numbers that captures meaning) and "
        "hit when a new prompt's embedding is **similar enough** (cosine similarity ≥ threshold)."
    )
    c1, c2, c3 = st.columns([3, 1, 1])
    seed = c1.text_input("Seed question (the one that gets answered and cached)", "What is prompt caching?",
                         key="t2_seed")
    n_para = c2.slider("Rewordings (N)", 3, 15, 10, key="t2_n", help="The book uses 10.")
    threshold = c3.slider("Similarity threshold", 0.50, 1.00, 0.85, 0.01, key="t2_thr")

    if st.button("Build both caches", type="primary", key="t2_go"):
        lm = get_model(model_id)
        embed = get_embedder()
        with st.spinner("Answering the seed question…"):
            answer = lm.generate(answer_messages(seed), 120)
        with st.spinner(f"Asking the local model for {n_para} rewordings…"):
            raw = lm.generate(paraphrase_messages(seed, n_para), 60 * n_para)
            paras = parse_paraphrases(raw, n_para, seed)
        pc = ParaphraseCache()
        pc.add([seed] + paras, answer)
        sc = SemanticCache(embed)
        sc.add(seed, answer)
        st.session_state["t2"] = {"seed": seed, "answer": answer, "paras": paras, "raw": raw, "pc": pc, "sc": sc}

    if t2 := st.session_state.get("t2"):
        a, b = st.columns(2)
        with a:
            st.markdown(f"**Paraphrase cache: {len(t2['paras']) + 1} exact keys**")
            st.dataframe(pd.DataFrame({"stored prompt": [t2["seed"]] + t2["paras"]}), hide_index=True,
                         width="stretch")
            with st.expander("The model's raw reply"):
                st.text(t2["raw"])
        with b:
            st.markdown("**Semantic cache: 1 embedding**")
            st.write(f"Stored: *{t2['seed']}*")
            st.markdown("**The cached answer** (both caches return this on a hit)")
            st.info(t2["answer"])

        st.markdown("#### Test questions")
        st.caption("Edit, add or remove rows. Tick **same meaning?** when the cached answer would be a correct reply.")
        tests = st.data_editor(
            pd.DataFrame(DEFAULT_TESTS, columns=["question", "same meaning?"]),
            num_rows="dynamic", hide_index=True, width="stretch", key="t2_tests",
        )
        sc: SemanticCache = t2["sc"]
        sc.threshold = threshold
        rows = []
        for q, same in zip(tests["question"], tests["same meaning?"]):
            if not isinstance(q, str) or not q.strip():
                continue
            same = bool(same)
            p = t2["pc"].lookup(q)
            s = sc.lookup(q)
            rows.append({
                "question": q, "same meaning?": same,
                "paraphrase cache": "hit" if p.hit else "miss", "paraphrase verdict": verdict(p.hit, same),
                "similarity": round(s.similarity, 3), "semantic cache": "hit" if s.hit else "miss",
                "semantic verdict": verdict(s.hit, same),
            })
        res = pd.DataFrame(rows)
        st.dataframe(res, hide_index=True, width="stretch",
                     column_config={"similarity": st.column_config.ProgressColumn(format="%.3f", min_value=0,
                                                                                   max_value=1)})

        def count(col, word):
            return int(res[col].str.contains(word).sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Paraphrase: false hits", count("paraphrase verdict", "false hit"))
        m2.metric("Paraphrase: missed", count("paraphrase verdict", "missed"))
        m3.metric("Semantic: false hits", count("semantic verdict", "false hit"))
        m4.metric("Semantic: missed", count("semantic verdict", "missed"))

        chart_df = res.assign(kind=res["same meaning?"].map({True: "same meaning", False: "different meaning"}))
        bars = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("similarity:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("question:N", sort="-x", title=None),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["same meaning", "different meaning"],
                                                      range=["#2e9e5b", "#d9534f"])),
        )
        rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color="black", strokeDash=[4, 4]).encode(x="t:Q")
        st.altair_chart(bars + rule, width="stretch")
        st.caption("Bars right of the dashed line are semantic-cache hits. A red bar right of the line is a false hit; "
                   "a green bar left of it is a missed reuse. Move the threshold slider: is there a value that "
                   "gets every row right?")

# ---- ③ prefix (KV) cache --------------------------------------------------------------------------------

LAYOUTS = {
    "Only the question changes (stable part first)": "question",
    "A timestamp is at the top of the system prompt": "timestamp",
    "One character changes inside the system prompt": "char",
}


def system_for(layout: str, when: str, frac: float) -> str:
    if layout == "timestamp":
        return f"Current time: {when}\n\n{BOOK_SYSTEM_PROMPT}"
    if layout == "char":
        return change_one_char(BOOK_SYSTEM_PROMPT, frac)[0]
    return BOOK_SYSTEM_PROMPT


def token_strip(lm: LanguageModel, ids: list[int], reused: int) -> str:
    spans = []
    for i, t in enumerate(ids):
        style = "background:#c8f0d2" if i < reused else "background:#ffd8a8"
        if i == reused:
            style += ";outline:2px solid #d9534f"
        spans.append(f'<span title="token {i}" style="{style};padding:1px 1px;margin:1px;border-radius:3px;'
                     f'font-family:monospace;font-size:12px">{html.escape(lm.token_label(t))}</span>')
    return ('<div style="max-height:260px;overflow-y:auto;line-height:1.9;border:1px solid #ddd;padding:8px">'
            + "".join(spans) + "</div>")


with tab3:
    st.subheader("③ Prefix (KV) cache")
    st.markdown(
        "Before writing a single word, the model reads the whole prompt and saves notes for every token: the "
        "**KV cache**. That reading pass is called **prefill**. If the next prompt **starts with exactly the same "
        "tokens**, those notes can be reused and only the new tail needs reading. Unlike tabs ① and ②, the model "
        "still writes a fresh answer, so it works even when every question is different."
    )
    c1, c2 = st.columns(2)
    q_old = c1.text_input("Earlier request: question (its KV cache gets saved)", "What is the capital of Monaco?",
                          key="t3_old")
    q_new = c2.text_input("New request: question", BOOK_QUESTION, key="t3_new")
    layout = LAYOUTS[st.radio("What else differs between the two requests?", list(LAYOUTS), key="t3_layout")]
    frac = 0.5
    if layout == "char":
        frac = st.slider("Where to change the character (how far into the system prompt)", 0.0, 1.0, 0.5, 0.05,
                         key="t3_frac")
    n_out = st.slider("Answer tokens to generate (to compare the outputs)", 10, 80, 30, key="t3_nout")

    if st.button("Run both prefills", type="primary", key="t3_go"):
        lm = get_model(model_id)
        old_sys = system_for(layout, "2026-11-08 09:00:00", 0) if layout != "char" else BOOK_SYSTEM_PROMPT
        new_sys = system_for(layout, "2026-11-08 09:00:07", frac)
        with st.spinner("Saving the earlier request's KV cache, then timing the new request with and without it…"):
            run = run_prefix_demo(
                lm,
                [{"role": "system", "content": old_sys}, {"role": "user", "content": q_old}],
                [{"role": "system", "content": new_sys}, {"role": "user", "content": q_new}],
                max_new_tokens=n_out,
            )
        st.session_state["t3"] = {"run": run, "model": model_id}

    if (t3 := st.session_state.get("t3")) and t3["model"] == model_id:
        run = t3["run"]
        lm = get_model(model_id)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tokens in the new prompt", len(run.new_ids))
        m2.metric("Reused from the cache", run.reused, f"{run.reused / len(run.new_ids):.0%}", delta_color="off")
        m3.metric("Recomputed", run.recomputed)
        m4.metric("Prefill speed-up", f"{run.cold_seconds / max(run.warm_seconds, 1e-6):.1f}×")
        m5, m6, m7 = st.columns(3)
        m5.metric("Prefill from scratch", fmt_seconds(run.cold_seconds))
        m6.metric("Prefill reusing the cache", fmt_seconds(run.warm_seconds))
        m7.metric("One-off cost to build the cache", fmt_seconds(run.build_seconds))
        st.caption("Times are the median of 3 runs on your machine. Only the prompt-reading part (prefill) is timed; "
                   "writing the answer takes the same time either way.")

        st.markdown("**The new prompt, token by token**: 🟩 reused from the cache · 🟧 recomputed · "
                    "red outline = first token that differs")
        st.markdown(token_strip(lm, run.new_ids, run.reused), unsafe_allow_html=True)

        a, b = st.columns(2)
        a.markdown("**Answer, prefill from scratch**")
        a.info(lm.decode(run.cold_output))
        b.markdown("**Answer, prefill reusing the cache**")
        b.info(lm.decode(run.warm_output))
        if run.outputs_match:
            st.success(f"Identical outputs (largest difference in next-token scores: {run.max_logit_diff:.2e}). "
                       "Reusing the cache changes the speed, never the answer.")
        else:
            st.warning("The outputs differ slightly. Floating-point rounding in a different order can flip a near-tie "
                       f"(largest score difference {run.max_logit_diff:.2e}).")

# ---- ④ provider prompt caching -------------------------------------------------------------------------

SYSTEM_CHOICES = {
    "The book's system prompt": BOOK_SYSTEM_PROMPT,
    "The book's system prompt × 4 (a bigger stable part)": "\n\n".join([BOOK_SYSTEM_PROMPT] * 4),
    "A short system prompt (below the 512-token minimum)": SHORT_SYSTEM_PROMPT,
}
MODES = {
    "No cache_control (the book's original anthropic_cache.py)": "none",
    "Cache breakpoint on the system prompt": "breakpoint",
}

with tab4:
    st.subheader("④ Provider prompt caching (Anthropic)")
    st.markdown(
        "Tab ③ on someone else's servers. You mark the end of the stable part with a **cache breakpoint** "
        "(`cache_control`). The first request is a **cache write** (charged a bit **more** than normal). Requests "
        "within the **TTL** that start with the identical prefix are **cache reads** (charged far **less**)."
    )
    if not api_key:
        st.warning("**Simulation mode.** No API key was found, so no requests are sent. Token counts are estimated "
                   "(about 4 characters per token) and the caching rules and prices are applied exactly as the "
                   "provider describes them. Add a key in the sidebar for real calls.")
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Claude model", list(PRICES), key="t4_model")
    mode = MODES[c2.radio("Request style", list(MODES), index=1, key="t4_mode")]
    ttl = c3.radio("TTL (how long an unused entry lives)", ["5m", "1h"], key="t4_ttl",
                   help="1h writes cost 2× base input instead of 1.25×, but survive longer gaps between requests.")
    system = SYSTEM_CHOICES[st.selectbox("System prompt (the stable part)", list(SYSTEM_CHOICES), key="t4_sys")]
    questions_text = st.text_area("Questions: one request per line, sent in order",
                                  "What is the capital of Monaco?\nWhat is prompt caching?\nName three uses of a cache.",
                                  key="t4_qs")
    max_tokens = st.slider("max_tokens per answer", 20, 400, 150, key="t4_max")
    questions = [q.strip() for q in questions_text.splitlines() if q.strip()]

    price = prices[model]
    approx_prefix = round(len(system) / 4)
    worst = len(questions) * (approx_prefix * max(price.write_1h, price.base_input) + max_tokens * price.output) / 1e6
    st.caption(f"Estimated worst-case cost of these {len(questions)} requests: **{fmt_usd(worst)}**. "
               f"Stable part ≈ {approx_prefix} tokens (minimum for caching on this model: "
               f"{MIN_CACHEABLE_TOKENS[model]}). One cache write pays for itself after "
               f"{break_even_reads(price, ttl):.2f} reads.")

    with st.expander("The request that will be sent (first question)"):
        req = build_request(model, system[:120] + ("…" if len(system) > 120 else ""), questions[0] if questions else "",
                            mode, ttl, max_tokens)
        st.code(json.dumps(req, indent=2), language="json")

    b1, b2 = st.columns([1, 4])
    go = b1.button("Send requests", type="primary", key="t4_go", disabled=not questions)
    if b2.button("Clear results", key="t4_clear"):
        st.session_state.pop("t4_calls", None)
    if go:
        calls = st.session_state.setdefault("t4_calls", [])
        sim = st.session_state.setdefault("t4_sim", Simulator())
        client = None
        if api_key:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        progress = st.progress(0.0)
        try:
            for i, q in enumerate(questions):
                request = build_request(model, system, q, mode, ttl, max_tokens)
                result = call_anthropic(client, request) if client else sim.call(request)
                calls.append({"result": result, "model": model, "mode": mode, "ttl": ttl})
                progress.progress((i + 1) / len(questions))
        except Exception as e:  # show API errors (bad key, rate limit) instead of crashing the tab
            st.error(f"The API call failed: {e}")

    if calls := st.session_state.get("t4_calls"):
        rows, totals = [], {"cost": 0.0, "without": 0.0}
        for i, c in enumerate(calls):
            r, u = c["result"], c["result"].usage
            cost = cost_of(u, prices[c["model"]])
            totals["cost"] += cost.total
            totals["without"] += cost.without_caching
            what = ("✍️ cache write" if u.cache_write else "📖 cache read" if u.cache_read else "— nothing cached")
            rows.append({
                "#": i + 1, "question": r.question, "model": c["model"], "style": c["mode"], "result": what,
                "input (after breakpoint)": u.input_tokens, "cache write": u.cache_write, "cache read": u.cache_read,
                "output": u.output_tokens, "latency": "n/a (simulated)" if r.simulated else fmt_seconds(r.seconds), "cost": fmt_usd(cost.total),
                "cost without caching": fmt_usd(cost.without_caching), "simulated": r.simulated,
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width="stretch")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total cost", fmt_usd(totals["cost"]))
        m2.metric("Same requests without caching", fmt_usd(totals["without"]))
        saved = totals["without"] - totals["cost"]
        m3.metric("Saved", fmt_usd(saved), f"{saved / totals['without']:.0%}" if totals["without"] else None)

        long_df = df.melt(id_vars=["#"], value_vars=["input (after breakpoint)", "cache write", "cache read", "output"],
                          var_name="token type", value_name="tokens")
        st.altair_chart(alt.Chart(long_df).mark_bar().encode(
            x=alt.X("#:O", title="request"), y="tokens:Q",
            color=alt.Color("token type:N", scale=alt.Scale(
                domain=["input (after breakpoint)", "cache write", "cache read", "output"],
                range=["#9aa5b1", "#f0ad4e", "#2e9e5b", "#5b8def"])),
        ), width="stretch")

        last = calls[-1]
        lu = last["result"].usage
        if last["mode"] == "breakpoint" and not lu.cache_write and not lu.cache_read:
            st.warning("A breakpoint was set but nothing was cached: the stable part is shorter than the model's "
                       "minimum cacheable length. The API doesn't raise an error; both counters just stay at 0.")
        elif last["mode"] == "none":
            st.info("No `cache_control` → nothing is cached, however long the prompt is. This is why the book's "
                    "`anthropic_cache.py` printed 0 for both cache counters.")
        if lu.cache_write or lu.cache_read:
            expires = last["result"].at + TTL_SECONDS[last["ttl"]]
            left = expires - time.time()
            st.caption(f"The cache entry expires at **{datetime.fromtimestamp(expires).strftime('%H:%M:%S')}** "
                       f"({max(left, 0) / 60:.1f} min from now) unless another request reads it first. "
                       "Every read restarts the clock.")
