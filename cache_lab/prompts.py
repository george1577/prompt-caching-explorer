"""Prompts and example questions shared by the tabs."""

from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

# The long system prompt from the book's anthropic_cache.py / openai_example_server.py, copied verbatim.
BOOK_SYSTEM_PROMPT = (DATA / "book_system_prompt.txt").read_text()

SHORT_SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and briefly."

# The book's example prompt for the response caches.
BOOK_QUESTION = "Explain the concept of prompt caching in AI systems."

ANSWER_SYSTEM = "You are a helpful assistant. Answer in at most three sentences."


def answer_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": question}]


def paraphrase_messages(question: str, n: int) -> list[dict]:
    """Ask the model for n rewordings: the job semantic_cache.py gives Claude."""
    return [
        {"role": "system", "content": "You rewrite questions. Reply with a numbered list and nothing else."},
        {"role": "user", "content": (
            f"Write {n} different ways a user might ask exactly this question. "
            f"Keep the same meaning and language.\n\nQuestion: {question}"
        )},
    ]


def near_miss_variants(prompt: str) -> list[tuple[str, str]]:
    """(description, variant) pairs that look the same to a person but differ by a character or two."""
    stripped = prompt.rstrip("?.! ")
    variants = [
        ("identical", prompt),
        ("trailing space", prompt + " "),
        ("all lowercase", prompt.lower()),
        ("double space", prompt.replace(" ", "  ", 1)),
        ("different ending", stripped + ("?" if not prompt.endswith("?") else ".")),
        ("leading 'Please '", "Please " + prompt[:1].lower() + prompt[1:]),
    ]
    return variants
