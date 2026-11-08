"""The only module that touches the local language model. Everything else works on plain text and token ids."""

import copy
import os
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]


def pick_device() -> str:
    # Apple's MPS backend is opt-in: recent torch builds crash on older macOS versions,
    # and a 0.5B model is fast enough on CPU. Set PCL_DEVICE=mps (or cuda) to override.
    if os.environ.get("PCL_DEVICE"):
        return os.environ["PCL_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Prefill:
    """The result of reading a prompt: the KV cache it produced, the next-token logits, and how long it took."""

    kv_cache: object
    last_logits: torch.Tensor
    seconds: float
    tokens_computed: int


class LanguageModel:
    def __init__(self, model_id: str = MODELS[0]):
        self.model_id = model_id
        self.device = pick_device()
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device)
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])

    # ---- text <-> ids ---------------------------------------------------------------

    def chat_text(self, messages: list[dict]) -> str:
        """The exact text the model reads for a chat, including the template's special markers."""
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def chat_ids(self, messages: list[dict]) -> list[int]:
        return self.encode(self.chat_text(messages))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def token_label(self, token_id: int) -> str:
        """A token's text with whitespace made visible."""
        return self.tokenizer.decode([token_id]).replace(" ", "␣").replace("\n", "⏎")

    # ---- model calls ----------------------------------------------------------------

    @torch.no_grad()
    def prefill(self, ids: list[int], reuse: Prefill | None = None, reuse_tokens: int = 0) -> Prefill:
        """Read `ids`. If `reuse` is given, keep its KV cache for the first `reuse_tokens` tokens and
        only compute the rest. The cached Prefill is copied, never modified."""
        past = None
        start = 0
        if reuse is not None and reuse_tokens > 0:
            reuse_tokens = min(reuse_tokens, len(ids) - 1)  # the model must read at least one new token
            past = copy.deepcopy(reuse.kv_cache)
            extra = past.get_seq_length() - reuse_tokens
            if extra > 0:
                past.crop(-extra)  # newer transformers: a negative number = tokens to drop from the end
            start = reuse_tokens
        new = ids[start:]
        t0 = time.perf_counter()
        out = self.model(input_ids=torch.tensor([new], device=self.device), past_key_values=past, use_cache=True)
        logits = out.logits[0, -1].float().cpu()
        seconds = time.perf_counter() - t0
        return Prefill(out.past_key_values, logits, seconds, len(new))

    @torch.no_grad()
    def continue_greedy(self, state: Prefill, max_new_tokens: int) -> list[int]:
        """Greedy-decode from the end of a prefill (copies the cache, so `state` can be reused)."""
        past = copy.deepcopy(state.kv_cache)
        token = int(state.last_logits.argmax())
        out_ids = []
        for _ in range(max_new_tokens):
            if token in self.eos_ids:
                break
            out_ids.append(token)
            out = self.model(input_ids=torch.tensor([[token]], device=self.device), past_key_values=past,
                             use_cache=True)
            past = out.past_key_values
            token = int(out.logits[0, -1].argmax())
        return out_ids

    def generate(self, messages: list[dict], max_new_tokens: int = 150) -> str:
        """A plain greedy chat completion."""
        state = self.prefill(self.chat_ids(messages))
        return self.decode(self.continue_greedy(state, max_new_tokens)).strip()
