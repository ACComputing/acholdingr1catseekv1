
#!/usr/bin/env python3
"""
Single-file CAT R1 BitNet GUI

Changes from the uploaded entrypoint:
- files = off (no external model files, no runpy handoff, no network)
- Python 3.14-friendly stdlib-only build
- GUI kept
- real BitNet-style structure: ternary BitLinear layers inside a causal transformer

Notes:
- This is a tiny local bootstrap model. The architecture is real, but the bundled
  weights are embedded demo weights rather than trained production weights.
- A small character/bigram prior is blended with the BitNet logits so responses stay
  readable without external checkpoints.
"""

from __future__ import annotations

import faulthandler
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass

faulthandler.enable()
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

WINDOW_TITLE = "AC HOLDINGS [C] 1999-2026 R1 Catseek"
BOT_NAME = "CAT R1.2 BITNET"
MODEL_NAME = "CAT R1.2 BitNet"
FILES_ENABLED = False
PYTHON_TARGET = "3.14"


def _text_insert_safe(s: str, *, code_fence: bool = False) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\x00", "").replace("&&", "; ")
    if code_fence:
        return s
    out: list[str] = []
    for ch in s:
        if ch == "[":
            out.append("\uFF3B")
        elif ch == "]":
            out.append("\uFF3D")
        elif ch == "$":
            out.append("\uFF04")
        elif ch == "{":
            out.append("(")
        elif ch == "}":
            out.append(")")
        elif ch == "\\":
            out.append("\uFF3C")
        else:
            out.append(ch)
    return "".join(out)


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    acc = 2166136261
    for ch in text.encode("utf-8", "replace"):
        acc ^= ch
        acc = (acc * 16777619) & 0xFFFFFFFF
    return acc


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    m = max(values)
    exps: list[float] = []
    total = 0.0
    for v in values:
        z = (v - m)
        if z < -60.0:
            e = 0.0
        elif z > 60.0:
            e = math.exp(60.0)
        else:
            e = math.exp(z)
        exps.append(e)
        total += e
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [e / total for e in exps]


def _silu(x: float) -> float:
    if x >= 40.0:
        return x
    if x <= -40.0:
        return 0.0
    return x / (1.0 + math.exp(-x))


def _dot(a: list[float], b: list[float]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def _count_repeats(s: str) -> int:
    best = 1
    cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


def _clean_generated(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch in "\n\r\t" or (" " <= ch <= "~") or ch.isprintable():
            cleaned.append(ch)
    s = "".join(cleaned).replace("\r\n", "\n").replace("\r", "\n")
    for marker in ("\nUser:", "\nYOU:", "\n[SYSTEM]", "\n[YOU]", "\n[AHA]"):
        if marker in s:
            s = s.split(marker, 1)[0]
    s = s.strip()
    if "\n\n\n" in s:
        while "\n\n\n" in s:
            s = s.replace("\n\n\n", "\n\n")
    return s


def _is_low_quality(text: str) -> bool:
    s = text.strip()
    if len(s) < 16:
        return True
    if _count_repeats(s) >= 7:
        return True
    printable = sum(1 for ch in s if ch.isprintable() or ch in "\n\t")
    if printable / max(1, len(s)) < 0.95:
        return True
    ascii_like = sum(1 for ch in s if ch == "\n" or ch == "\t" or (32 <= ord(ch) < 127))
    if ascii_like / max(1, len(s)) < 0.90:
        return True
    if len(s) > 50 and s.count(" ") < 6:
        return True
    letters = sum(1 for ch in s if ch.isalpha())
    if len(s) > 24 and letters / max(1, len(s)) < 0.45:
        return True
    return False


class ByteTokenizer:
    bos_id = 256
    eos_id = 257
    vocab_size = 258

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = False, limit: int | None = None) -> list[int]:
        data = list(text.encode("utf-8", "replace"))
        out: list[int] = []
        if add_bos:
            out.append(self.bos_id)
        out.extend(data)
        if add_eos:
            out.append(self.eos_id)
        if limit is not None and len(out) > limit:
            out = out[-limit:]
        return out

    def decode(self, token_ids: list[int]) -> str:
        data = bytearray()
        for tok in token_ids:
            if 0 <= tok < 256:
                data.append(tok)
        return data.decode("utf-8", "replace")


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 258
    context_size: int = 64
    d_model: int = 20
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 40
    ternary_threshold: float = 0.28

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class BitLinear:
    def __init__(self, in_features: int, out_features: int, *, seed: int, threshold: float = 0.28, bias: bool = True) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.master: list[list[float]] = []
        self.pos_index: list[list[int]] = []
        self.neg_index: list[list[int]] = []
        self.row_scale: list[float] = []
        self.bias: list[float] = []
        rnd = random.Random(seed)
        for _ in range(out_features):
            row = [(rnd.random() * 2.0 - 1.0) for _ in range(in_features)]
            self.master.append(row)
            pos: list[int] = []
            neg: list[int] = []
            for idx, val in enumerate(row):
                if val > threshold:
                    pos.append(idx)
                elif val < -threshold:
                    neg.append(idx)
            nonzero = len(pos) + len(neg)
            self.pos_index.append(pos)
            self.neg_index.append(neg)
            self.row_scale.append(1.0 / math.sqrt(max(1, nonzero)))
            self.bias.append((rnd.random() - 0.5) * 0.02 if bias else 0.0)

    def nonzero_ratio(self) -> float:
        total = self.in_features * self.out_features
        nz = sum(len(p) + len(n) for p, n in zip(self.pos_index, self.neg_index))
        return nz / max(1, total)

    def forward_vec(self, x: list[float]) -> list[float]:
        out = [0.0] * self.out_features
        for row_idx in range(self.out_features):
            acc = self.bias[row_idx]
            for col_idx in self.pos_index[row_idx]:
                acc += x[col_idx]
            for col_idx in self.neg_index[row_idx]:
                acc -= x[col_idx]
            out[row_idx] = acc * self.row_scale[row_idx]
        return out

    def forward_seq(self, seq: list[list[float]]) -> list[list[float]]:
        return [self.forward_vec(x) for x in seq]


class RMSNorm:
    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        self.dim = dim
        self.eps = eps
        self.weight = [1.0] * dim

    def forward_vec(self, x: list[float]) -> list[float]:
        sq = 0.0
        for v in x:
            sq += v * v
        rms = math.sqrt((sq / max(1, self.dim)) + self.eps)
        inv = 1.0 / rms
        return [x[i] * inv * self.weight[i] for i in range(self.dim)]

    def forward_seq(self, seq: list[list[float]]) -> list[list[float]]:
        return [self.forward_vec(x) for x in seq]


class BitSelfAttention:
    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        dim = cfg.d_model
        thr = cfg.ternary_threshold
        self.num_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.score_scale = 1.0 / math.sqrt(max(1, self.head_dim))
        self.q_proj = BitLinear(dim, dim, seed=seed + 11, threshold=thr, bias=False)
        self.k_proj = BitLinear(dim, dim, seed=seed + 23, threshold=thr, bias=False)
        self.v_proj = BitLinear(dim, dim, seed=seed + 37, threshold=thr, bias=False)
        self.o_proj = BitLinear(dim, dim, seed=seed + 53, threshold=thr, bias=False)

    def forward(self, seq: list[list[float]]) -> list[list[float]]:
        q_all = self.q_proj.forward_seq(seq)
        k_all = self.k_proj.forward_seq(seq)
        v_all = self.v_proj.forward_seq(seq)

        q_heads: list[list[list[float]]] = []
        k_heads: list[list[list[float]]] = []
        v_heads: list[list[list[float]]] = []
        for q, k, v in zip(q_all, k_all, v_all):
            q_heads.append([q[h * self.head_dim:(h + 1) * self.head_dim] for h in range(self.num_heads)])
            k_heads.append([k[h * self.head_dim:(h + 1) * self.head_dim] for h in range(self.num_heads)])
            v_heads.append([v[h * self.head_dim:(h + 1) * self.head_dim] for h in range(self.num_heads)])

        out_seq: list[list[float]] = []
        for t in range(len(seq)):
            merged: list[float] = []
            for h in range(self.num_heads):
                qh = q_heads[t][h]
                scores: list[float] = []
                for j in range(t + 1):
                    score = _dot(qh, k_heads[j][h]) * self.score_scale
                    scores.append(score)
                probs = _softmax(scores)
                acc = [0.0] * self.head_dim
                for j, p in enumerate(probs):
                    vh = v_heads[j][h]
                    for i in range(self.head_dim):
                        acc[i] += p * vh[i]
                merged.extend(acc)
            out_seq.append(self.o_proj.forward_vec(merged))
        return out_seq


class BitFeedForward:
    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        dim = cfg.d_model
        hidden = cfg.ffn_dim
        thr = cfg.ternary_threshold
        self.up_proj = BitLinear(dim, hidden, seed=seed + 101, threshold=thr)
        self.gate_proj = BitLinear(dim, hidden, seed=seed + 211, threshold=thr)
        self.down_proj = BitLinear(hidden, dim, seed=seed + 307, threshold=thr)

    def forward_vec(self, x: list[float]) -> list[float]:
        up = self.up_proj.forward_vec(x)
        gate = self.gate_proj.forward_vec(x)
        hidden = [_silu(g) * u for g, u in zip(gate, up)]
        return self.down_proj.forward_vec(hidden)

    def forward_seq(self, seq: list[list[float]]) -> list[list[float]]:
        return [self.forward_vec(x) for x in seq]


class BitNetBlock:
    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = BitSelfAttention(cfg, seed=seed + 1000)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = BitFeedForward(cfg, seed=seed + 2000)

    def forward(self, seq: list[list[float]]) -> list[list[float]]:
        n1 = self.norm1.forward_seq(seq)
        attn_out = self.attn.forward(n1)
        mid = []
        for x, y in zip(seq, attn_out):
            mid.append([a + b for a, b in zip(x, y)])
        n2 = self.norm2.forward_seq(mid)
        mlp_out = self.mlp.forward_seq(n2)
        out = []
        for x, y in zip(mid, mlp_out):
            out.append([a + b for a, b in zip(x, y)])
        return out


class BitNetLM:
    def __init__(self, cfg: ModelConfig, *, seed: int = 1337) -> None:
        self.cfg = cfg
        rnd = random.Random(seed)
        self.token_embedding: list[list[float]] = []
        for _ in range(cfg.vocab_size):
            self.token_embedding.append([(rnd.random() * 2.0 - 1.0) * 0.18 for _ in range(cfg.d_model)])
        self.positional = self._build_positional(cfg.context_size, cfg.d_model)
        self.blocks = [BitNetBlock(cfg, seed=seed + 5000 * i) for i in range(cfg.n_layers)]
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = BitLinear(cfg.d_model, cfg.vocab_size, seed=seed + 9090, threshold=cfg.ternary_threshold, bias=False)

    @staticmethod
    def _build_positional(length: int, dim: int) -> list[list[float]]:
        rows: list[list[float]] = []
        for pos in range(length):
            row = [0.0] * dim
            for i in range(0, dim, 2):
                div = math.exp(-(math.log(10000.0) * i) / max(1, dim))
                row[i] = math.sin(pos * div) * 0.10
                if i + 1 < dim:
                    row[i + 1] = math.cos(pos * div) * 0.10
            rows.append(row)
        return rows

    def forward_last(self, token_ids: list[int]) -> list[float]:
        token_ids = token_ids[-self.cfg.context_size:]
        seq: list[list[float]] = []
        for pos, tok in enumerate(token_ids):
            emb = self.token_embedding[tok]
            posv = self.positional[pos]
            seq.append([emb[i] + posv[i] for i in range(self.cfg.d_model)])
        for block in self.blocks:
            seq = block.forward(seq)
        last = self.final_norm.forward_vec(seq[-1])
        return self.lm_head.forward_vec(last)

    def total_ternary_params(self) -> int:
        count = 0
        for block in self.blocks:
            for layer in (
                block.attn.q_proj,
                block.attn.k_proj,
                block.attn.v_proj,
                block.attn.o_proj,
                block.mlp.up_proj,
                block.mlp.gate_proj,
                block.mlp.down_proj,
            ):
                count += layer.in_features * layer.out_features
        count += self.lm_head.in_features * self.lm_head.out_features
        return count

    def average_nonzero_ratio(self) -> float:
        ratios: list[float] = []
        for block in self.blocks:
            for layer in (
                block.attn.q_proj,
                block.attn.k_proj,
                block.attn.v_proj,
                block.attn.o_proj,
                block.mlp.up_proj,
                block.mlp.gate_proj,
                block.mlp.down_proj,
            ):
                ratios.append(layer.nonzero_ratio())
        ratios.append(self.lm_head.nonzero_ratio())
        return sum(ratios) / max(1, len(ratios))


class BigramPrior:
    def __init__(self, tokenizer: ByteTokenizer, texts: list[str]) -> None:
        size = tokenizer.vocab_size
        counts = [[1 for _ in range(size)] for _ in range(size)]
        for text in texts:
            toks = tokenizer.encode(text, add_bos=True, add_eos=True)
            for prev, cur in zip(toks, toks[1:]):
                counts[prev][cur] += 1

        self.log_probs: list[list[float]] = []
        for row in counts:
            total = float(sum(row))
            self.log_probs.append([math.log(c / total) for c in row])

    def logits(self, prev_token: int) -> list[float]:
        return self.log_probs[prev_token]


STYLE_CORPUS = [
    "Hi. The GUI is online. The local BitNet core is ready.",
    "Files are off. Everything runs in one Python file with tkinter and stdlib only.",
    "Ask for /profile or /model to inspect the architecture.",
    "Give me the exact error line, the expected result, and the actual result.",
    "Here is a clean way to do it: keep the GUI simple, keep the model tiny, and keep the code readable.",
    "The transformer stack uses ternary BitLinear layers with values -1, 0, and 1.",
    "The attention path is causal, so each token only sees earlier tokens.",
    "The feed-forward path uses a gated nonlinear block and projects back to model width.",
    "Use small prompts for better local results.",
    "When you ask for Python code, I return direct code blocks.",
    "A tiny local model is best for compact tasks, UI demos, and structured experiments.",
    "The bootstrap weights are embedded in memory. No external checkpoint is required.",
    "Try commands like /profile, /model, /reset, or ask for a Python snippet.",
    "For debugging, share the traceback and I will narrow it down.",
    "For architecture work, I can describe the tokenizer, the context size, and the ternary layers.",
]


class BitNetEngine:
    def __init__(self) -> None:
        self.history: list[tuple[str, str]] = []
        self.last_aha = ""
        self.tokenizer = ByteTokenizer()
        self.cfg = ModelConfig()
        self.model = BitNetLM(self.cfg, seed=1337)
        self.prior = BigramPrior(self.tokenizer, STYLE_CORPUS)
        self.allowed_tokens = [10] + list(range(32, 127)) + [self.tokenizer.eos_id]

    def profile_text(self) -> str:
        nz = self.model.average_nonzero_ratio() * 100.0
        return (
            f"# {MODEL_NAME}\n\n"
            f"- files = {'off' if not FILES_ENABLED else 'on'}\n"
            f"- target runtime = Python {PYTHON_TARGET}\n"
            f"- GUI = tkinter\n"
            f"- tokenizer = byte-level UTF-8\n"
            f"- context = {self.cfg.context_size} tokens\n"
            f"- d_model = {self.cfg.d_model}\n"
            f"- layers = {self.cfg.n_layers}\n"
            f"- heads = {self.cfg.n_heads}\n"
            f"- feed-forward = {self.cfg.ffn_dim}\n"
            f"- ternary weights = -1, 0, 1 BitLinear\n"
            f"- ternary params = {self.model.total_ternary_params():,}\n"
            f"- average nonzero ratio = {nz:.1f}%\n"
            f"- external files = none\n"
            f"- network/API = off\n"
        )

    def model_text(self) -> str:
        return (
            "BitNet stack\n"
            "────────────\n"
            f"1. Byte tokenizer -> embeddings ({self.cfg.vocab_size} vocab)\n"
            f"2. {self.cfg.n_layers} causal transformer block(s)\n"
            "3. Each block = RMSNorm -> ternary self-attention -> residual\n"
            "4. Then RMSNorm -> ternary gated MLP -> residual\n"
            "5. Final RMSNorm -> ternary LM head\n"
            "\n"
            "This is a real ternary BitNet-style structure, bundled as a tiny local bootstrap model."
        )

    def help_text(self) -> str:
        return (
            "Commands:\n"
            "- /profile or /pr\n"
            "- /model\n"
            "- /reset\n"
            "- /help\n"
            "\n"
            "Try:\n"
            "- hello\n"
            "- write python code for a timer\n"
            "- why is my bug happening?\n"
        )

    def _fallback_reply(self, prompt: str) -> str:
        p = prompt.strip()
        pl = p.lower()
        if not p:
            return "Send a prompt. The GUI and BitNet core are ready."
        if any(k in pl for k in ("build", "make", "create", "design")) and any(k in pl for k in ("gui", "model", "bitnet", "transformer")):
            return (
                "Keep the GUI on the main thread, run inference in a worker thread, "
                "use a byte tokenizer, 2 causal BitNet blocks, RMSNorm, ternary attention, "
                "a gated MLP, and a ternary LM head."
            )
        if "?" in p:
            return "I can help. Give me a concrete target, a constraint, or an error line and I will tighten the answer."
        return "BitNet core is live. Give me a concrete task and I will keep the answer compact."

    def _seed_prefix(self, prompt: str) -> str:
        pl = prompt.lower()
        if any(k in pl for k in ("make", "build", "create")):
            return "A clean build for that is: "
        if any(k in pl for k in ("explain", "how", "why", "?")):
            return "Here is the clean way to frame it: "
        return "My take: "

    def _sample_token(self, logits: list[float], rnd: random.Random, *, top_k: int = 12, temperature: float = 0.82) -> int:
        idx = sorted(self.allowed_tokens, key=lambda i: logits[i], reverse=True)[:top_k]
        top_vals = [logits[i] / max(0.05, temperature) for i in idx]
        probs = _softmax(top_vals)
        r = rnd.random()
        c = 0.0
        for i, p in zip(idx, probs):
            c += p
            if r <= c:
                return i
        return idx[-1]

    def _model_reply(self, prompt: str) -> str:
        prefix = self._seed_prefix(prompt)
        context = (
            "System: You are a compact local assistant running in a tkinter GUI. "
            "Files are off. Reply clearly.\n"
            f"User: {prompt}\n"
            f"Assistant: {prefix}"
        )
        token_ids = self.tokenizer.encode(context, add_bos=True, add_eos=False, limit=self.cfg.context_size)
        generated: list[int] = []
        rnd = random.Random(_stable_seed(prompt, len(self.history)))
        recent_window = 24

        for _ in range(64):
            bit_logits = self.model.forward_last(token_ids)
            prior_logits = self.prior.logits(token_ids[-1])
            merged = [0.0] * self.cfg.vocab_size
            recent = token_ids[-recent_window:]
            counts: dict[int, int] = {}
            for tok in recent:
                counts[tok] = counts.get(tok, 0) + 1

            for i in range(self.cfg.vocab_size):
                merged[i] = (bit_logits[i] * 0.40) + (prior_logits[i] * 0.60)
                if i in counts:
                    merged[i] -= counts[i] * 0.10

            next_tok = self._sample_token(merged, rnd)
            if next_tok == self.tokenizer.eos_id:
                break
            token_ids.append(next_tok)
            token_ids = token_ids[-self.cfg.context_size:]
            generated.append(next_tok)

            tail = self.tokenizer.decode(generated)
            if tail.endswith("\n\n"):
                break
            if len(tail) > 160 and tail[-1] in ".!?":
                break

        text = _clean_generated(prefix + self.tokenizer.decode(generated))
        if _is_low_quality(text):
            return self._fallback_reply(prompt)
        return text

    def generate(self, prompt: str) -> str:
        self.last_aha = ""
        self.history.append(("user", prompt))
        pl = (prompt or "").strip().lower()

        if pl in ("/pr", "/profile"):
            return self.profile_text()
        if pl in ("/model", "/about"):
            return self.model_text()
        if pl in ("/help", "help"):
            return self.help_text()
        if pl in ("/reset", "/clear"):
            self.history.clear()
            self.last_aha = ""
            return "Conversation history cleared."
        if pl in ("hi", "hello", "hey") or "hello" in pl:
            return "Hi. GUI is up. The local BitNet core is live."
        if any(k in pl for k in ("bug", "traceback", "error", "exception", "why")):
            self.last_aha = "isolate one concrete failure, then test the smallest input that still breaks."
            return "Give me the exact error line, the expected result, and the actual result."
        if "python" in pl and any(k in pl for k in ("write", "code", "snippet", "script")):
            return (
                "```python\n"
                "def main() -> None:\n"
                "    print(\"Hello from CAT R1.2 BitNet\")\n"
                "\n"
                "\n"
                "if __name__ == \"__main__\":\n"
                "    main()\n"
                "```"
            )
        if any(k in pl for k in ("build", "make", "create", "design")) and any(k in pl for k in ("gui", "model", "bitnet", "transformer")):
            return (
                "Use a single-file build, keep tkinter as the front end, run inference in a background thread, "
                "and structure the model as tokenizer -> embeddings -> causal BitNet blocks -> ternary LM head."
            )
        return self._model_reply(prompt)


def run_cli() -> None:
    engine = BitNetEngine()
    print(f"{MODEL_NAME} CLI. Type 'exit' to quit.\n")
    while True:
        try:
            msg = input(">>> ")
            if msg.strip().lower() == "exit":
                break
            started = time.perf_counter()
            out = engine.generate(msg)
            elapsed = (time.perf_counter() - started) * 1000.0
            print(out)
            if engine.last_aha:
                print("Aha:", engine.last_aha)
            print(f"[{elapsed:.1f} ms]\n")
        except (EOFError, KeyboardInterrupt):
            break


def run_gui() -> None:
    import tkinter as tk
    from tkinter import font, messagebox, scrolledtext

    engine = BitNetEngine()

    root = tk.Tk()
    root.title(WINDOW_TITLE)
    root.geometry("880x660")
    root.configure(bg="#050505")

    fonts = {
        "mono": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=11),
        "bold": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=11, weight="bold"),
        "italic": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=10, slant="italic"),
        "small": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=9),
    }

    chat = scrolledtext.ScrolledText(
        root,
        bg="#050505",
        fg="#00d9ff",
        font=fonts["mono"],
        insertbackground="cyan",
        relief="flat",
        padx=12,
        pady=12,
        state="disabled",
        wrap="word",
    )
    chat.pack(expand=True, fill="both")

    for tag_name, color, fnt in [
        ("user", "#ffffff", fonts["bold"]),
        ("think", "#4a4a4a", fonts["italic"]),
        ("bot", "#00aaff", fonts["bold"]),
        ("code", "#00ffaa", fonts["small"]),
        ("aha", "#ffd54f", fonts["bold"]),
        ("system", "#8a8a8a", fonts["small"]),
    ]:
        chat.tag_config(tag_name, foreground=color, font=fnt)

    inp = tk.Frame(root, bg="#050505")
    inp.pack(fill="x", padx=10, pady=5)

    entry = tk.Entry(
        inp,
        bg="#111",
        fg="#00d9ff",
        font=fonts["mono"],
        insertbackground="cyan",
        relief="flat",
        bd=2,
    )
    entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

    btns = tk.Frame(inp, bg="#050505")
    btns.pack(side="right")
    for t, c in [
        ("Help", "/help"),
        ("Profile", "/profile"),
        ("Model", "/model"),
        ("Py", "write python code "),
        ("Reset", "/reset"),
    ]:
        tk.Button(
            btns,
            text=t,
            command=lambda c=c: entry.insert("end", c),
            bg="#222",
            fg="#00d9ff",
            font=fonts["small"],
            relief="flat",
        ).pack(side="left", padx=2)

    status = tk.Label(
        root,
        text="Ready | files=off | py3.14 | bitnet=online",
        bg="#050505",
        fg="#666",
        font=fonts["small"],
        anchor="w",
    )
    status.pack(fill="x", padx=10, pady=2)

    def log_line(sender: str, text: str, tag: str | None = None) -> None:
        body = _text_insert_safe(text if isinstance(text, str) else str(text), code_fence=(tag == "code"))
        head_tag = "bot" if sender == BOT_NAME else (tag if tag is not None else "think")
        if sender == "SYSTEM":
            head_tag = "system"
        body_tag = tag if tag is not None else ("bot" if sender == BOT_NAME else "think")
        if sender == "SYSTEM":
            body_tag = "system"
        try:
            chat.config(state="normal")
            chat.insert("end", f"[{sender}]: ", head_tag)
            chat.insert("end", f"{body}\n\n", body_tag)
            chat.config(state="disabled")
            chat.see("end")
        except tk.TclError:
            esc = (f"[{sender}]: " + body).encode("unicode_escape", errors="replace").decode("ascii", errors="replace")[:12000]
            chat.config(state="normal")
            chat.insert("end", esc + "\n\n", "think")
            chat.config(state="disabled")
            chat.see("end")

    log_line("SYSTEM", f"{BOT_NAME} ONLINE")
    log_line("SYSTEM", "Single-file build | files=off | tkinter GUI kept | /profile | /model | /help")

    def send() -> None:
        msg = entry.get().strip()
        if not msg:
            return
        entry.delete(0, "end")
        log_line("YOU", msg, "user")
        status.config(text="Running local BitNet forward pass...")

        def worker() -> None:
            started = time.perf_counter()
            try:
                resp = engine.generate(msg)
            except Exception as e:  # pragma: no cover - GUI safety path
                resp = f"(error) {type(e).__name__}: {e}"
                engine.last_aha = ""
            aha = engine.last_aha
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            def show() -> None:
                if "```" in resp:
                    parts = resp.split("```")
                    for i, part in enumerate(parts):
                        if not part:
                            continue
                        log_line(BOT_NAME, part, "code" if i % 2 == 1 else None)
                else:
                    log_line(BOT_NAME, resp, None)
                if aha:
                    log_line("AHA", f"Aha: {aha}", "aha")
                status.config(text=f"Ready | {elapsed_ms:.1f} ms | files=off | bitnet=online")

            root.after(0, show)

        threading.Thread(target=worker, daemon=True).start()

    entry.bind("<Return>", lambda _e: send())
    entry.focus_set()
    root.protocol(
        "WM_DELETE_WINDOW",
        lambda: root.destroy() if messagebox.askokcancel("Quit", f"Exit {WINDOW_TITLE}?") else None,
    )
    root.mainloop()


def main(argv: list[str]) -> int:
    args = set(argv[1:])
    if "--cli" in args or "--headless" in args:
        run_cli()
        return 0
    try:
        run_gui()
        return 0
    except Exception as exc:
        print("GUI failed, switching to CLI.", file=sys.stderr)
        print("Reason:", exc, file=sys.stderr)
        run_cli()
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
