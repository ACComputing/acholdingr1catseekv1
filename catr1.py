#!/usr/bin/env python3
"""
CAT R1 entrypoint (r1.py)

Default: runs the full BitNet / R1-style / aha / `/pr` implementation in catseekr1.py
via runpy (single source of truth — no duplicated engine logic).

  python3 r1.py

Optional toy fallback (no numpy BitNet stack; GUI matches program4.22.26 layout):

  python3 r1.py --mini
  python3 r1.py --mini --cli

Full app uses tiktoken only if enabled (see catseekr1.py); optional:

  CATSEEK_USE_TIKTOKEN=1 python3 r1.py
"""

from __future__ import annotations

import os
import sys
import threading
import runpy
import faulthandler

faulthandler.enable()
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_CATSEEK_MAIN = os.path.join(_HERE, "catseekr1.py")

# Same branding as catseekr1.py / program4.22.26.py main window
WINDOW_TITLE = "AC HOLDINGS [C] 1999-2026 R1 Catseek"
BOT_NAME = "CAT R1.1.X"

MINI_PROFILE = """# CAT R1 (mini)

This is the lightweight `--mini` build (no BitNet numpy forward, no HTTP API).

Full BitNet R1.1.X UI: run `python3 r1.py` without `--mini` (loads catseekr1.py).

Weights: -1, 0, or 1 (ternary idea only in mini mode; not a trained model here).
"""


# ──────────────────────────────────────────────────────────────
# Tcl-safe helpers (mini GUI only)
# ──────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────
# SAFE ENGINE (minimal stable core)
# ──────────────────────────────────────────────────────────────
class SimpleEngine:
    def __init__(self) -> None:
        self.history: list[tuple[str, str]] = []
        self.last_aha = ""

    def generate(self, prompt: str) -> str:
        self.history.append(("user", prompt))
        p = (prompt or "").strip()
        pl = p.lower()
        self.last_aha = ""

        if pl in ("/pr", "/profile"):
            self.last_aha = ""
            return MINI_PROFILE
        if "hello" in pl or pl in ("hi", "hey"):
            return "Hi. Ready."
        if "why" in pl or "bug" in pl:
            self.last_aha = "Aha: separate what you observed from what you assumed—then test one link."
            return "Tell me one concrete symptom or error line to narrow it down."
        if "python" in pl:
            return """```python
def main():
    print("Hello World")

if __name__ == "__main__":
    main()
```"""
        return "Send a prompt. I will handle it."


# ──────────────────────────────────────────────────────────────
# CLI MODE (mini)
# ──────────────────────────────────────────────────────────────
def run_cli() -> None:
    engine = SimpleEngine()
    print("CAT R1 mini CLI. Type 'exit' to quit.\n")

    while True:
        try:
            msg = input(">>> ")
            if msg.strip().lower() == "exit":
                break
            out = engine.generate(msg)
            print(out)
            if engine.last_aha:
                print("Aha:", engine.last_aha)
        except (EOFError, KeyboardInterrupt):
            break


# ──────────────────────────────────────────────────────────────
# GUI MODE (mini) — layout aligned with program4.22.26.py CatR11GUI
# ──────────────────────────────────────────────────────────────
def run_gui() -> None:
    import tkinter as tk
    from tkinter import scrolledtext, font, messagebox

    engine = SimpleEngine()

    root = tk.Tk()
    root.title(WINDOW_TITLE)
    root.geometry("850x620")
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
    )
    chat.pack(expand=True, fill="both")

    for tag_name, color, fnt in [
        ("user", "#ffffff", fonts["bold"]),
        ("think", "#4a4a4a", fonts["italic"]),
        ("bot", "#00aaff", fonts["bold"]),
        ("code", "#00ffaa", fonts["small"]),
        ("aha", "#ffd54f", fonts["bold"]),
    ]:
        chat.tag_config(tag_name, foreground=color, font=fnt)

    inp = tk.Frame(root, bg="#050505")
    inp.pack(fill="x", padx=10, pady=5)
    entry = tk.Entry(inp, bg="#111", fg="#00d9ff", font=fonts["mono"], insertbackground="cyan", relief="flat", bd=2)
    entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
    btns = tk.Frame(inp, bg="#050505")
    btns.pack(side="right")
    for t, c in [("Help", "help"), ("Profile", "readme"), ("Py", "write python code")]:
        tk.Button(
            btns,
            text=t,
            command=lambda c=c: entry.insert("end", c + " "),
            bg="#222",
            fg="#00d9ff",
            font=fonts["small"],
            relief="flat",
        ).pack(side="left", padx=2)

    status = tk.Label(root, text="Ready", bg="#050505", fg="#666", font=fonts["small"], anchor="w")
    status.pack(fill="x", padx=10, pady=2)

    def log_line(sender: str, text: str, tag: str | None = None) -> None:
        body = _text_insert_safe(text if isinstance(text, str) else str(text), code_fence=(tag == "code"))
        head_tag = "bot" if sender == BOT_NAME else (tag if tag is not None else "think")
        body_tag = tag if tag is not None else ("bot" if sender == BOT_NAME else "think")
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
        if sender == "SYSTEM":
            status.config(text=body[:65])

    log_line("SYSTEM", f"{BOT_NAME} mini ONLINE (no API)")
    log_line("SYSTEM", "BitNet stack: use `python3 r1.py` without --mini for full catseekr1.py.")

    def send() -> None:
        msg = entry.get().strip()
        if not msg:
            return
        entry.delete(0, "end")
        log_line("YOU", msg, "user")
        status.config(text="Quantizing and routing...")

        def worker() -> None:
            try:
                resp = engine.generate(msg)
            except Exception as e:
                resp = f"(error) {type(e).__name__}: {e}"
                engine.last_aha = ""
            aha = engine.last_aha

            def show() -> None:
                if "```" in resp:
                    parts = resp.split("```")
                    for i, p in enumerate(parts):
                        chunk = p + ("```" if i < len(parts) - 1 and i % 2 == 0 else "")
                        log_line(BOT_NAME, chunk, "code" if i % 2 == 1 else None)
                else:
                    log_line(BOT_NAME, resp, None)
                if aha:
                    log_line("AHA", f"Aha: {aha}", "aha")
                status.config(text="Ready")

            root.after(0, show)

        threading.Thread(target=worker, daemon=True).start()

    entry.bind("<Return>", lambda e: send())
    entry.focus_set()
    root.protocol(
        "WM_DELETE_WINDOW",
        lambda: root.destroy() if messagebox.askokcancel("Quit", f"Exit {WINDOW_TITLE}?") else None,
    )
    root.mainloop()


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = set(sys.argv[1:])
    use_mini = "--mini" in args
    force_cli = "--cli" in args or "--headless" in args

    if use_mini:
        if force_cli:
            run_cli()
            sys.exit(0)
        try:
            run_gui()
        except Exception as e:
            print("GUI failed, switching to CLI.", file=sys.stderr)
            print("Reason:", e, file=sys.stderr)
            run_cli()
        sys.exit(0)

    if not os.path.isfile(_CATSEEK_MAIN):
        print(f"Missing {_CATSEEK_MAIN}; install catseekr1.py alongside r1.py or use --mini.", file=sys.stderr)
        sys.exit(1)

    runpy.run_path(_CATSEEK_MAIN, run_name="__main__")
