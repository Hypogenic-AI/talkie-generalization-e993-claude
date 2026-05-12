"""Run HumanEval pass@1 on a Talkie checkpoint.

Usage::

    HF_HOME=.hf_cache python -m src.run_humaneval \\
        --model talkie-1930-13b-base --output results/humaneval_1930.json

Greedy decoding (deterministic), with optional in-context demonstrations of
Python syntax. The model is loaded once, the 164 problems are scored
sequentially, and we save per-problem records plus aggregate pass@1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets_io import load_humaneval
from src.sandbox import grade_humaneval
from src.scoring import greedy_complete


# ---------------------------------------------------------------------------
# In-context-demo prefix for k>0 conditioning. Two trivial worked examples
# carefully chosen to be representative Python without leaking HumanEval
# patterns. The demos teach: function definition, ``return``, conditional,
# basic arithmetic, list iteration.
# ---------------------------------------------------------------------------

_DEMO_PREFIX = '''# Python is a programming language. Functions are written with `def`,
# return values with `return`, and indented blocks form the body.

def double(x):
    """Return x doubled."""
    return x * 2

def is_positive(n):
    """Return True if n is greater than zero, else False."""
    if n > 0:
        return True
    else:
        return False

def sum_list(xs):
    """Return the sum of a list of numbers."""
    total = 0
    for x in xs:
        total = total + x
    return total

'''


def build_prompt(item_prompt: str, k_demos: int) -> str:
    if k_demos == 0:
        return item_prompt
    return _DEMO_PREFIX + item_prompt


def truncate_completion(completion: str) -> str:
    """Cut completion at the first line that starts a NEW top-level def
    or class (i.e., the model started writing a new function, not finishing
    the current one). Standard HumanEval-style truncation."""
    lines = completion.splitlines(keepends=True)
    out: list[str] = []
    for i, line in enumerate(lines):
        stripped_left = line.rstrip("\r\n")
        # Skip the first line (continuation of the docstring or function body).
        if i == 0:
            out.append(line)
            continue
        if stripped_left.startswith(("def ", "class ", "if __name__")):
            break
        out.append(line)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Talkie model name (e.g., talkie-1930-13b-base).")
    ap.add_argument("--output", required=True, help="Output JSON path.")
    ap.add_argument("--k_demos", type=int, default=0,
                    help="Number of in-context Python demonstrations.")
    ap.add_argument("--max_tokens", type=int, default=384)
    ap.add_argument("--n_problems", type=int, default=164,
                    help="Subset for debugging (default: full 164).")
    args = ap.parse_args()

    os.environ.setdefault(
        "HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf_cache")
    )

    from talkie import Talkie  # late import after env

    print(f"[run_humaneval] loading {args.model}…", flush=True)
    t0 = time.time()
    m = Talkie(args.model, cache_dir=os.path.join(os.environ["HF_HOME"], "hub"))
    print(f"[run_humaneval] loaded in {time.time()-t0:.0f}s", flush=True)

    items = load_humaneval()[: args.n_problems]
    print(f"[run_humaneval] running {len(items)} problems, k_demos={args.k_demos}", flush=True)

    records: list[dict] = []
    n_pass = 0
    t_start = time.time()
    for i, it in enumerate(items):
        full_prompt = build_prompt(it.prompt, args.k_demos)
        try:
            raw = greedy_complete(
                m, full_prompt,
                max_tokens=args.max_tokens,
                stop_strings=("\nclass ", "\nif __name__", "\n#", "\n\n\n"),
            )
        except Exception as e:
            raw = ""
            err = f"{type(e).__name__}: {e}"
        else:
            err = None
        completion = truncate_completion(raw)
        grade = grade_humaneval(it.prompt, completion, it.test, it.entry_point)
        rec = {
            "task_id": it.task_id,
            "entry_point": it.entry_point,
            "completion": completion,
            "raw_completion": raw,
            "passed": grade.passed,
            "grade_reason": grade.reason,
            "gen_error": err,
        }
        records.append(rec)
        if grade.passed:
            n_pass += 1
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(items) - i - 1) / rate if rate > 0 else 0
            print(f"[{i+1:>3}/{len(items)}] pass={n_pass} ({n_pass/(i+1):.1%}) "
                  f"rate={rate:.2f}/s eta={eta:.0f}s", flush=True)

    pass_at_1 = n_pass / len(items) if items else 0.0
    summary = {
        "model": args.model,
        "k_demos": args.k_demos,
        "n_problems": len(items),
        "n_pass": n_pass,
        "pass_at_1": pass_at_1,
        "max_tokens": args.max_tokens,
        "wall_seconds": time.time() - t_start,
    }
    out = {"summary": summary, "records": records}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[run_humaneval] DONE pass@1 = {pass_at_1:.3f} ({n_pass}/{len(items)})", flush=True)
    print(f"[run_humaneval] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
