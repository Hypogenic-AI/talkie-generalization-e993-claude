"""HumanEval pass@1 via OpenAI Chat Completions (gpt-4.1 default).

We prompt for code completion and grade with the same sandbox used for
Talkie. Used for the meta-attribution comparison; HumanEval is well-known
to be in modern training data (the original Codex paper itself trained on
GitHub Python), so for the meta-analysis the GPT-4.1 score is treated as
``unknown contamination'' and excluded from "uncontested generalization
claims" attribution unless explicitly justified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datasets_io import load_humaneval
from src.sandbox import grade_humaneval


_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _call(messages, model, max_tokens=512):
    payload = {"model": model, "messages": messages,
               "max_completion_tokens": max_tokens}
    if not model.startswith("gpt-5"):
        payload["temperature"] = 0.0
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
               "Content-Type": "application/json"}
    for attempt in range(4):
        r = requests.post(_OPENAI_URL, headers=headers, json=payload, timeout=90)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError("retries exhausted")


def extract_code(content: str) -> str:
    """If response is wrapped in ```python … ```, peel it. Else return as-is."""
    if "```" in content:
        # Find first code fence start, prefer python; then end fence
        import re
        m = re.search(r"```(?:python)?\n(.*?)```", content, re.DOTALL)
        if m:
            return m.group(1)
    return content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=164)
    args = ap.parse_args()
    items = load_humaneval()[: args.n]
    print(f"[he-openai] running {len(items)} on {args.model}", flush=True)

    records = []
    n_pass = 0
    t0 = time.time()
    for i, it in enumerate(items):
        sys_msg = ("You are a Python coding assistant. Complete the function "
                   "given by the user. Reply with ONLY the function body "
                   "(no markdown fences, no docstring repeat, no top-level "
                   "code, no `def` line) — i.e., just the indented body.")
        msg = [{"role": "system", "content": sys_msg},
               {"role": "user", "content": it.prompt}]
        try:
            resp = _call(msg, args.model)
            content = resp["choices"][0]["message"]["content"]
            completion = extract_code(content)
            # If the model returned a full def, strip the signature.
            stripped = completion.strip()
            if stripped.startswith("def "):
                # Keep everything after the first ":" + newline
                lines = completion.splitlines(keepends=True)
                body_start = 0
                for j, l in enumerate(lines):
                    if l.rstrip().endswith(":") and l.strip().startswith("def "):
                        body_start = j + 1
                        break
                completion = "".join(lines[body_start:])
        except Exception as e:
            completion = ""
            content = ""
            err = f"{type(e).__name__}: {e}"
        else:
            err = None

        # Re-indent if the model didn't use 4-space indent.
        # HumanEval prompts end mid-function-body; the completion should
        # already be indented one level. If not, indent every line.
        if completion and not completion.lstrip("\n").startswith((" ", "\t")):
            completion = "\n".join("    " + l if l.strip() else l
                                   for l in completion.splitlines()) + "\n"

        grade = grade_humaneval(it.prompt, completion, it.test, it.entry_point)
        rec = {"task_id": it.task_id, "entry_point": it.entry_point,
               "completion": completion, "raw_completion": content,
               "passed": grade.passed, "grade_reason": grade.reason,
               "error": err}
        records.append(rec)
        if grade.passed:
            n_pass += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(items)}] pass={n_pass} "
                  f"({n_pass/(i+1):.1%}) {time.time()-t0:.0f}s", flush=True)

    summary = {"model": args.model, "n_problems": len(items),
               "n_pass": n_pass, "pass_at_1": n_pass / len(items),
               "wall_seconds": time.time() - t0}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"[he-openai] DONE pass@1={summary['pass_at_1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
