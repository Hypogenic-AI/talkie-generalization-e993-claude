"""Run probes + MMLU + HumanEval on a Talkie checkpoint, in that order.

Loading a 13B model takes ~2 minutes; doing all three evals back-to-back in
one process amortises the cost.

Usage::

    python -m src.run_all_talkie --model talkie-1930-13b-base \\
        --tag 1930 --device cuda:0 \\
        --mmlu_n_per_subject 8 --humaneval_n 164 --humaneval_kshot 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datasets_io import (
    load_humaneval, load_probes, load_mmlu, stratified_subsample_mmlu,
)
from src.date_strata import tag_mmlu_item
from src.sandbox import grade_humaneval
from src.scoring import (
    score_completion, greedy_complete, score_single_token_choices,
)


_LETTERS = ["A", "B", "C", "D"]


# --------------------------------------------------------------------- probes
def run_probes(m, tag: str, output: Path):
    probes = load_probes()
    print(f"\n[probes] {len(probes)} items on tag={tag}…", flush=True)
    records = []
    t0 = time.time()
    for i, p in enumerate(probes):
        target = p.expected_modern
        if not p.prompt.endswith((" ", "\n", "\t", '"', "'", "(", "{", "[")):
            target = " " + target
        try:
            sc = score_completion(m, p.prompt, target)
            score_dict = {"sum_logp": sc.sum_logp, "n_tokens": sc.n_tokens,
                          "mean_logp": sc.mean_logp,
                          "bits_per_token": sc.bits_per_token}
        except Exception as e:
            score_dict = None
        try:
            gen = greedy_complete(m, p.prompt, max_tokens=32, stop_strings=("\n",))
        except Exception:
            gen = ""
        em = p.expected_modern.strip().lower() in gen.strip().lower()
        records.append({
            "id": p.id, "year": p.year, "category": p.category,
            "kind": p.kind, "concept": p.concept, "prompt": p.prompt,
            "expected_modern": p.expected_modern, "scored_target": target,
            "score": score_dict, "greedy_gen": gen,
            "greedy_contains_expected": em,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(probes)}] {time.time()-t0:.0f}s", flush=True)

    summary = {"model": tag, "n_probes": len(records),
               "wall_seconds": time.time() - t0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"[probes] wrote {output}", flush=True)


# ----------------------------------------------------------------------- mmlu
def _mmlu_prompt(item) -> str:
    parts = ["The following is a multiple choice question. "
             "Reply with the single letter of the best answer.", ""]
    parts.append(f"Question: {item.question}")
    for L, c in zip(_LETTERS, item.choices):
        parts.append(f"{L}. {c}")
    parts.append("Answer:")
    return "\n".join(parts)


def run_mmlu(m, tag: str, output: Path, n_per_subject: int, seed: int = 0):
    items = load_mmlu()
    sub = stratified_subsample_mmlu(items, n_per_subject, seed=seed)
    print(f"\n[mmlu] {len(sub)} items on tag={tag}…", flush=True)
    records = []
    n_correct = 0
    t0 = time.time()
    for i, it in enumerate(sub):
        prompt = _mmlu_prompt(it)
        try:
            lps = score_single_token_choices(m, prompt, [f" {L}" for L in _LETTERS])
            pred_idx = max(range(4), key=lambda j: lps[j])
            pred = _LETTERS[pred_idx]
        except Exception as e:
            lps = [None] * 4
            pred = None
        tag_d = tag_mmlu_item(it.question, it.choices)
        rec = {"subject": it.subject, "question": it.question,
               "choices": it.choices, "answer": it.answer,
               "predicted": pred, "correct": (pred == it.answer),
               "letter_logps": lps, "stratum": tag_d.stratum,
               "stratum_triggers": tag_d.triggers[:5]}
        records.append(rec)
        if rec["correct"]: n_correct += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(sub)}] acc={n_correct/(i+1):.3f} "
                  f"{time.time()-t0:.0f}s", flush=True)

    summary = {"model": tag, "n_items": len(records), "n_correct": n_correct,
               "overall_acc": n_correct / len(records),
               "wall_seconds": time.time() - t0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"[mmlu] DONE acc={summary['overall_acc']:.3f} -> {output}", flush=True)


# ------------------------------------------------------------------- humaneval
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


def truncate_completion(completion: str) -> str:
    lines = completion.splitlines(keepends=True)
    out = []
    for i, line in enumerate(lines):
        s = line.rstrip("\r\n")
        if i == 0:
            out.append(line); continue
        if s.startswith(("def ", "class ", "if __name__")):
            break
        out.append(line)
    return "".join(out)


def run_humaneval(m, tag: str, output: Path, n_problems: int = 164,
                  k_demos: int = 0, max_tokens: int = 384):
    items = load_humaneval()[:n_problems]
    print(f"\n[humaneval] {len(items)} items, k_demos={k_demos}, tag={tag}…",
          flush=True)
    records = []
    n_pass = 0
    t0 = time.time()
    incremental_path = output.parent / (output.stem + "_progress.jsonl")
    incremental_path.parent.mkdir(parents=True, exist_ok=True)
    inc_f = open(incremental_path, "w", buffering=1)
    for i, it in enumerate(items):
        full_prompt = (_DEMO_PREFIX + it.prompt) if k_demos > 0 else it.prompt
        try:
            raw = greedy_complete(m, full_prompt, max_tokens=max_tokens,
                                  stop_strings=("\ndef ", "\nclass ",
                                                "\nif __name__",
                                                "\nimport ",
                                                "\nfrom ",
                                                "\n#", "\n\n\n"))
        except Exception as e:
            raw = ""
        completion = truncate_completion(raw)
        grade = grade_humaneval(it.prompt, completion, it.test, it.entry_point)
        rec = {"task_id": it.task_id, "entry_point": it.entry_point,
               "completion": completion, "raw_completion": raw,
               "passed": grade.passed, "grade_reason": grade.reason}
        records.append(rec)
        inc_f.write(json.dumps(rec) + "\n")
        if grade.passed: n_pass += 1
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(items)}] pass={n_pass} "
                  f"({n_pass/(i+1):.1%}) {elapsed:.0f}s "
                  f"eta={(len(items)-i-1)/rate:.0f}s", flush=True)

    inc_f.close()
    summary = {"model": tag, "k_demos": k_demos, "n_problems": len(items),
               "n_pass": n_pass,
               "pass_at_1": n_pass / len(items),
               "wall_seconds": time.time() - t0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"[humaneval] DONE pass@1={summary['pass_at_1']:.3f} -> {output}",
          flush=True)


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True,
                    help="Short tag used in output filenames (e.g., '1930').")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mmlu_n_per_subject", type=int, default=8)
    ap.add_argument("--humaneval_n", type=int, default=164)
    ap.add_argument("--humaneval_kshot", type=int, default=0)
    ap.add_argument("--humaneval_max_tokens", type=int, default=384)
    ap.add_argument("--skip_humaneval", action="store_true")
    ap.add_argument("--skip_mmlu", action="store_true")
    ap.add_argument("--skip_probes", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(REPO / ".hf_cache"))
    cache_dir = os.path.join(os.environ["HF_HOME"], "hub")

    from talkie import Talkie

    print(f"\n=== loading {args.model} on {args.device} ===", flush=True)
    t0 = time.time()
    m = Talkie(args.model, device=args.device, cache_dir=cache_dir)
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    results_dir = REPO / "results"

    if not args.skip_probes:
        run_probes(m, args.tag, results_dir / f"probes_{args.tag}.json")
    if not args.skip_mmlu:
        run_mmlu(m, args.tag, results_dir / f"mmlu_{args.tag}.json",
                 n_per_subject=args.mmlu_n_per_subject)
    if not args.skip_humaneval:
        ksuf = f"_k{args.humaneval_kshot}"
        run_humaneval(m, args.tag,
                      results_dir / f"humaneval_{args.tag}{ksuf}.json",
                      n_problems=args.humaneval_n,
                      k_demos=args.humaneval_kshot,
                      max_tokens=args.humaneval_max_tokens)


if __name__ == "__main__":
    main()
