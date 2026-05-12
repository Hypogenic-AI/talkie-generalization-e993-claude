"""Run a date-stratified MMLU subsample on a Talkie checkpoint.

We score the four ``A``/``B``/``C``/``D`` letter-choice continuations of a
standard MMLU prompt, picking the highest-log-likelihood letter as the
prediction. Items are tagged with our date-stratification heuristic
(``src.date_strata``); the eval reports per-stratum accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.date_strata import tag_mmlu_item
from src.datasets_io import load_mmlu, stratified_subsample_mmlu
from src.scoring import score_single_token_choices


_LETTERS = ["A", "B", "C", "D"]


def format_prompt(item) -> str:
    """Standard MMLU few-shot-zero prompt (no demonstrations)."""
    parts = [f"The following is a multiple choice question. "
             f"Reply with the single letter of the best answer."]
    parts.append("")
    parts.append(f"Question: {item.question}")
    for letter, choice in zip(_LETTERS, item.choices):
        parts.append(f"{letter}. {choice}")
    parts.append("Answer:")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_per_subject", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault(
        "HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf_cache")
    )
    from talkie import Talkie  # late import

    print(f"[mmlu] loading {args.model}…", flush=True)
    t0 = time.time()
    m = Talkie(args.model, cache_dir=os.path.join(os.environ["HF_HOME"], "hub"))
    print(f"[mmlu] loaded in {time.time()-t0:.0f}s", flush=True)

    items = load_mmlu()
    sub = stratified_subsample_mmlu(items, n_per_subject=args.n_per_subject,
                                    seed=args.seed)
    print(f"[mmlu] running {len(sub)} items "
          f"(stratified across {len(set(it.subject for it in sub))} subjects)",
          flush=True)

    records: list[dict] = []
    n_correct = 0
    t_start = time.time()
    for i, it in enumerate(sub):
        prompt = format_prompt(it)
        # Score each of " A", " B", " C", " D" via single-token next-token
        # log-probabilities (one forward pass per item, not four).
        try:
            sum_lps = score_single_token_choices(m, prompt, [f" {l}" for l in _LETTERS])
            pred_idx = max(range(4), key=lambda j: sum_lps[j])
            pred = _LETTERS[pred_idx]
            err = None
        except Exception as e:
            sum_lps = [None] * 4
            pred = None
            err = f"{type(e).__name__}: {e}"

        tag = tag_mmlu_item(it.question, it.choices)

        rec = {
            "subject": it.subject,
            "question": it.question,
            "choices": it.choices,
            "answer": it.answer,
            "predicted": pred,
            "correct": (pred == it.answer),
            "letter_logps": sum_lps,
            "stratum": tag.stratum,
            "stratum_triggers": tag.triggers[:5],
            "error": err,
        }
        records.append(rec)
        if rec["correct"]:
            n_correct += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"[mmlu {i+1:>4}/{len(sub)}] acc={n_correct/(i+1):.3f} "
                  f"rate={rate:.2f}/s eta={(len(sub)-i-1)/rate:.0f}s",
                  flush=True)

    # Aggregate per stratum and per subject
    def acc(filt):
        sub2 = [r for r in records if filt(r)]
        if not sub2:
            return {"n": 0, "acc": None}
        c = sum(1 for r in sub2 if r["correct"])
        return {"n": len(sub2), "acc": c / len(sub2)}

    summary = {
        "model": args.model,
        "n_per_subject": args.n_per_subject,
        "n_items": len(records),
        "n_correct": n_correct,
        "overall_acc": n_correct / len(records),
        "wall_seconds": time.time() - t_start,
        "stratum_post1930": acc(lambda r: r["stratum"] == "post1930"),
        "stratum_pre1930":  acc(lambda r: r["stratum"] == "pre1930"),
        "stratum_unknown":  acc(lambda r: r["stratum"] == "unknown"),
    }

    out = {"summary": summary, "records": records}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[mmlu] DONE — wrote {args.output}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
