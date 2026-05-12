"""Run the 50-item post-1930 probe battery on a Talkie checkpoint.

For each probe we record:
- expected-completion log-likelihood (sum and per-token), bits/token
- greedy decoded continuation (first ~32 tokens)
- whether the greedy continuation contains the expected_modern string
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets_io import load_probes
from src.scoring import score_completion, greedy_complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_gen_tokens", type=int, default=32)
    args = ap.parse_args()

    os.environ.setdefault(
        "HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf_cache")
    )
    from talkie import Talkie  # late import

    print(f"[probes] loading {args.model}…", flush=True)
    t0 = time.time()
    m = Talkie(args.model, cache_dir=os.path.join(os.environ["HF_HOME"], "hub"))
    print(f"[probes] loaded in {time.time()-t0:.0f}s", flush=True)

    probes = load_probes()
    print(f"[probes] running {len(probes)} probes", flush=True)

    records: list[dict] = []
    t_start = time.time()
    for i, p in enumerate(probes):
        # Score the expected modern completion. For ICL items, the prompt
        # already contains worked examples; we still score expected_modern.
        # For completion-style probes, expected_modern is the canonical answer.
        # We add a leading space if the prompt does not end with whitespace,
        # to mimic natural continuation.
        target = p.expected_modern
        if not p.prompt.endswith((" ", "\n", "\t", '"', "'", "(", "{", "[")):
            target = " " + target

        try:
            sc = score_completion(m, p.prompt, target)
            score_err = None
        except Exception as e:
            sc = None
            score_err = f"{type(e).__name__}: {e}"

        try:
            gen = greedy_complete(m, p.prompt, max_tokens=args.max_gen_tokens,
                                  stop_strings=("\n",))
            gen_err = None
        except Exception as e:
            gen = ""
            gen_err = f"{type(e).__name__}: {e}"

        # Surface metric: did the greedy continuation contain expected_modern?
        em_norm = p.expected_modern.strip().lower()
        gen_norm = gen.strip().lower()
        contains_expected = em_norm in gen_norm

        rec = {
            "id": p.id, "year": p.year, "category": p.category,
            "kind": p.kind, "concept": p.concept, "prompt": p.prompt,
            "expected_modern": p.expected_modern,
            "scored_target": target,
            "score": None if sc is None else {
                "sum_logp": sc.sum_logp,
                "n_tokens": sc.n_tokens,
                "mean_logp": sc.mean_logp,
                "bits_per_token": sc.bits_per_token,
            },
            "score_error": score_err,
            "greedy_gen": gen,
            "gen_error": gen_err,
            "greedy_contains_expected": contains_expected,
        }
        records.append(rec)

        if (i + 1) % 10 == 0:
            print(f"[probes {i+1:>3}/{len(probes)}] elapsed={time.time()-t_start:.0f}s", flush=True)

    # Aggregate by stratum
    def agg(filt):
        scored = [r for r in records if filt(r) and r["score"]]
        if not scored:
            return {"n": 0, "mean_bits": None, "n_em": 0}
        bits = [r["score"]["bits_per_token"] for r in scored]
        em = sum(1 for r in records if filt(r) and r["greedy_contains_expected"])
        n_total = sum(1 for r in records if filt(r))
        return {"n": n_total, "n_scored": len(scored),
                "mean_bits_per_token": sum(bits) / len(bits),
                "n_em": em}

    summary = {
        "model": args.model,
        "n_probes": len(records),
        "wall_seconds": time.time() - t_start,
        "agg_post1930": agg(lambda r: r["year"] >= 1930 and r["kind"] == "completion"),
        "agg_pre1930":  agg(lambda r: r["year"] < 1930 and r["kind"] == "completion"),
        "agg_icl":      agg(lambda r: r["kind"] == "icl"),
    }

    out = {"summary": summary, "records": records}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[probes] DONE — wrote {args.output}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
