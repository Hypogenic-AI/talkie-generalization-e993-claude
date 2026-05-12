"""Aggregate all experiment outputs and produce figures + summary tables.

Reads the JSON outputs of run_humaneval, run_probes, run_mmlu (per model,
plus the optional GPT-4.1 baseline) and emits:

* ``results/aggregated.json`` — one tidy summary blob.
* ``figures/*.png`` — comparison plots.
* tables printed to stdout for the REPORT.

Bootstrap confidence intervals are computed for proportions (pass@1, EM rate,
MMLU accuracy) and statistical tests for paired comparisons across models.

The "attribution-clarity" meta-analysis (the main novel contribution) is
computed in :func:`compute_attribution_bits` — see docstring for the prior /
likelihood definitions.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGURES = REPO / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_proportion_ci(successes: int, n: int, n_resamples: int = 2000,
                            seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    if successes == n:
        return (alpha ** (1 / n), 1.0)
    if successes == 0:
        return (0.0, 1 - alpha ** (1 / n))
    rng = random.Random(seed)
    bools = [True] * successes + [False] * (n - successes)
    samples = []
    for _ in range(n_resamples):
        s = sum(rng.choices(bools, k=n))
        samples.append(s / n)
    samples.sort()
    lo = samples[int(alpha / 2 * n_resamples)]
    hi = samples[int((1 - alpha / 2) * n_resamples)]
    return (lo, hi)


def paired_perm_test(a: list[float], b: list[float], n_perm: int = 5000,
                     seed: int = 0) -> float:
    """Two-sided permutation test on paired differences."""
    assert len(a) == len(b)
    if not a:
        return float("nan")
    diffs = [x - y for x, y in zip(a, b)]
    obs = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_perm):
        signs = [rng.choice([-1, 1]) for _ in diffs]
        s = sum(d * sg for d, sg in zip(diffs, signs)) / len(diffs)
        if abs(s) >= abs(obs):
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def lenient_em(expected: str, generated: str) -> bool:
    """Lenient EM: case-fold, strip punctuation, collapse whitespace, substring."""
    def norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[\W_]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()
    e = norm(expected); g = norm(generated)
    return bool(e) and (e in g)


# ---------------------------------------------------------------------------
# Attribution-clarity meta-analysis
# ---------------------------------------------------------------------------

def attribution_bits(p_correct: float, p_memo: float,
                     p_correct_given_memo: float = 1.0,
                     prior_generalize: float = 0.5,
                     prior_memo: float = 0.5) -> float:
    """Bits of evidence in favour of "the model generalised" vs.
    "the model memorised", given that we observed ``p_correct``.

    Bayes factor:
        BF_gen = [P(correct | gen) * P(gen)]
                 / [P(correct | memo) * P(memo)]

    With ``p_memo`` the *prior* probability that the answer was in
    training (=0 by construction for Talkie-1930 on post-1930 items).

    For Talkie-1930 on post-1930 items we use ``p_memo ≈ 0`` → bits → ∞.
    For modern LLMs whose corpus is unknown we use ``p_memo ≈ 1`` → bits = 0
    (cannot rule out memorisation).
    """
    p_memo_eff = max(p_memo, 1e-12)
    num = max(p_correct, 1e-12) * prior_generalize
    den = p_correct_given_memo * p_memo_eff
    return math.log2(num / den)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _bar(ax, labels, vals, errs, title, ylabel, color="steelblue"):
    xs = list(range(len(labels)))
    ax.bar(xs, vals, yerr=errs, color=color, alpha=0.85, capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    out: dict[str, Any] = {}

    # ----------------------------------------------------------------- HumanEval
    he_files = {
        "talkie-1930 (k=0)": RESULTS / "humaneval_1930_k0.json",
        "talkie-1930 (k=3)": RESULTS / "humaneval_1930_k3.json",
        "talkie-web (k=0)":  RESULTS / "humaneval_web_k0.json",
        "talkie-web (k=3)":  RESULTS / "humaneval_web_k3.json",
        "gpt-4.1":           RESULTS / "humaneval_gpt41.json",
    }
    he = {}
    for label, p in he_files.items():
        d = load_json(p)
        if d is None: continue
        s = d["summary"]
        n, k = s["n_problems"], s["n_pass"]
        ci = bootstrap_proportion_ci(k, n)
        he[label] = {"n": n, "n_pass": k, "pass_at_1": k / n if n else None,
                     "ci_lo": ci[0], "ci_hi": ci[1]}
    out["humaneval"] = he

    # -------------------------------------------------------------------- Probes
    probe_files = {
        "talkie-1930":   RESULTS / "probes_1930.json",
        "talkie-web":    RESULTS / "probes_web.json",
        "gpt-4.1":       RESULTS / "probes_gpt41.json",
    }
    probe_summary = {}
    probe_records: dict[str, list[dict]] = {}
    for label, p in probe_files.items():
        d = load_json(p)
        if d is None: continue
        records = d.get("records", [])
        probe_records[label] = records

        def em_for(filt, lenient: bool = True) -> dict:
            sub = [r for r in records if filt(r)]
            if not sub: return {"n": 0}
            def hit(r):
                strict = r.get("greedy_contains_expected",
                               r.get("contains_expected", False))
                if strict or not lenient: return strict
                gen = r.get("greedy_gen") or r.get("completion") or ""
                return lenient_em(r["expected_modern"], gen)
            em = sum(1 for r in sub if hit(r))
            strict_em = sum(1 for r in sub
                            if r.get("greedy_contains_expected",
                                     r.get("contains_expected", False)))
            ci = bootstrap_proportion_ci(em, len(sub))
            return {"n": len(sub), "em": em, "em_rate": em / len(sub),
                    "strict_em": strict_em,
                    "strict_em_rate": strict_em / len(sub),
                    "ci_lo": ci[0], "ci_hi": ci[1]}

        # bits-per-token (only for Talkie checkpoints — not GPT-4.1)
        def bits_for(filt) -> dict:
            sub = [r for r in records if filt(r) and r.get("score")]
            if not sub: return {"n": 0}
            bits = [r["score"]["bits_per_token"] for r in sub]
            return {"n": len(sub),
                    "mean_bits": sum(bits) / len(bits),
                    "min_bits": min(bits), "max_bits": max(bits)}

        probe_summary[label] = {
            "post1930_completion": em_for(
                lambda r: r["year"] >= 1930 and r["kind"] == "completion"),
            "pre1930_completion":  em_for(
                lambda r: r["year"] < 1930 and r["kind"] == "completion"),
            "icl":                 em_for(lambda r: r["kind"] == "icl"),
            "bits_post1930_completion": bits_for(
                lambda r: r["year"] >= 1930 and r["kind"] == "completion"),
            "bits_pre1930_completion": bits_for(
                lambda r: r["year"] < 1930 and r["kind"] == "completion"),
            "bits_icl": bits_for(lambda r: r["kind"] == "icl"),
        }
    out["probes"] = probe_summary

    # ---------------------------------------------------------------------- MMLU
    mmlu_files = {
        "talkie-1930":   RESULTS / "mmlu_1930.json",
        "talkie-web":    RESULTS / "mmlu_web.json",
        "gpt-4.1":       RESULTS / "mmlu_gpt41.json",
    }
    mmlu = {}
    mmlu_records: dict[str, list[dict]] = {}
    for label, p in mmlu_files.items():
        d = load_json(p)
        if d is None: continue
        records = d.get("records", [])
        mmlu_records[label] = records

        def acc_for(filt) -> dict:
            sub = [r for r in records if filt(r)]
            if not sub: return {"n": 0}
            c = sum(1 for r in sub if r["correct"])
            ci = bootstrap_proportion_ci(c, len(sub))
            return {"n": len(sub), "n_correct": c, "acc": c / len(sub),
                    "ci_lo": ci[0], "ci_hi": ci[1]}

        mmlu[label] = {
            "overall":  acc_for(lambda r: True),
            "post1930": acc_for(lambda r: r["stratum"] == "post1930"),
            "pre1930":  acc_for(lambda r: r["stratum"] == "pre1930"),
            "unknown":  acc_for(lambda r: r["stratum"] == "unknown"),
        }
    out["mmlu"] = mmlu

    # ---------------------------------------------------------- Attribution bits
    bits = {}
    for label, s in he.items():
        if s.get("pass_at_1") is None:
            continue
        if "talkie-1930" in label:
            p_memo = 1e-12   # zero-by-construction: HumanEval is post-1930 (Python)
        else:
            p_memo = 1.0     # cannot rule out
        bits[label] = {
            "pass_at_1": s["pass_at_1"],
            "p_memo_assumed": p_memo,
            "bits_for_generalization": attribution_bits(
                s["pass_at_1"], p_memo=p_memo),
        }
    out["attribution_bits_humaneval"] = bits

    # Same for post-1930 probe completion items.
    bits_probes = {}
    for label, summ in probe_summary.items():
        rec = summ.get("post1930_completion", {})
        if rec.get("n", 0) == 0: continue
        p_correct = rec["em_rate"]
        p_memo = 1e-12 if label == "talkie-1930" else 1.0
        bits_probes[label] = {
            "em_rate": p_correct,
            "p_memo_assumed": p_memo,
            "bits_for_generalization": attribution_bits(p_correct, p_memo=p_memo),
        }
    out["attribution_bits_probes_post1930"] = bits_probes

    # ----------------------------------------------------------------- Plots
    if he:
        labels = list(he.keys())
        vals = [he[l]["pass_at_1"] for l in labels]
        errs_lo = [vals[i] - he[l]["ci_lo"] for i, l in enumerate(labels)]
        errs_hi = [he[l]["ci_hi"] - vals[i] for i, l in enumerate(labels)]
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        _bar(ax, labels, vals, [errs_lo, errs_hi],
             "HumanEval pass@1 (95% bootstrap CI)", "pass@1")
        ax.set_ylim(0, 1.0)
        fig.tight_layout(); fig.savefig(FIGURES / "humaneval_pass1.png", dpi=140)
        plt.close(fig)

    if mmlu:
        labels = list(mmlu.keys())
        strata = ["pre1930", "post1930", "unknown", "overall"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        width = 0.2
        xs = list(range(len(labels)))
        for i, st in enumerate(strata):
            vals = [mmlu[l][st].get("acc", 0) or 0 for l in labels]
            ax.bar([x + (i - 1.5) * width for x in xs], vals,
                   width=width, label=st, alpha=0.85)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=10)
        ax.set_ylabel("Accuracy")
        ax.set_title("MMLU accuracy by date stratum (subsample, 8/subject = 456 items)")
        ax.axhline(0.25, color="grey", linestyle="--", linewidth=1, alpha=0.7,
                   label="random")
        ax.legend(ncol=5, fontsize=8); ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(FIGURES / "mmlu_strata.png", dpi=140)
        plt.close(fig)

    if probe_summary:
        labels = list(probe_summary.keys())
        strata = ["pre1930_completion", "post1930_completion", "icl"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        width = 0.25; xs = list(range(len(labels)))
        for i, st in enumerate(strata):
            vals = [probe_summary[l][st].get("em_rate", 0) or 0 for l in labels]
            ax.bar([x + (i - 1) * width for x in xs], vals,
                   width=width, label=st, alpha=0.85)
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_ylabel("Lenient EM rate")
        ax.set_title("Post-1930 probe battery — exact-match rate by stratum")
        ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(FIGURES / "probes_em.png", dpi=140)
        plt.close(fig)

    # bits-per-token surprisal-by-year scatter
    if "talkie-1930" in probe_records and "talkie-web" in probe_records:
        rec_old = {r["id"]: r for r in probe_records["talkie-1930"]}
        rec_new = {r["id"]: r for r in probe_records["talkie-web"]}
        ids = sorted(rec_old.keys() & rec_new.keys())
        xs, ys_o, ys_w, kinds = [], [], [], []
        for i in ids:
            ro, rw = rec_old[i], rec_new[i]
            if ro.get("score") and rw.get("score"):
                xs.append(ro["year"])
                ys_o.append(ro["score"]["bits_per_token"])
                ys_w.append(rw["score"]["bits_per_token"])
                kinds.append(ro["kind"])
        if xs:
            fig, ax = plt.subplots(figsize=(9, 4.8))
            for k, color, marker in [("completion", "tab:red", "o"),
                                     ("icl", "tab:blue", "s")]:
                xs_k = [x for x, kk in zip(xs, kinds) if kk == k]
                yo_k = [y for y, kk in zip(ys_o, kinds) if kk == k]
                yw_k = [y for y, kk in zip(ys_w, kinds) if kk == k]
                ax.scatter(xs_k, yo_k, c=color, marker=marker, s=55,
                           alpha=0.85, label=f"talkie-1930 ({k})")
                ax.scatter(xs_k, yw_k, c=color, marker=marker, s=55,
                           alpha=0.4, edgecolors="black", linewidths=0.5,
                           label=f"talkie-web ({k})")
            ax.axvline(1930, color="grey", linestyle="--",
                       label="cutoff 1930", alpha=0.7)
            ax.set_xlabel("Year of expected concept")
            ax.set_ylabel("bits / token of expected completion")
            ax.set_title("Probe surprisal by year — lower = better; "
                         "post-1930 gap is the cutoff signature")
            ax.legend(fontsize=8, loc="upper left", ncol=2)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(FIGURES / "probe_surprisal_by_year.png", dpi=140)
            plt.close(fig)

    # Attribution-bits comparison plot — based on the *probe* battery,
    # which is the cleaner signal than HumanEval (both Talkies fail at
    # HumanEval due to base-model alignment, see §4.3 of REPORT.md).
    if bits_probes:
        labels = list(bits_probes.keys())
        vals = [bits_probes[l]["bits_for_generalization"] for l in labels]
        # Cap "infinity" for plotting
        capped = [min(v, 60) for v in vals]
        colors = ["steelblue" if "talkie-1930" in l else "lightgrey"
                  for l in labels]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(range(len(labels)), capped, color=colors, alpha=0.85)
        for i, v in enumerate(vals):
            em = bits_probes[labels[i]]["em_rate"]
            txt = (f"≈∞\n(em={em:.0%})" if v > 60
                   else f"{v:+.1f} bits\n(em={em:.0%})")
            ax.text(i, max(capped[i] + 2, 5), txt, ha="center", fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("bits of evidence for generalization (per correct item)")
        ax.set_title("Attribution clarity on the post-1930 probe battery\n"
                     "Talkie-1930: P(memo)≈0 by construction → +34 bits/item.\n"
                     "Modern LLMs: P(memo) unknown → ≤ 0 bits without "
                     "contamination audit.")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_ylim(min(capped) - 5, max(50, max(capped) + 10))
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(FIGURES / "attribution_bits.png", dpi=140)
        plt.close(fig)

    # ---------------------------------------------------------- Statistical tests
    out["tests"] = {}
    if "talkie-1930" in probe_records and "talkie-web" in probe_records:
        rec_old = {r["id"]: r for r in probe_records["talkie-1930"]}
        rec_new = {r["id"]: r for r in probe_records["talkie-web"]}
        post_ids = [i for i, r in rec_old.items()
                    if r["year"] >= 1930 and r["kind"] == "completion"]
        a, b = [], []
        for i in post_ids:
            if (rec_old[i].get("score") is None or
                rec_new.get(i, {}).get("score") is None):
                continue
            a.append(rec_old[i]["score"]["bits_per_token"])
            b.append(rec_new[i]["score"]["bits_per_token"])
        if a:
            p = paired_perm_test(a, b)
            out["tests"]["probes_post1930_bits_perm_p"] = p
            out["tests"]["probes_post1930_n_paired"] = len(a)
            out["tests"]["probes_post1930_mean_diff_bits_old_minus_new"] = (
                sum(a) / len(a) - sum(b) / len(b))

        # ICL-paired test too
        icl_ids = [i for i, r in rec_old.items() if r["kind"] == "icl"]
        a, b = [], []
        for i in icl_ids:
            if (rec_old[i].get("score") is None or
                rec_new.get(i, {}).get("score") is None):
                continue
            a.append(rec_old[i]["score"]["bits_per_token"])
            b.append(rec_new[i]["score"]["bits_per_token"])
        if a:
            p = paired_perm_test(a, b)
            out["tests"]["probes_icl_bits_perm_p"] = p
            out["tests"]["probes_icl_n_paired"] = len(a)
            out["tests"]["probes_icl_mean_diff_bits_old_minus_new"] = (
                sum(a) / len(a) - sum(b) / len(b))

    # ----------------------------------------------------------------------- save
    out_path = RESULTS / "aggregated.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Wrote {out_path}")
    print(f"Wrote figures to {FIGURES}/")
    # Print a tidy summary table.
    print("\n========== SUMMARY ==========\n")
    if he:
        print("HumanEval pass@1:")
        for l, s in he.items():
            print(f"  {l:25s}  {s['pass_at_1']:.3f}  "
                  f"({s['n_pass']}/{s['n']})  "
                  f"95% CI [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]")
    if probe_summary:
        print("\nProbe EM rates (lenient):")
        for l, s in probe_summary.items():
            for k in ["pre1930_completion", "post1930_completion", "icl"]:
                d = s.get(k, {})
                if d.get("n", 0) == 0: continue
                print(f"  {l:14s} {k:25s} {d['em_rate']:.3f} ({d['em']}/{d['n']})  "
                      f"strict={d['strict_em']}/{d['n']}")
    if mmlu:
        print("\nMMLU accuracy by stratum:")
        for l, s in mmlu.items():
            for k in ["overall", "post1930", "pre1930", "unknown"]:
                d = s.get(k, {})
                if d.get("n", 0) == 0: continue
                print(f"  {l:14s} {k:10s} {d['acc']:.3f} ({d['n_correct']}/{d['n']})  "
                      f"95% CI [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]")
    if bits:
        print("\nAttribution bits on HumanEval:")
        for l, b in bits.items():
            v = b["bits_for_generalization"]
            label = "≈∞ (zero memorization by construction)" if v > 60 else f"{v:.2f} bits"
            print(f"  {l:25s}  pass@1={b['pass_at_1']:.3f}  "
                  f"P(memo)={b['p_memo_assumed']:.0e}  -> {label}")


if __name__ == "__main__":
    main()
