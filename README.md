# Is it easier to test if Talkie generalizes?

> *Holtzman's hypothesis (IdeaHub, May 2026): Talkie-1930 (Levine, Duvenaud,
> Radford, April 2026) was trained on data that covers human experience much
> more sparsely than current LLMs — does that make it easier to test that it
> generalizes?*

This repository contains an end-to-end automated investigation of that
hypothesis. The full write-up is in **[REPORT.md](REPORT.md)**.

## TL;DR

**Yes — qualitatively easier**, even though Talkie's absolute scores are
much lower than modern LLMs:

* Talkie-1930 attains 3% post-1930 factual recall (vs. 73% for the matched
  modern-architecture twin Talkie-web and 80% for GPT-4.1) on a 30-item
  probe set, with mean per-token surprisal **+6.84 bits/token** higher than
  Talkie-web (paired permutation p < 0.001, n=30).
* Talkie-1930 *also* produces non-trivial post-1930 generalization on its
  own: 4/11 ICL probes solved exactly (e.g., correctly continuing
  `def triple(x): return ` → `x * 3` and
  `cubes = [x**3 for x in range(5)]   # cubes is ` → `[0, 1, 8, 27, 64]`
  from a single Python worked example, despite never having seen Python
  in training).
* Per the attribution-bits framework: Talkie-1930 yields ≈ 34 bits of
  evidence per correct post-1930 answer (because P(memorisation) ≈ 0 by
  construction); Talkie-web and GPT-4.1 yield ≤ 0 bits each on the same
  items, until an expensive contamination audit of their training data is
  performed.

## Repository layout

```
.
├── REPORT.md                    ← primary deliverable
├── planning.md                  ← Phase-0 motivation + Phase-1 plan
├── literature_review.md         ← background reading (gathered up-front)
├── resources.md                 ← catalog of papers, datasets, code
├── papers/                      ← downloaded PDFs + 3-page chunks
├── datasets/
│   ├── HumanEval.jsonl
│   ├── mmlu/data/{test,val,dev}/*.csv
│   └── post1930_probes/{seed_probes.jsonl, probes.jsonl}
├── code/                        ← cloned baselines: talkie, lm-eval-harness, …
├── src/                         ← our experimental code
│   ├── scoring.py               ← log-prob + greedy decode helpers
│   ├── datasets_io.py
│   ├── date_strata.py           ← MMLU date-stratification heuristic
│   ├── build_probes.py          ← extends seed → 50-item probe set
│   ├── sandbox.py               ← HumanEval grading sandbox
│   ├── run_all_talkie.py        ← combined Talkie evaluator (probes + MMLU + HumanEval)
│   ├── run_openai_baseline.py   ← GPT-4.1 baselines for probes & MMLU
│   ├── run_humaneval_openai.py  ← GPT-4.1 HumanEval
│   └── analyze.py               ← aggregation, statistics, plots
├── results/                     ← per-experiment JSON outputs
└── figures/                     ← png plots referenced from REPORT.md
```

## Reproduce

```bash
uv venv && source .venv/bin/activate
uv add huggingface_hub torch tiktoken matplotlib requests hatchling
uv pip install -e ./code/talkie

export HF_HOME=$PWD/.hf_cache             # 53 GB per Talkie checkpoint
export OPENAI_API_KEY=...                 # for the baseline

python -m src.build_probes                # extends 20-item seed → 50

# Talkie evaluations on a 48 GB GPU each
python -m src.run_all_talkie --model talkie-1930-13b-base --tag 1930 --device cuda:0
python -m src.run_all_talkie --model talkie-web-13b-base  --tag web  --device cuda:1

# GPT-4.1 baselines
python -m src.run_openai_baseline probes --model gpt-4.1 \
    --output results/probes_gpt41.json
python -m src.run_openai_baseline mmlu --model gpt-4.1 \
    --n_per_subject 8 --output results/mmlu_gpt41.json
python -m src.run_humaneval_openai --model gpt-4.1 \
    --output results/humaneval_gpt41.json

# Aggregate + plot
python -m src.analyze
```

Total wall-clock on 4× RTX A6000: ~25 min for probes+MMLU on both Talkies,
~6 min for all GPT-4.1 baselines, ~30–60 min for HumanEval depending on
generation length and parallelism.

Total OpenAI cost: ≈ $1.

## Key figures

| Figure | What it shows |
|---|---|
| `figures/probes_em.png` | Probe EM rates by stratum (3 models × 3 strata) |
| `figures/probe_surprisal_by_year.png` | Per-token surprisal vs. concept year (1930 cutoff visible as a sharp jump for talkie-1930 only) |
| `figures/mmlu_strata.png` | MMLU accuracy by date stratum |
| `figures/humaneval_pass1.png` | HumanEval pass@1 with bootstrap CIs |
| `figures/attribution_bits.png` | Bits-of-evidence-for-generalization per benchmark item |

See [REPORT.md](REPORT.md) for the full discussion, statistical tests,
limitations, and recommended follow-ups.
