# Research Plan — *Is it easier to test if Talkie generalizes?*

## Motivation & Novelty Assessment

### Why This Research Matters

The dominant methodological problem in modern LLM evaluation is *attribution*:
when a model answers a benchmark item correctly, did it **generalize** from
underlying competence, or did it **memorize** something close to the answer
during pretraining? Wang/Antoniades et al. (ICLR 2025) showed this matters
quantitatively — for factual tasks, modern LLM accuracy is largely explained
by n-gram co-occurrence frequencies in the pretraining corpus, not by
generalization. Cleanly attributing a result to generalization currently
requires expensive contamination analyses (n-gram indices over multi-TB
corpora) or building post-cutoff benchmarks (AntiLeak-Bench, evolveQA).
**Talkie-1930** (Levine/Duvenaud/Radford, April 2026) — a 13B LLM trained on
260B tokens of strictly pre-1931 English text — promises a *structural* shortcut:
its pretraining corpus has zero overlap with any post-1930 concept by
construction, so any non-trivial Talkie performance on a modern benchmark
*must* be generalization. Holtzman's hypothesis pushes this further: because
the *coverage* of human experience in Talkie's corpus is so much sparser, the
generalization signal-to-noise ratio is dramatically higher than for any
modern model — making generalization tests qualitatively easier.

### Gap in Existing Work

No published study has yet operationalised the meta-claim that Talkie's sparse
corpus *makes generalization easier to test*. The Talkie blog reports a few
flagship probes (HumanEval, NYT On-This-Day BpB curve, anachronism-filtered
MMLU); the IdeaHub submission proposes promoting this from one team's
methodology to a *general* evaluation methodology. What's missing is a
quantitative comparison of the *attribution clarity* one obtains from running
the same probes on Talkie vs. on modern LLMs. We supply that comparison here.

### Our Novel Contribution

We test the **meta-hypothesis** directly: by running matched probes on
`talkie-1930-13b-base`, `talkie-web-13b-base` (the matched modern-twin), and a
modern frontier LLM (GPT-4.1 via API), we quantify how much of Talkie's
behaviour can be cleanly attributed to generalization vs. memorization, and
contrast that with the indistinguishable case for modern LLMs. Three
contributions:

1. **Replicate-and-extend the Talkie team's HumanEval probe** with a tighter
   experimental protocol and a quantified attribution-clarity score.
2. **Run a 50-item knowledge-cutoff/ICL probe battery** that combines factual
   recall (post-1930 concepts) with in-context generalization (Python, JS),
   showing the asymmetry between pretrained knowledge and ICL transfer.
3. **Date-stratified MMLU** with explicit per-question dating, comparing the
   Talkie-1930 vs Talkie-web vs GPT-4.1 gap on pre-/post-1930-decidable
   subsets.
4. **Attribution-clarity meta-analysis**: a simple Bayesian framework giving
   bits-of-evidence per benchmark item for "Talkie-1930 is generalising,"
   compared with "GPT-4.1 is generalising," for the same items.

### Experiment Justification

| # | Experiment | Why needed |
|---|---|---|
| 1 | HumanEval pass@1 on `talkie-1930-13b-base`, `talkie-web-13b-base`, k∈{0,3} demos | Python ≫ 1930 cutoff. Any pass@1>0 on Talkie-1930 is unambiguous generalization. The matched twin controls for architecture/FLOPs. |
| 2 | 50-item post-1930 probe battery: factual completion + ICL Python/JS items | Measures the asymmetry between knowledge (Talkie should fail) and in-context generalization (Talkie may succeed). The expected modern completion vs. Talkie's actual log-likelihood distributions diagnose the cutoff cleanly. |
| 3 | Date-stratified MMLU subsample (~500 items) | Demonstrates that the date partition is a *useful* generalization probe for Talkie-1930 (Talkie-1930 ≪ Talkie-web on post-1930; gap ≈ 0 on pre-1930) and *useless* for modern LLMs (GPT-4.1 saturates both). Directly evidences "easier to test on Talkie." |
| 4 | Attribution-clarity meta-analysis | Quantifies the central claim. For each correct answer, computes bits of evidence ruling out memorization. Talkie-1930: hard-zero memorization probability for post-1930 items → maximal evidence per item. GPT-4.1: requires expensive contamination analysis to establish anything. |
| 5 | Modern-LLM API contrast (GPT-4.1) | Establishes the "we cannot distinguish memorization from generalization" baseline — necessary to compute the relative attribution-clarity gain Talkie provides. |

DO NOT proceed to implementation until this section is complete.  ✓ Done.

---

## Research Question

> *Is it easier to test whether **Talkie-1930** generalizes than to test whether
> a contemporaneous modern LLM (e.g., `talkie-web-13b-base` or GPT-4.1)
> generalizes?*

We decompose into three sub-questions:

- **Q1 (positive evidence).** Does Talkie-1930 produce non-trivial
  generalization signals on benchmarks built from post-1930 concepts?
- **Q2 (cleanliness of the partition).** Is the cutoff-induced partition
  (pre-1930-decidable vs. post-1930-decidable) actually a sharp boundary in
  practice?
- **Q3 (meta-claim).** For a fixed budget of probes, do we extract more bits
  of evidence about generalization from Talkie-1930 than from modern LLMs?

## Background and Motivation

See `literature_review.md` §1–§7 for the full picture. The two most relevant
results are:

- **Distributional memorisation** (Wang/Antoniades 2024): factual QA
  performance correlates with co-occurrence frequencies in pretraining data.
- **AbsenceBench** (Fu/Holtzman 2025): even strong models fail diagnostic
  probes that probe *absence* — diagnostic design matters more than headline
  numbers.

Talkie-1930's structural decontamination (1930 cutoff) lets us trade
expensive co-occurrence accounting for a free zero-by-construction lower
bound, *if* we can show the resulting probes are still informative. That is
the load-bearing experimental claim.

## Hypothesis Decomposition

| H | Statement | Direction predicted |
|---|---|---|
| H1 | Talkie-1930 achieves pass@1 > 0 on HumanEval (Python is post-1930). | true |
| H2 | Talkie-web pass@1 ≫ Talkie-1930 pass@1 on HumanEval. | true |
| H3 | Talkie-1930 surprisal is markedly higher than Talkie-web on post-1930 factual probes. | true |
| H4 | Talkie-1930 ICL probe success rate (with worked example) ≫ zero-shot factual probe success rate. | true (if generalization works) |
| H5 | On MMLU, Talkie-1930 ≪ Talkie-web on post-1930-decidable items; gap is much smaller on pre-1930-decidable items. | true |
| H6 | The Talkie-1930 vs. Talkie-web gap on post-1930 items is qualitatively larger than the GPT-4.1 vs. (any reasonable baseline) gap on the same items. | true (saturation hides modern memorization) |
| H7 | Per-item bits-of-evidence for "model is generalising" is higher for Talkie-1930 than for GPT-4.1 on the same items, by a quantifiable margin. | true (this is the meta-hypothesis) |

## Proposed Methodology

### Approach

We run three model-comparison studies (HumanEval / probes / MMLU) plus one
meta-analysis layered on top. All three studies use the same triplet of
models where possible:

- **Talkie-1930-13B-base** (vintage corpus, 1930 cutoff)
- **Talkie-web-13B-base** (matched modern-twin on FineWeb)
- **GPT-4.1** via OpenAI API (frontier modern LLM, sanity baseline)

### Experimental Steps

1. **Setup**: install `talkie`, download both Talkie checkpoints, build
   tokenizer for byte-exact log-prob scoring.
2. **Extend `model.py`** with a `forward_all_logits` helper so we can score
   completions in one forward pass rather than O(N²).
3. **HumanEval (Exp. 1)**: score pass@1 on all 164 problems with k∈{0,3}
   demonstrations. Also report per-problem detail.
4. **Post-1930 probes (Exp. 2)**: extend the 20-item seed to 50 items by
   scripted addition of factual + ICL items. Score expected-completion
   log-likelihood (sum over completion tokens) under each model. Also a
   "free-form generation" view (top-k decoded continuation).
5. **Date-stratified MMLU (Exp. 3)**: subsample 500 MMLU test items,
   automatically tag each with a "decidability date" via regex + simple
   heuristics, partition. Score 4-letter-likelihood for each.
6. **Modern-API contrast (Exp. 4)**: same Exp. 2 and Exp. 3 items run via
   OpenAI for GPT-4.1, scored by free-form output equality where possible.
7. **Attribution-clarity meta-analysis (Exp. 5)**: combine all results via a
   simple "bits of evidence" framework (see §Stat plan below).

### Baselines

- `talkie-web-13b-base` — matched-architecture modern twin.
- GPT-4.1 — modern API baseline; included for the meta-analysis only.
- Random-letter baseline for MMLU (1/4 = 25%).
- Greedy-prefix baseline for completion probes (always predict prompt-prefix
  next-token frequency from the corpus distribution; we approximate this by
  the unigram base-rate in our held-out test text).

### Evaluation Metrics

- **pass@1** (HumanEval) — standard, computed via the official test cases.
- **Mean log-likelihood per token** of expected completion (probes) — primary
  metric for the cutoff effect.
- **Top-1 exact match** of greedy completion vs. expected completion (probes)
  — interpretable surface metric.
- **Letter-choice accuracy** (MMLU).
- **Spearman ρ between log-likelihoods of {Talkie-1930, Talkie-web}** — does
  the model order items the same way? Indicates whether they share a
  generalization spine.
- **Δ(Talkie-1930, Talkie-web) split by date stratum** — primary cutoff
  diagnostic.
- **Bits-of-evidence per item** =
  log₂[P(Talkie correct | generalising) / P(Talkie correct | memorising)],
  with P(memorising) ≈ 0 by construction for post-1930 items, yielding a
  large positive number that bounds the strength of the generalization claim.

### Statistical Analysis Plan

- Bootstrap 1000-sample 95% CIs for all proportion estimates (pass@1, MMLU
  accuracy, exact-match).
- Wilcoxon signed-rank test for paired log-likelihood comparisons across
  models on the same items.
- Mann–Whitney U test for cross-stratum comparisons (pre-1930 vs.
  post-1930 MMLU items).
- All tests two-sided, α = 0.05, with Bonferroni correction across the small
  number of pre-registered tests (n_tests ≤ 8).

## Expected Outcomes

If the meta-hypothesis holds we should see:

- (E1) Talkie-1930 pass@1 > 0 but ≪ Talkie-web — clean generalization signal.
- (E2) Talkie-1930 log-likelihood on post-1930 factual completions is ≫
  Talkie-web (i.e., much worse). For ICL probes, the gap shrinks toward zero
  for items where the ICL exemplar suffices.
- (E3) On MMLU, the Δ(Talkie-1930, Talkie-web) is much larger on
  post-1930-decidable items. Evidence that the date partition is a clean
  diagnostic.
- (E4) GPT-4.1 saturates both date strata; the date partition tells us
  *nothing* about its memorization profile. Concretely, the
  bits-of-evidence-per-item from Talkie-1930 should exceed that of GPT-4.1 by
  many orders of magnitude (because GPT-4.1's training corpus is unknown and
  cannot be assumed to exclude any modern benchmark item).

If the hypothesis fails: Talkie-1930 either (a) matches Talkie-web on
post-1930 items (cutoff is leaky), or (b) achieves near-zero on every probe
(no generalization). Either failure would also be informative.

## Timeline and Milestones

(All times approximate, single continuous session)

| Phase | Time | What happens |
|---|---|---|
| 0 — Motivation | done | This document |
| 1 — Plan | done | This document |
| 2 — Setup & download | 15–25 min | `uv add` deps, two 53 GB checkpoints downloaded in parallel |
| 3 — Implement harness | 30–40 min | log-prob scorer, HumanEval runner, MMLU scorer, probe scorer, GPT-4.1 wrapper |
| 4 — Run experiments | 60–120 min | sequential across the two Talkie models + a small GPT-4.1 sweep |
| 5 — Analysis | 30 min | aggregate, plot, statistical tests |
| 6 — Documentation | 30 min | REPORT.md + README.md |

## Potential Challenges

1. **Download speed.** ~106 GB total over HuggingFace. Mitigation: download
   in parallel, log progress, gracefully exit if disk fills.
2. **Inference throughput.** 13B bf16 on a single A6000 ≈ 30–50 tok/s. With
   greedy decoding for HumanEval (avg ~150 tokens generated), 164 problems ≈
   12 minutes per model. Doable.
3. **MMLU scale.** 14K items × 4 choices × 2 models ≫ time budget. Mitigate
   by subsampling 500 items, stratified across the 57 subjects.
4. **Subject-matter confound.** Talkie-1930 ≪ Talkie-web on a post-1930 item
   could be due to subject mix, not only date. Mitigation: also stratify by
   subject; ICL probes that hold subject constant.
5. **Tokeniser asymmetry.** Talkie tokenizer was trained on pre-1930 text;
   modern terms tokenise to many tokens. Mitigation: report per-token NLL,
   not raw NLL; report token-count of completions.
6. **OCR noise** in pretraining corpus → may bias Talkie failure modes.
   Mitigation: report only differential effects (Δ across models), not
   absolute Talkie numbers.
7. **GPT-4.1 cost.** Cap to the 50 probes + 500 MMLU items = ≤ 600 calls
   ≈ $1–2 budget.

## Success Criteria

The research is **successful** if we can produce, end-to-end:

- ≥ 1 reproducible experiment showing Talkie-1930 > 0 generalization signal
  on a clearly post-1930 task (likely HumanEval).
- A quantified attribution-clarity number (bits/item) for both Talkie-1930
  and GPT-4.1, with the gap supported by statistical tests.
- A REPORT.md that explicitly answers "Is it easier?" with an evidence-based
  yes/no/qualified-yes.

The research is **partially successful** if we obtain only one of the above
(e.g., only HumanEval works because MMLU stratification is too noisy).

The research **fails** if Talkie-1930 either matches Talkie-web on post-1930
items (no cutoff effect) or achieves zero everywhere (no generalization at
all). Even in failure we will report what we observed.
