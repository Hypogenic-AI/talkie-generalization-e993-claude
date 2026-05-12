# Literature Review

**Project:** *Is it easier to test if Talkie generalizes?*
**Hypothesis:** Because Talkie (the 1930 cutoff LLM) was trained on data that covers human experience much more sparsely than current LLMs, it may be easier to test whether it generalizes.

---

## 1. Research-area overview

The question of whether LLMs *genuinely generalize* or *merely retrieve from training data* is the central methodological problem of modern LLM evaluation. Three converging lines of evidence frame the field:

1. **Memorisation dominates knowledge-intensive tasks.** Wang & Antoniades et al. (ICLR 2025) — *Generalization v.s. Memorization* — show that for Pythia models on The Pile, factual QA performance is largely explained by the frequency of input/output n-gram co-occurrences in pretraining data, while translation and math are not. Memorisation isn't an aberration; it's the *default* explanation that any generalisation claim must rule out.
2. **Contamination is pervasive.** Lee et al. 2021 (*Deduplicating Training Data*) and the AntiLeak-Bench line of work (Wu et al. 2024) demonstrate that train/test overlap silently inflates benchmark scores at large scale, and that contamination-free benchmarks must be built deliberately — typically by exploiting the train-cutoff/test-creation date gap.
3. **Out-of-distribution evaluation is brittle.** Zhang et al. 2024 (*On the OOD Generalization of MLLMs*), McCoy/Pavlick/Linzen 2019 (HANS), Hendrycks et al. 2020 (*Pretrained Transformers Improve OOD Robustness*), and Yang et al. 2022 (GLUE-X) all establish that observed OOD performance reflects a tangle of spurious correlations, training-set composition, and benchmark construction artifacts. There is no universally accepted protocol for cleanly attributing a result to "generalisation."

**Talkie-1930** (Levine, Duvenaud, Radford, released ~April 2026) is a 13B-parameter decoder-only LLM trained on 260 B tokens of strictly pre-1931 English text — books, newspapers, periodicals, scientific journals, patents, case law. Its sibling `talkie-web-13b-base` is the same architecture and FLOPs trained on modern FineWeb. Together they form a controlled experiment: same model class, same compute, *very different* training distributions.

The Talkie blog post explicitly frames the contribution as "contamination-free by construction, enabling unique generalization experiments." Holtzman's hypothesis — submitted to the IdeaHub one week ago — pushes one step further: because the *coverage* of human experience in Talkie's corpus is so much sparser than in modern web data, the generalisation signal-to-noise ratio is dramatically better. Most modern tasks land in 100% OOD territory, leaving no room for the usual confounds.

The only existing analogue is **Ranke-4B** (Göttlich/Loibner/Jiang/Voth, UZH/Cologne, pre-release Dec 2025): a 4 B-parameter Qwen3-architecture family with cutoffs at 1913 / 1929 / 1933 / 1939 / 1946. Ranke offers cross-cutoff *gradients* that Talkie alone cannot, and is the natural co-baseline.

---

## 2. Key papers

### 2.1 Wang, Antoniades, Elazar et al. — *Generalization v.s. Memorization* (ICLR 2025)

| | |
|---|---|
| **Source** | ICLR 2025; `papers/2407.14985_generalization_vs_memorization.pdf` |
| **Key contribution** | Defines **distributional memorisation** (correlation between LLM output probabilities and pretraining-data frequencies) and **distributional generalisation** (divergence). Operationalised by counting semantically-aligned input/output n-gram pair co-occurrences in the pretraining corpus → "task-gram language model". |
| **Methodology** | (1) Mine semantically-aligned input/output n-gram pairs from task data. (2) Search Pile for co-occurrences using WIMBD / infini-gram. (3) Build a task-gram LM. (4) Correlate task-gram probabilities with the LLM's predictions. |
| **Datasets** | WMT (translation), TriviaQA (factual QA), MMLU (world knowledge), GSM8K (math) |
| **Results** | TriviaQA: high distributional memorisation correlation. MMLU: weaker. GSM8K, WMT: insignificant. As scale grows, TriviaQA improvements come from more memorisation; the others from more generalisation. |
| **Code** | https://github.com/a-antoniades/llm-corpus-search (cloned in `code/llm-corpus-search/`) |
| **Why central to us** | This paper is the most mature framework for *quantifying* the memorisation-vs-generalisation split, and it's the bar any Talkie experiment should clear or exceed. Its limitation — needing an n-gram index over a multi-terabyte modern pretraining corpus — is *eliminated* for Talkie because the corpus pre-dates the test items: co-occurrence is hard-zero by construction. |

### 2.2 Fu, Shrivastava, Moore, West, Tan, Holtzman — *AbsenceBench* (2025)

| | |
|---|---|
| **Source** | Preprint 2025; `papers/2506.11440_absencebench_holtzman.pdf` |
| **Key contribution** | A medium-context benchmark that asks models to identify *omitted* (vs. inserted) content. Even Claude-3.7-Sonnet achieves only 69.6% F1 at 5 K-token contexts despite being near-superhuman on Needle-in-a-Haystack. |
| **Why central to us** | Co-authored by the hypothesis author. Demonstrates Holtzman's research style — diagnostic benchmarks that distinguish surface success from understanding. Sets the methodological tone for any Talkie-generalisation follow-up: design the cleanest probe possible, then look for *unexpected* failure modes, not just headline accuracy numbers. |

### 2.3 Li, Hopkins, Bau, Viégas, Pfister, Wattenberg — *Emergent World Representations (Othello-GPT)* (ICLR 2023)

| | |
|---|---|
| **Source** | `papers/2210.13382_othello_emergent_world.pdf` |
| **Key contribution** | Trains a GPT only on legal Othello move sequences (no rules, no board state). Linear/nonlinear probes recover a faithful internal representation of the board state, and interventional edits causally control the model's outputs. |
| **Why central to us** | The canonical example of *narrow-corpus → controlled generalisation claim*. Talkie is the natural-language analogue, with the symmetry that "what's outside the training distribution" is precisely "the modern world" rather than "an unseen board configuration". The probing/interventional methodology should be replicable: e.g., probe Talkie's hidden state for "year" or "post-1930-ness" of a passage and check whether suppressing such features changes behaviour. |

### 2.4 Chen et al. — *Evaluating Large Language Models Trained on Code (Codex / HumanEval)* (2021)

| | |
|---|---|
| **Source** | `papers/2107.03374_humaneval.pdf` |
| **Key contribution** | The HumanEval benchmark — 164 hand-written Python problems with unit tests — and the pass@k metric. |
| **Why central to us** | The Talkie team's own flagship generalisation experiment. Python was first released in 1991; a 1930-cutoff model that solves any Python problem from in-context demonstrations alone must be generalising. We can replicate, then extend with: more demonstrations, JavaScript / Lisp variants, type-shifted problems, etc. |

### 2.5 Hendrycks et al. — *MMLU* (2020)

| | |
|---|---|
| **Source** | `papers/2009.03300_mmlu.pdf` |
| **Key contribution** | 57-subject, ~14 K-question multiple-choice knowledge benchmark. Already a target task in Wang/Antoniades. |
| **Why central to us** | Primary general-knowledge probe. Crucially, MMLU questions are date-stratifiable: many require knowledge that postdates 1930 (e.g., us_foreign_policy items about 2016 percentages), while many do not (e.g., the 1850 GDP per-capita item). The Talkie blog reports an experiment of exactly this form — filter anachronistic items and the Talkie-1930 vs. Talkie-web gap halves. |

### 2.6 Wu, Pan et al. — *AntiLeak-Bench* (2024)

| | |
|---|---|
| **Source** | `papers/2312.16337_antileak_bench.pdf` |
| **Key contribution** | Automated framework for constructing benchmarks of knowledge *explicitly absent* from a target model's training set, by chaining knowledge updates and timestamps. |
| **Why central to us** | The closest existing approach to what Talkie offers structurally. AntiLeak-Bench has to work hard to find/construct contamination-free items; Talkie's hard 1930 cutoff supplies them automatically. Useful comparison for arguing that Talkie's setup is *strictly easier* than the contamination-avoidance route. |

### 2.7 Lee et al. — *Deduplicating Training Data* (2021)

| | |
|---|---|
| **Source** | `papers/2104.08758_deduplicating.pdf` |
| **Key contribution** | Quantifies pretraining-corpus duplication and benchmark train/test overlap. >4% of standard-benchmark validation items overlap with C4. Deduplication reduces verbatim memorisation 10×. |
| **Why central to us** | Establishes the baseline severity of the problem we're trying to side-step. Talkie's 1930 cutoff is a structural deduplication-and-decontamination that no amount of post-hoc filtering can match. |

### 2.8 Background (group B)

- **Jang et al. 2022 (TemporalWiki)** — Wikipedia-diff benchmark; design template for date-stratified probes.
- **Park et al. 2024 (ChroKnowledge)** — taxonomy distinguishing evolving vs. immutable knowledge; the latter is where Talkie should plausibly *succeed*.
- **Sun et al. 2024 (Head-to-Tail)** — popularity dependency of LLM factual knowledge.
- **McCoy et al. 2019 (HANS)** — canonical "models exploit spurious heuristics" study.
- **Hendrycks et al. 2020 (OOD)** and **Yang et al. 2022 (GLUE-X)** — OOD-eval framings.
- **Zhang et al. 2024 (OOD MLLMs)** — recent benchmark survey for OOD eval.
- **Biderman et al. 2024 (lm-evaluation-harness)** — the evaluation framework.
- **Schaeffer et al. 2024 (Predicting downstream)** — relevant when extending Talkie to multiple scales.

---

## 3. Common methodologies in the literature

| Methodology | Used by | Suitability for Talkie |
|---|---|---|
| **N-gram co-occurrence in pretraining corpus** | Wang/Antoniades 2024, WIMBD, infini-gram | Excellent — Talkie's pre-1930 corpus has zero overlap with modern-task n-grams *by construction*; the quantitative correlation collapses to a near-trivial bound. |
| **Train/test overlap filtering** | Lee et al. 2021, Carlini et al. 2022 | Largely *unnecessary* with Talkie because the overlap is structurally zero for modern benchmarks. |
| **Automated benchmark refresh after cutoff** | AntiLeak-Bench (Wu et al. 2024), Daily Oracle, evolveQA, OracleProto | Useful complementary technique, but Talkie removes the need to chase the cutoff because the cutoff is fixed and far enough back to admit almost any modern benchmark. |
| **Linear / non-linear probing of hidden states** | Othello-GPT (Li et al. 2022), passive-causal-learning (Lampinen 2023) | Highly applicable: probe whether Talkie develops representations of, e.g., post-1930 concepts after in-context exposure. |
| **In-context learning over a syntactically novel language** | Hahn & Goyal 2023, Lampinen 2023, the Talkie team's own HumanEval | The *core* generalisation probe for Talkie — Python wasn't invented yet, so any pass@k > 0 with in-context demos is a positive generalisation result. |
| **Compositional generalisation benchmarks (SCAN, COGS, CFQ)** | Hosseini et al. 2022, Kobayashi et al. 2024 | Could be applied to Talkie, but the language priors of these benchmarks lean modern-English. Less clean than HumanEval-style ICL. |
| **Counterfactual memorisation (retrain-with/without)** | Feldman 2020, Feldman & Zhang 2020 | Not feasible at Talkie scale (would require retraining 13 B); skip. |

---

## 4. Standard baselines

For our hypothesis-testing pipeline, the natural baselines are:

1. **`talkie-web-13b-base`** — the matched modern-twin. Same architecture, same FLOPs, FineWeb training. Any Talkie-1930 ≪ Talkie-web gap on post-1930 knowledge is *expected*; the interesting question is whether the gap is smaller than naive guessing on subset-X for any X.
2. **`talkie-1930-13b-it`** — instruction-tuned variant. Lets us probe whether RL/SFT post-training itself contributes generalisation independently of pretraining coverage.
3. **A modern frontier model with the same model size class** — e.g., Llama-3-8B / Mistral-7B-v0.3 / Qwen2.5-7B. Pin the scale near Talkie's 13 B.
4. **Ranke-4B-1929** — the cleanest cross-replication. Different architecture, different corpus, similar cutoff.
5. **Random baseline / corpus-statistical baseline** — for any task, what does an n-gram LM trained only on Talkie's pre-1930 corpus produce? (No transformer, just frequency counts.) The gap between that and Talkie-1930 quantifies the contribution of *deep* generalisation vs. *shallow* corpus statistics.

---

## 5. Evaluation metrics

| Metric | When to use | Notes |
|---|---|---|
| **Pass@k (k ∈ {1, 10, 100})** | HumanEval & code-ICL probes | Talkie team already uses pass@100. |
| **Accuracy (multiple-choice)** | MMLU, ARC, HellaSwag, TruthfulQA-MC | Use letter-prefix log-likelihood scoring; standard via lm-evaluation-harness. |
| **Bits per byte (BpB) on held-out text** | NYT On-This-Day historical-surprise probe | The Talkie team's own metric for the per-year surprisal curve. |
| **Calibration / Brier score** | Future-event prediction (DeepFund-style) | Compare Talkie-1930 vs Talkie-web on questions whose answers were determined after 1930. |
| **Distributional-memorisation correlation (Spearman, ρ)** | Wang/Antoniades 2024-style analysis | For Talkie we expect ρ ≈ 0 by construction on modern tasks — a *negative* baseline for memorisation. |
| **Probe accuracy on hidden states** | Internal-representation analysis | Linear probe for "year present in text", "post-1930 concept density", etc. |
| **F1 / EM on omission detection** | AbsenceBench-style analogues | If we want to align with Holtzman's recent diagnostic style. |

---

## 6. Datasets in the literature

| Dataset | Used by | Relevance to Talkie |
|---|---|---|
| **HumanEval** (164 Python items) | Talkie blog, Chen et al. 2021 | Strong. Python ⇒ unambiguously post-1930. |
| **MMLU** (14 K MC questions, 57 subjects) | Wang/Antoniades, Talkie blog | Strong. Date-stratifiable. |
| **TriviaQA** | Wang/Antoniades, many others | Useful but heavily knowledge-anchored. |
| **GSM8K** | Wang/Antoniades | Useful — math reasoning is largely time-invariant. Talkie should plausibly succeed if it can do reasoning at all. |
| **WMT** translation | Wang/Antoniades | Less central: Talkie's corpus is English-only. |
| **NYT On-This-Day** | Talkie blog (~5 K items) | Highly relevant; not packaged as a public dataset, but easily reconstructed from the NYT archive. |
| **The Pile** (Pythia training data) | Pythia models, baseline | Reference distribution for *modern* corpora; we compare against Talkie's corpus structurally. |
| **Common Pile / Common Corpus** | Talkie pretraining source | The actual Talkie training distribution, partly. |
| **FineWeb** | `talkie-web-13b-base` pretraining | Modern-twin distribution. |
| **Ranke-4B training corpus** (UZH 600 B) | Ranke-4B | Adjacent corpus for cross-validation of generalisation claims. |
| **AntiLeak-Bench** | Wu et al. 2024 | Comparable contamination-controlled benchmark; could be re-used directly. |
| **TemporalWiki** | Jang et al. 2022 | Wikipedia-diff items; useful for date-stratified probes. |
| **Daily Oracle / FreshBench / evolveQA** | Dai et al., Zhu et al. 2024–25 | Real-time benchmarks; mostly *too* current for Talkie (which fails on 1931+ uniformly). |
| **SCAN, COGS, CFQ** | Hosseini et al. 2022 | Compositional-generalisation toy benchmarks. Mostly orthogonal to Talkie's framing. |
| **HANS** | McCoy et al. 2019 | NLI heuristics probe. Could be useful as a sanity check. |

---

## 7. Gaps and opportunities

The hypothesis the project is built around is largely *unaddressed* in the literature in the following ways:

1. **No published paper has yet exploited a 1930-cutoff model for the express purpose of cleanly measuring generalisation.** The Talkie blog post is the first such artifact (April 2026); Holtzman's IdeaHub submission is the first proposal to push it as a *general methodology*.
2. **The closest existing technique** is to construct cutoff-aware contamination-free benchmarks (AntiLeak-Bench, evolveQA, ExAnte). These require ongoing maintenance and yield small-N test sets. Talkie sidesteps the maintenance problem entirely.
3. **Ranke-4B's multi-cutoff design** offers a generalisation *scaling* story (1913 → 1946) that Talkie alone can't deliver. The two together form a cleaner story than either does in isolation.
4. **No prior work measures the same generalisation gap on the same architecture across pre-modern vs. modern corpora.** Talkie + Talkie-web is the first such pairing in the public record.
5. **Probing / interventional analyses on Talkie are wide open.** Othello-GPT-style probes for "knows X is post-1930" representations, "physical/chemical principles" representations, etc., should be feasible and informative.

---

## 8. Recommendations for our experiment

### 8.1 Headline experiment (must-do)
**Replicate-and-extend the Talkie team's HumanEval probe.** Run Talkie-1930-13b-base, Talkie-1930-13b-it, and Talkie-web-13b-base on HumanEval with `k ∈ {0, 1, 3, 10}` in-context demonstrations of Python syntax. Report pass@1 / pass@10 / pass@100 broken down by:
- problem difficulty,
- presence of post-1930 library imports (e.g., `numpy`, `pandas`),
- demo count.

The Talkie-1930 vs. Talkie-web gap *at fixed in-context conditioning* is the cleanest single number quantifying the generalisation question.

### 8.2 Date-stratified MMLU
Build an automated date-extractor (regex + NER for `19xx`/`20xx` and post-1930 entities); partition MMLU into pre-1930-decidable, post-1930-decidable, and ambiguous. Compare Talkie-1930 vs. Talkie-web accuracy on each subset. The Talkie blog reports the gap halves after this filter; reproduce with public code and confidence intervals.

### 8.3 Distributional-memorisation lower bound
Run a Wang/Antoniades-style task-gram analysis on Talkie's pretraining corpus — if we can get even a partial n-gram index. Even *partial* coverage suffices to show that modern tasks have near-zero memorisation contribution, which is the load-bearing claim of the hypothesis.

### 8.4 NYT On-This-Day surprise curve
Reconstruct from the NYT archive feed; reproduce the Talkie team's per-decade BpB curve; add Talkie-web as a control to show the curve is *flat* without the cutoff.

### 8.5 In-context generalisation, beyond Python
Construct minimal "concept" datasets where Talkie has zero prior exposure:
- a few hand-built post-1930 vocabulary items (`smartphone`, `quark`, `Higgs boson` — see `datasets/post1930_probes/`),
- a tiny JavaScript ICL probe (`code/javascript_examples`),
- a chemistry-naming probe using post-1930 compounds.

Measure: zero-shot baseline, k-shot improvement, and whether `talkie-1930-13b-it` (with modern-style RL refinement) recovers any of the gap that the base model misses.

### 8.6 Cross-validation with Ranke-4B-1929
If compute permits, run the same battery on Ranke-1929. Any result reproducible across both architectures + corpora generalises beyond Talkie itself, which is the load-bearing claim for the *methodology*.

---

## 9. Methodological considerations / risks

1. **Subject-matter confound.** The Talkie README warns: temporal coverage is *not* the only thing that differs between Talkie's corpus and FineWeb — subject mix, register, OCR quality, and prose style all differ. Any single comparison can be confounded; we should always (a) interpret the Talkie-web baseline as a *direction*, not a ceiling, and (b) use Ranke as a cross-check.
2. **OCR noise.** Pre-1930 text is OCR-extracted; the Talkie team reports ~30 % learning efficiency vs. clean transcriptions. Some "generalisation failures" may be OCR-noise failures.
3. **Tokeniser asymmetry.** Talkie's tokeniser (`vocab.txt`) is trained on pre-1930 text; modern terminology gets fragmented inefficiently. This biases against Talkie on ICL probes — but the bias is *visible* (count token lengths) and can be quantified.
4. **Reverse-asymmetry in benchmark construction.** Every modern NLP benchmark was written by post-1930 humans; the *style* of the benchmark items already encodes 20th-century framings. This is not fixable; we should report it as a limitation.
5. **Anachronistic RL refinement.** The Talkie team observes that RLAIF (using modern models as judges) "inevitably shapes talkie's behaviour anachronistically." The IT variant is contaminated by judge feedback even if its pretraining is not. For *generalisation* claims, prefer the base model.
6. **Single-scale claim.** Only one Talkie size is released (13 B). Any scaling claim is speculative until 7 B / 70 B variants exist. Frame results as point estimates, not scaling laws.
7. **Compute budget.** A full HumanEval pass@100 sweep across three Talkie checkpoints with k ∈ {0,1,3,10} demos × multiple seeds is on the order of $1 K–$10 K of A100 time. Plan accordingly.

The overall conclusion is that the hypothesis is methodologically sound — Talkie is a uniquely clean testbed — and the experiment design is fairly well-specified. The risks are subject-matter confounds, tokeniser/OCR artifacts, and compute. The papers and tools gathered in `papers/`, `code/`, and `datasets/` directly support the experiment plan above.
