# Resources Catalog

Comprehensive catalog of all papers, datasets, and code repositories gathered for *Is it easier to test if Talkie generalizes?*

---

## Summary

- **18 papers** downloaded to `papers/` (~67 MB total)
- **3 datasets** downloaded to `datasets/` (~4.6 MB local; pretraining corpora not committed)
- **4 code repositories** cloned to `code/`
- Talkie official inference library, EleutherAI lm-evaluation-harness, the Wang/Antoniades llm-corpus-search code, and the UZH Ranke-4B history-llms repo

---

## Papers

See `papers/README.md` for detailed annotations. Group A is directly load-bearing; Group B is background.

### Group A — load-bearing

| # | File | Authors | Year | Key info |
|---|---|---|---|---|
| A.1 | `papers/2407.14985_generalization_vs_memorization.pdf` | Wang, Antoniades, Elazar et al. | 2024 (ICLR 2025) | Defines distributional memorisation; task-gram methodology; reference paper for our framework. |
| A.2 | `papers/2506.11440_absencebench_holtzman.pdf` | Fu, Shrivastava, Moore, West, Tan, **Holtzman** | 2025 | Diagnostic-benchmark methodology by the hypothesis author. |
| A.3 | `papers/2506.01732_common_pile.pdf` | Langlais et al. (Pleias) | ICLR 2026 | The Common Pile corpus underlying Talkie's pretraining. |
| A.4 | `papers/2210.13382_othello_emergent_world.pdf` | Li, Hopkins, Bau, Viégas, Pfister, Wattenberg | ICLR 2023 | Othello-GPT: narrow-corpus → controlled generalisation claim. |
| A.5 | `papers/2107.03374_humaneval.pdf` | Chen et al. (OpenAI Codex) | 2021 | HumanEval benchmark + pass@k. |
| A.6 | `papers/2009.03300_mmlu.pdf` | Hendrycks et al. | 2020 | MMLU. |
| A.7 | `papers/2312.16337_antileak_bench.pdf` | Wu, Pan et al. | 2024 | Contamination-free benchmark automation. |
| A.8 | `papers/2104.08758_deduplicating.pdf` | Lee, Ippolito, Carlini et al. | 2021 | Deduplication and train/test overlap. |

### Group B — background

| # | File | Authors | Year | Topic |
|---|---|---|---|---|
| B.1 | `papers/2204.14211_temporalwiki.pdf` | Jang et al. | 2022 | TemporalWiki temporal-misalignment benchmark. |
| B.2 | `papers/2410.04699_chronoknowledge.pdf` | Park et al. | 2024 | ChroKnowledge taxonomy. |
| B.3 | `papers/2410.16454_unlearning_quantization.pdf` | Zhang et al. | ICLR 2025 | Unlearning fragility (why train-from-scratch helps). |
| B.4 | `papers/2308.10168_head_to_tail_knowledge.pdf` | Sun et al. (Meta) | 2024 | Long-tail factual knowledge. |
| B.5 | `papers/2207.14241_glue_x.pdf` | Yang et al. | 2022 | GLUE-X OOD eval. |
| B.6 | `papers/2004.06100_pretrained_transformers_ood.pdf` | Hendrycks et al. | 2020 | OOD robustness foundation. |
| B.7 | `papers/1902.01007_hans_mccoy.pdf` | McCoy, Pavlick, Linzen | 2019 | HANS spurious-heuristic study. |
| B.8 | `papers/2402.06599_ood_generalization_mllm.pdf` | Zhang et al. | 2024 | Modern OOD-generalisation survey. |
| B.9 | `papers/2405.14782_lm_eval_harness.pdf` | Biderman et al. (EleutherAI) | 2024 | lm-evaluation-harness paper. |
| B.10 | `papers/2406.04391_predicting_downstream_capabilities.pdf` | Schaeffer et al. | 2024 | Predictability-of-scaling background. |

### Pages output

Each Group A paper has been split into 3-page chunks under `papers/pages/` for selective deep reading. Manifests are in the same directory.

### Paper-finder JSONL outputs

Raw paper-finder result files are in `paper_search_results/`. The non-empty file is `knowledge_cutoff_temporal_LLM_evaluation_benchmark_20260512_015833.jsonl` (39 papers). The "diligent" search files are empty because the diligent endpoint timed out — fall back to the fast-mode results documented inline above if you need to expand the citation list.

---

## Datasets

See `datasets/README.md` for full per-dataset documentation and load instructions.

| Name | Location | Size | Format | Purpose |
|---|---|---|---|---|
| HumanEval | `datasets/HumanEval.jsonl` (+ `.gz`) | 164 problems, ~210 KB | JSONL | Python codegen / pass@k. |
| MMLU | `datasets/mmlu/data/{dev,val,test}/*.csv` | ~14 K test items + 1.5 K val + 285 dev, ~4.3 MB | CSV per subject | 57-subject MC knowledge benchmark. |
| Post-1930 probes | `datasets/post1930_probes/seed_probes.jsonl` | 20 hand-curated items | JSONL | Date-anchored generalisation probes. |

Pretraining corpora (Talkie's pre-1930 corpus, FineWeb) and model weights are **not** in this repo — they are listed in `datasets/README.md` as on-demand downloads for the experiment runner. A `datasets/.gitignore` excludes any tar/zip/parquet files that get pulled later while preserving small text/CSV/JSONL files for documentation.

---

## Code repositories

See `code/README.md` for detailed documentation.

| Repo | Path | URL | Purpose |
|---|---|---|---|
| Talkie inference library | `code/talkie/` | https://github.com/talkie-lm/talkie | Run Talkie-1930 / Talkie-web inference; model registry with all three HF repo IDs. |
| lm-evaluation-harness | `code/lm-evaluation-harness/` | https://github.com/EleutherAI/lm-evaluation-harness | Standard evaluation framework for MMLU, HumanEval, ARC, GSM8K, etc. |
| llm-corpus-search | `code/llm-corpus-search/` | https://github.com/a-antoniades/llm-corpus-search | Reference impl. of Wang/Antoniades 2024 task-gram methodology. |
| history-llms (Ranke-4B) | `code/history-llms/` | https://github.com/DGoettlich/history-llms | Pre-release of the UZH/Cologne 4B time-locked LLM family. |

---

## Resource gathering notes

### Search strategy

1. Identified that *Talkie-1930* is a real released model (April 2026), not a hypothetical, via web search. Confirmed authors (Levine/Duvenaud/Radford), separated from the *idea* author (Holtzman).
2. Read the official Talkie blog post (`https://talkie-lm.com/introducing-talkie`) to extract architecture, training corpus, the team's own experiments (HumanEval, NYT On-This-Day, anachronism-filtered MMLU), and contamination-free framing.
3. Ran paper-finder in fast mode (diligent timed out) across four query themes:
   - "knowledge cutoff temporal LLM evaluation benchmark"
   - "LLM generalization vs memorization pretraining data"
   - "out-of-distribution generalization language model held out evaluation"
   - "in-context learning generalization compositional out of distribution probing"
   - "language model trained controlled corpus probing emergent capabilities"
   - "historical language model corpus pre 1930 time-locked LLM" (yielded only noise — no relevant pre-1930 LLM papers exist outside Talkie/Ranke).
4. Cross-referenced the top results with Ari Holtzman's published work (via web search) to identify AbsenceBench as the relevant Holtzman methodological precedent.
5. Searched for similar time-locked LLM efforts → discovered Ranke-4B (UZH/Cologne) as the unique analogue.

### Selection criteria

- *Group A papers*: must be either (a) the hypothesis author's own work, (b) Talkie blog references, or (c) the most-cited / most-direct methodological precedents for measuring generalisation vs. memorisation. Capped at 8 papers.
- *Group B papers*: representative samples from each related sub-literature, capped at 10.
- *Code repos*: every repo that the experiment runner will actually invoke or extend.
- *Datasets*: only benchmarks that are (i) small enough to commit, (ii) directly used by either the Talkie blog or one of the load-bearing papers, and (iii) have a clear role in the planned experiments. Larger downloads (Common Pile, FineWeb, Talkie weights) are documented as on-demand.

### Challenges encountered

1. **Initial confusion about whether Talkie was real or hypothetical.** Resolved by web search; Talkie is a real April-2026 release. Ari Holtzman is *not* a Talkie team member but proposed the test-methodology hypothesis separately.
2. **Paper-finder diligent mode timed out** twice. Fast mode worked; results are sufficient.
3. **arXiv rate-limiting** during parallel PDF downloads. Switched to sequential serial downloads with delays.
4. **A few arXiv IDs were misguessed** (we re-named or removed those papers — see `papers/README.md`).
5. **No public Talkie pretraining dataset.** It's built from Common Pile + Institutional Data Initiative + Internet Archive — a composite that's not packaged anywhere. For task-gram analysis the experiment runner will need to either (a) build a local index or (b) settle for the "co-occurrence = 0 by construction" structural argument.

### Gaps and workarounds

- *No clean post-1930 ICL benchmark exists* outside HumanEval. We supply a 20-item seed (`datasets/post1930_probes/`) and outline the structure for auto-extending it.
- *No academic paper specifically on Talkie* yet. The blog post and HuggingFace model cards are the only authoritative sources. We've captured the key quantitative claims from the blog inline in `literature_review.md` and `papers/README.md`.
- *Ranke-4B weights are not yet publicly downloadable* (pre-release). The repo is included so the experiment runner can be ready when the weights drop.

---

## Recommendations for experiment design

(Summary — see `literature_review.md` §8 for the full plan.)

### Primary dataset(s)
1. **HumanEval** — flagship generalisation probe (Python ≫ 1930 cutoff).
2. **MMLU (date-stratified)** — knowledge generalisation across pre-/post-1930 subsets.
3. **`post1930_probes` seed** — targeted probes for neologisms, scientific concepts, ICL of post-1930 languages.

### Baseline methods
1. `talkie-web-13b-base` (matched modern-twin) — the load-bearing baseline.
2. `talkie-1930-13b-it` — to test whether anachronistic RL post-training closes the gap.
3. Ranke-4B-1929 — cross-architecture replication.
4. A modern public 7–13B model (Llama-3-8B / Mistral-7B / Qwen2.5-7B) — sanity-check that the absolute Talkie-web numbers are not architecture-specific.

### Evaluation metrics
- pass@1, pass@10, pass@100 (code).
- Accuracy (MC).
- Bits per byte / per-token surprisal (continuous).
- Distributional-memorisation correlation (Spearman ρ over n-gram counts).

### Code to adapt / reuse
- `code/talkie/` — load and run the models.
- `code/lm-evaluation-harness/` — wrap and run benchmarks.
- `code/llm-corpus-search/` — adapt for any task-gram analysis we run on Talkie's corpus subset.
- `code/history-llms/` — Ranke-4B cross-replication.
