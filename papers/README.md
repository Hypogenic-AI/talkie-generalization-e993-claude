# Downloaded Papers

Papers gathered for the "Is it easier to test if Talkie generalizes?" project.
All files are arXiv PDFs. Group A papers are directly load-bearing for the
hypothesis; Group B are general background.

---

## Group A: Directly load-bearing

### A.1 — `2407.14985_generalization_vs_memorization.pdf`
**Wang, Antoniades, Elazar, Amayuelas, Albalak, Zhang, Wang. *Generalization v.s. Memorization: Tracing Language Models' Capabilities Back to Pretraining Data*. ICLR 2025.**

- Defines **distributional memorization**: correlation between LLM output probabilities and pretraining data frequency, and **distributional generalization** as the divergence.
- Methodology: builds a **task-gram language model** by counting co-occurrences of semantically-related input/output n-gram pairs in the pretraining corpus (using WIMBD / infini-gram).
- Tasks: WMT (translation), TriviaQA (factual QA), MMLU (world knowledge), GSM8K (math). Models: Pythia trained on The Pile.
- Findings: TriviaQA shows strongest memorisation; MMLU weaker; WMT and GSM8K largely generalisation. As scale grows, TriviaQA gains from more memorisation; others gain from more generalisation.
- **Why this is THE key methodological paper for us:** their framework reduces generalisation testing to n-gram lookup in the pretraining corpus. Talkie's pre-1930 corpus, by construction, contains zero co-occurrences of modern input/output n-grams — so the "memorisation contribution" is hard-zero on modern tasks, leaving any non-trivial Talkie performance attributable to generalisation. This sidesteps the *entire* contamination-adjustment burden that this paper had to engineer for Pythia/Pile.
- Code: https://github.com/a-antoniades/llm-corpus-search (cloned in `code/llm-corpus-search/`)

### A.2 — `2506.11440_absencebench_holtzman.pdf`
**Fu, Shrivastava, Moore, West, Tan, Holtzman. *AbsenceBench: Language Models Can't Tell What's Missing*. Preprint 2025.**

- Co-authored by Ari Holtzman (the hypothesis author). Sets out his recent style: design clean diagnostic benchmarks that distinguish surface success from understanding.
- Finds that even SOTA models drop from ~95% on Needle-in-a-Haystack to ~70% on AbsenceBench (detecting deliberately omitted content).
- Argues this stems from a Transformer-attention limit: omissions have no key to attend to.
- **Why it matters for us:** demonstrates Holtzman's general research strategy — find diagnostic tasks where reasonable-looking generalisation breaks. The Talkie hypothesis sits in this same lineage.

### A.3 — `2506.01732_common_pile.pdf`
**Langlais, Chizhov, Arnett, et al. *CommonCorpus: The Largest Collection of Ethical Data for LLM Pre-Training*. ICLR 2026.**

- Common Pile / Common Corpus is the public-domain pre-training corpus that Talkie's pre-1930 cohort is drawn from.
- Documents corpus composition, licensing, and quality filtering.
- **Why it matters for us:** specifies what's actually in Talkie's training data, which is what makes the generalisation framing legitimate.

### A.4 — `2210.13382_othello_emergent_world.pdf`
**Li, Hopkins, Bau, Viégas, Pfister, Wattenberg. *Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task*. ICLR 2023.**

- Famous "Othello-GPT" paper. Trains a GPT on legal Othello move sequences and shows that despite no exposure to rules or board state, the model develops a non-linear internal representation of the board (verified by probing and intervention).
- **Why it matters for us:** the foundational example of using a *narrow, controlled training distribution* to make a sharp generalisation claim. Talkie is the natural-language analogue: train on a tightly bounded corpus, then ask whether the model has internalised structure that extends past the corpus boundary.

### A.5 — `2107.03374_humaneval.pdf`
**Chen et al. *Evaluating Large Language Models Trained on Code* (Codex / HumanEval). 2021.**

- Introduces HumanEval (164 Python problems) and pass@k.
- **Why it matters for us:** HumanEval is the Talkie team's flagship generalisation experiment — Python didn't exist in 1930, so passing Python unit tests after a few in-context demos is the cleanest possible "did you generalise?" question. We can replicate and extend.

### A.6 — `2009.03300_mmlu.pdf`
**Hendrycks et al. *Measuring Massive Multitask Language Understanding* (MMLU). 2020.**

- 57-subject multiple-choice benchmark. Mix of pre- and post-1930 topics.
- **Why it matters for us:** primary "general knowledge" probe. Many subjects are date-stratifiable, letting us measure the Talkie-1930 vs. Talkie-web gap on pre- vs. post-1930 content separately.

### A.7 — `2312.16337_antileak_bench.pdf`
**Wu, Pan, Xie, Zhou, Zhao, Ma, Du, Mao, Luu, Wang. *AntiLeak-Bench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge*. 2024.**

- Automated benchmark-construction framework that explicitly builds samples containing knowledge *absent* from a target model's training set.
- **Why it matters for us:** AntiLeak-Bench's contamination-avoidance pipeline is the closest existing analogue to what we get *for free* with Talkie's hard 1930 cutoff. Useful prior art for justifying that Talkie's setup actually beats current contamination workarounds.

### A.8 — `2104.08758_deduplicating.pdf`
**Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, Carlini. *Deduplicating Training Data Makes Language Models Better*. 2021.**

- Foundational paper on how pretraining-data duplicates inflate verbatim memorisation in evaluation, and on the train/test overlap rates in standard benchmarks.
- **Why it matters for us:** quantifies the size of the contamination problem we're trying to side-step by using Talkie.

---

## Group B: Background and supporting

### B.1 — `2204.14211_temporalwiki.pdf`
Jang et al., *TemporalWiki: A Lifelong Benchmark for Training and Evaluating Ever-Evolving Language Models*. EMNLP 2022. Wikipedia-diff-based benchmark for temporal misalignment. Provides one model of how to operationalise "knowledge cutoff testing" — relevant when designing date-stratified MMLU subsets.

### B.2 — `2410.04699_chronoknowledge.pdf`
Park et al., *ChroKnowledge: Unveiling Chronological Knowledge of Language Models in Multiple Domains*. 2024. Distinguishes knowledge that evolves over time (laws, history) from knowledge that doesn't (math). The latter is exactly what we'd hope Talkie generalises on.

### B.3 — `2410.16454_unlearning_quantization.pdf`
*Catastrophic Failure of LLM Unlearning via Quantization*, ICLR 2025. Relevant for the alternative-to-Talkie strategy: trying to *remove* knowledge from a modern LLM rather than retrain. Shows the unlearning route is brittle, strengthening the case for the from-scratch Talkie approach.

### B.4 — `2308.10168_head_to_tail_knowledge.pdf`
Sun et al., *Head-to-Tail: How Knowledgeable are LLMs?* (Meta Reality Labs). Demonstrates the long-tail/popularity dependency of LLM factual knowledge. Useful when interpreting MMLU disparities.

### B.5 — `2207.14241_glue_x.pdf`
Yang et al., *GLUE-X: Evaluating NLU Models from an OOD Generalization Perspective*. 2022. A standard OOD-eval framing for NLU — a useful reference point for the kind of evaluation surface we may want to construct for Talkie.

### B.6 — `2004.06100_pretrained_transformers_ood.pdf`
Hendrycks et al., *Pretrained Transformers Improve Out-of-Distribution Robustness*. 2020. Foundational OOD-robustness paper; establishes that pretraining diversity helps OOD performance — which makes the Talkie design (pre-1930 *only*) an extreme corner of that design space.

### B.7 — `1902.01007_hans_mccoy.pdf`
McCoy, Pavlick, Linzen. *Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in NLI* (HANS). 2019. The canonical "models exploit spurious heuristics" paper. Same philosophical lineage as Holtzman's research style.

### B.8 — `2402.06599_ood_generalization_mllm.pdf`
Zhang et al., *On the Out-Of-Distribution Generalization of Multimodal Large Language Models*. 2024. Recent OOD evaluation work; useful as a comparison point on how OOD is operationalised in current literature.

### B.9 — `2405.14782_lm_eval_harness.pdf`
Biderman et al., *Lessons from the Trenches on Reproducible Evaluation of Language Models*. 2024. The `lm-evaluation-harness` paper. We've cloned the corresponding repo (`code/lm-evaluation-harness/`) — the standard scaffold for running our MMLU/HumanEval/etc. comparisons.

### B.10 — `2406.04391_predicting_downstream_capabilities.pdf`
Schaeffer et al., *Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?* 2024. Background reference for the *scaling* dimension of any Talkie follow-ups (only one model size released so far).
