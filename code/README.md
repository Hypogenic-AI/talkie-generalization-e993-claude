# Cloned Code Repositories

Repositories cloned for the "Is it easier to test if Talkie generalizes?" project.

| Repo | Path | Purpose |
|------|------|---------|
| `talkie-lm/talkie` | `code/talkie/` | Official inference library for Talkie-1930-13B (base / IT / web). |
| `EleutherAI/lm-evaluation-harness` | `code/lm-evaluation-harness/` | Standard evaluation framework for MMLU, HellaSwag, ARC, GSM8K, HumanEval, etc. |
| `a-antoniades/llm-corpus-search` | `code/llm-corpus-search/` | Reference implementation of the *Generalization v.s. Memorization* (Wang/Antoniades ICLR 2025) task-gram methodology. |
| `DGoettlich/history-llms` | `code/history-llms/` | Ranke-4B family (UZH/Cologne) — time-locked 4B-parameter LLMs with cutoffs 1913/1929/1933/1939/1946. Closest existing analogue to Talkie. |

---

## 1. `code/talkie/` — Talkie inference library

**Repo:** https://github.com/talkie-lm/talkie
**Authors:** Alec Radford, Nick Levine, David Duvenaud (non-profit)
**License:** Apache 2.0

### What it provides
- A minimal PyTorch inference library for the three Talkie 13B checkpoints.
- A CLI (`talkie generate`, `talkie chat`, `talkie download`, `talkie list`).
- Model registry (`src/talkie/config.py`) listing the three HF repos:
  - `talkie-lm/talkie-1930-13b-base` (vintage base)
  - `talkie-lm/talkie-1930-13b-it`  (vintage instruction-tuned, RL-refined via online DPO)
  - `talkie-lm/talkie-web-13b-base` (modern twin trained on FineWeb, same architecture & FLOPs)

### Architecture (from `src/talkie/model.py`)
```python
@dataclass
class GPTConfig:
    vocab_size: int = 65536
    n_layer:    int = 40
    n_head:     int = 40
    n_embd:     int = 5120
    head_dim:   int = 128
```
A 40-layer, 40-head decoder-only GPT with RoPE, SwiGLU, RMSNorm, embedding skip connections, and per-head/per-layer gain parameters. ≈ 13 B parameters.

### Hardware requirements
- Python ≥ 3.11, PyTorch ≥ 2.1 (CUDA 12.8 build)
- GPU with **≥ 28 GB VRAM** for bfloat16 inference
- ~26–50 GB disk per checkpoint

### Key usage
```python
from talkie import Talkie, Message

m_old = Talkie("talkie-1930-13b-base")
m_new = Talkie("talkie-web-13b-base")

r_old = m_old.generate("In the year 1969, the first man on the Moon was", max_tokens=50)
r_new = m_new.generate("In the year 1969, the first man on the Moon was", max_tokens=50)
```

### How the experiment runner should use it
1. `download_model("talkie-1930-13b-base")` and `download_model("talkie-web-13b-base")` upfront.
2. Log-probability scoring of candidate completions (the library currently exposes greedy/temperature sampling and streaming; log-prob scoring will require a small extension to `model.py` to return token logits without sampling).
3. The IT model expects the `format_chat` / `format_prompt` template from `chat.py`; for base-model probing, just call `.generate()` directly.

---

## 2. `code/lm-evaluation-harness/`

**Repo:** https://github.com/EleutherAI/lm-evaluation-harness
**Paper:** Biderman et al. 2024 — `papers/2405.14782_lm_eval_harness.pdf`

The standard evaluation framework. Out-of-the-box supports MMLU, HellaSwag, ARC, GSM8K, BBH, TruthfulQA, HumanEval, MATH, and dozens more — all with consistent prompting and scoring.

### How the experiment runner should use it
- Add a thin `lm_eval.models.Talkie`-style wrapper that delegates to the `talkie` library.
- Then `lm_eval --model talkie --tasks mmlu,hellaswag,arc_easy,arc_challenge,truthfulqa_mc1,humaneval --model_args pretrained=talkie-1930-13b-base` runs the whole suite.
- For our hypothesis the key tasks are: `mmlu`, `humaneval`, `arc_*`, plus any newly registered Talkie-specific tasks.

---

## 3. `code/llm-corpus-search/`

**Repo:** https://github.com/a-antoniades/llm-corpus-search
**Paper:** Wang/Antoniades et al. ICLR 2025 — `papers/2407.14985_generalization_vs_memorization.pdf`

### What it provides
- `wimbd_search.py` — n-gram search against WIMBD (Allen AI's index over The Pile and other large corpora) or the public `infini-gram` API.
- `wimbd_preprocess.py`, `wimbd_process.py` — pipeline for extracting task n-gram pairs from MMLU, TriviaQA, WMT, GSM8K.
- `analysis/` — notebooks for plotting the distributional-memorisation correlations.

### How the experiment runner should use it
- The methodology is what matters more than the code: for any task, build the task-gram table, then count co-occurrences in the *Talkie* pretraining corpus.
- Talkie's corpus is not in WIMBD/infini-gram. Two options:
  1. **Easier**: argue from the hard 1930 cutoff that modern-task co-occurrence counts are *zero by construction* — so any non-trivial Talkie performance is non-memorisation. Costs us a quantitative correlation but buys us the entire methodological move for free.
  2. **Harder**: build a local n-gram index over Talkie's pretraining corpus and reproduce the full analysis. The repo's preprocessing pipeline is the template.

---

## 4. `code/history-llms/` — Ranke-4B (UZH/Cologne)

**Repo:** https://github.com/DGoettlich/history-llms
**Authors:** Göttlich, Loibner, Jiang, Voth
**Status:** Pre-release as of 2025-12; full release pending.

### What it provides
A family of **4 B-parameter Qwen3-architecture** LLMs trained from scratch on **80 B tokens** of historical text up to cutoffs ∈ {1913, 1929, 1933, 1939, 1946}, drawn from a curated 600 B-token corpus. Plus SFT and GRPO post-training, and an explicit validation step: "the model learns pre-, but not post-knowledge-cutoff facts."

### Why we care
- **The only other family of time-locked models in existence.** Ranke-1929 has a cutoff one year earlier than Talkie-1930, on a different architecture and corpus → natural cross-replication baseline.
- Ranke spans multiple cutoffs → cleanly measures *how the generalisation gap shrinks* as the cutoff approaches modern times. Talkie alone can't show this trend.
- Ranke explicitly *does not* deduplicate (frequency = historical importance) and *does not* filter for toxicity. Talkie's corpus filtering is different (Talkie is heavily Common Pile-driven). Comparing the two illuminates which generalisation results are corpus-artifact vs. genuine.

### How the experiment runner should use it
- Pull from HuggingFace: https://huggingface.co/uzh-echist-org (gated, but the repo lists this as the model home).
- Pre-release notes in `code/history-llms/ranke-4b/prerelease_notes.md` — read these before running anything; many design decisions differ from Talkie's.

---

## Notes on running

- **VRAM:** Both Talkie 13B and Ranke 4B comfortably fit on a single H100 / A100-80G. Ranke fits on a 24 GB GPU; Talkie does not (needs 28 GB+). A 40 GB A100 is the safest single-GPU target.
- **Throughput:** Both libraries use vanilla PyTorch; expect ~30–60 tokens/sec at 13 B / bf16 on a single A100. For HumanEval (164 problems × 100 samples each at temperature 0.8) plan for tens of hours on a single GPU.
- **CPU-only sanity testing:** Possible but slow — load with `device="cpu"` and use short prompts.
