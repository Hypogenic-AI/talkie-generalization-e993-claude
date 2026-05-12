# Datasets

Datasets gathered for the "Is it easier to test if Talkie generalizes?" project.
Data files are not committed to git (except small samples and seeds); see
`.gitignore` and the per-dataset download instructions below.

## Summary

| Name | Size | Format | Purpose |
|------|------|--------|---------|
| `HumanEval.jsonl` | 164 problems, ~210 KB | JSONL | OOD Python-codegen probe — Python postdates Talkie's 1930 cutoff, so HumanEval is *by construction* a pure generalisation test. |
| `mmlu/` | 14,042 test items + 1,531 val + 285 dev across 57 subjects, ~4.3 MB | CSV (per subject) | Standard knowledge benchmark. Many subjects contain post-1930 facts → ideal for measuring the Talkie-1930 vs. Talkie-web knowledge gap, and for the "filter anachronistic questions" experiment described in the Talkie blog. |
| `post1930_probes/` | 20 hand-curated seed items | JSONL | Custom probe set focused on date-anchored generalisation tests. Intended to be extended by the experiment runner. |

The training corpora (Talkie's pre-1930 corpus and FineWeb) are **not** downloadable here:
- Talkie's 260 B-token pre-1930 corpus is not released as a single artifact; it was built from Common Pile (Common Corpus / Pleias), Institutional Data Initiative, and Internet Archive scans.
- FineWeb is hosted on HuggingFace (`HuggingFaceFW/fineweb`) and is multi-TB.
- For generalisation/memorisation analysis we recommend using the `infini-gram` API instead of holding the full corpora locally.

---

## Dataset 1: HumanEval

### Overview
- **Source:** https://github.com/openai/human-eval
- **Paper:** Chen et al. 2021, *Evaluating Large Language Models Trained on Code* — `papers/2107.03374_humaneval.pdf`
- **Size:** 164 problems, ~210 KB uncompressed
- **Format:** JSONL — each line has `task_id`, `prompt`, `entry_point`, `canonical_solution`, `test`
- **Task:** Generate a Python function body to pass the unit tests; evaluated with pass@k

### Why it matters for Talkie
Python was first released in **1991**. None of Python's syntax, semantics, idioms, or the very *concept* of a high-level interpreted language existed before 1930. Asking Talkie-1930 to do HumanEval with in-context Python demonstrations is therefore a textbook out-of-distribution generalisation probe — and the Talkie team already report results for this in their blog post. Reproducing and extending this is the cleanest single experiment for the hypothesis.

### Already downloaded
`datasets/HumanEval.jsonl` (and `.gz`)

### Loading
```python
import json
with open("datasets/HumanEval.jsonl") as f:
    problems = [json.loads(line) for line in f]
print(len(problems), problems[0]["task_id"])  # → 164 HumanEval/0
```

To re-download:
```bash
curl -L -o datasets/HumanEval.jsonl.gz \
    https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
gunzip datasets/HumanEval.jsonl.gz
```

---

## Dataset 2: MMLU (Massive Multitask Language Understanding)

### Overview
- **Source:** Hendrycks et al. 2020, https://github.com/hendrycks/test
- **Paper:** `papers/2009.03300_mmlu.pdf`
- **Size:** 4.3 MB total (test 3.4 MB, val 553 KB, dev 277 KB). The auxiliary_train split is removed to keep the directory small.
- **Format:** One CSV per subject; columns are `question, A, B, C, D, answer_letter`
- **Subjects:** 57 (anatomy, astronomy, ML, world_religions, US_foreign_policy, ...)

### Why it matters for Talkie
1. **Standard generalisation benchmark** the Talkie blog post already reports on.
2. **Date-stratification probe**: filter questions that reference post-1930 entities/events and compare Talkie-1930 vs. Talkie-web accuracy on the *complement* vs. *intersection* with the 1930 cutoff.
3. **Memorisation analysis from Wang et al. 2024** uses MMLU as a key task — the same task-gram methodology can be applied to Talkie's corpus (modulo getting an index, e.g. via `infini-gram`).

### Already downloaded
`datasets/mmlu/data/{dev,val,test}/<subject>_<split>.csv`

### Loading
```python
import csv, glob
def load_split(split="test"):
    items = []
    for path in glob.glob(f"datasets/mmlu/data/{split}/*.csv"):
        subject = path.split("/")[-1].rsplit("_", 1)[0]
        with open(path) as f:
            for row in csv.reader(f):
                if len(row) >= 6:
                    items.append({"subject": subject, "question": row[0],
                                  "choices": row[1:5], "answer": row[5]})
    return items
test_items = load_split("test")
print(len(test_items))  # ~14,042
```

To re-download:
```bash
curl -L -o data.tar https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xf data.tar -C datasets/mmlu/
rm -rf datasets/mmlu/data/auxiliary_train   # ~40 MB, optional
```

### Sample question (us_foreign_policy)
```
"As of 2016, about what percentage of adults aged 18 years or older were overweight?",10%,20%,40%,80%,C
What was GDP per capita in the United States in 1850 when adjusting for inflation and PPP in 2011 prices?,About $300,About $3k,About $8k,About $15k,B
```
(The first is post-1930-anchored → Talkie-1930 should fail; the second is pre-1930-anchored → Talkie-1930 should plausibly succeed if it generalises numerical reasoning.)

---

## Dataset 3: Post-1930 probe seed

### Overview
- **Source:** Hand-curated for this project — see `datasets/post1930_probes/seed_probes.jsonl`
- **Size:** 20 items
- **Format:** JSONL with `id`, `year`, `category`, `concept`, `prompt`, `expected_modern`, `kind`

### Why it matters
A small, *targeted* probe set covering categories the larger benchmarks don't cleanly isolate:
- post-1930 **neologisms** (smartphone, Turing Test)
- post-1930 **scientific concepts** (quark, REM sleep, Higgs boson)
- **ICL probes** that worked-example Python and JavaScript and ask Talkie to continue (the *generalisation* hypothesis test)
- date-anchored **events** with verifiable dates

Intended to be auto-extended by the experiment runner using NYT "On This Day", Wikidata-by-year, and date-extracted MMLU items.

---

## Downloads needed for the experiment runner (not yet pulled)

These are *optional*: the experiment runner should pull them only if the chosen experiments need them.

| Dataset | Source | Why |
|---------|--------|-----|
| Talkie-1930-13b-base weights | https://huggingface.co/talkie-lm/talkie-1930-13b-base | The model under test. ~26 GB. Requires ≥28 GB VRAM. |
| Talkie-web-13b-base weights | https://huggingface.co/talkie-lm/talkie-web-13b-base | Modern-twin baseline. Same shape. |
| Talkie-1930-13b-it weights | https://huggingface.co/talkie-lm/talkie-1930-13b-it | Instruction-tuned variant; useful for chat-style probing. |
| Common Corpus (subset) | https://huggingface.co/datasets/PleIAs/common_corpus | The *source* of Talkie's pre-1930 corpus; useful for n-gram memorisation analysis. |
| infini-gram API | https://infini-gram.io | Lookup service for n-gram counts in major pretraining corpora (Pile, RedPajama, Dolma). Talkie's corpus is not indexed there; the `wimbd` tool from llm-corpus-search can build a local index if needed. |
| NYT "On This Day" | NYT archive / Wayback | The Talkie team's historical-surprise dataset (~5,000 items). Requires scraping; the experiment runner should construct this on demand from the NYT archive feed, with the cleaned-up format `{date, text}`. |

---

## Notes on dataset sufficiency

- For the *core hypothesis* — "is Talkie a cleaner testbed for generalisation than current LLMs?" — the bare minimum needed is **HumanEval + Talkie-1930 + Talkie-web**. Everything else strengthens the evidence.
- MMLU + a date-extractor lets you partition questions by whether their answer depends on post-1930 knowledge — this is the single most informative experiment for *isolating* generalisation from knowledge access.
- The `post1930_probes` seed lets you test categories MMLU under-samples (neologisms, ICL).
