# Post-1930 Probe Dataset (Seed)

A small, curated seed dataset of probes for testing Talkie-1930's behaviour on
concepts, events, terminology, and programming languages that postdate its
1930 training cutoff. The intent is to be expanded by the experiment runner.

## Schema (one JSON object per line)

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `year` | int | Year the concept/event entered the world |
| `category` | string | `invention` / `event` / `discovery` / `scientific-concept` / `neologism` / `icl-python` / `icl-language` / `physics` / `mathematics` / `icl-physics-edge-case` |
| `concept` | string | Short description of the post-1930 thing |
| `prompt` | string | Text to give to the model |
| `expected_modern` | string | What a modern, knowledgeable model would complete with (used for *contrast*, not as ground truth for Talkie) |
| `kind` | string | `completion` (greedy/log-likelihood) or `icl` (in-context learning probe with worked example before the test item) |

## Intended uses

1. **Knowledge boundary check** — Talkie-1930 should be unable to produce
   the `expected_modern` completion for the `completion` items. Quantify with
   surprisal (bits/byte) and exact-match.
2. **In-context generalisation** — for `icl` items, Talkie has never seen
   Python/JavaScript syntax in training. Test whether a few in-context
   demonstrations are enough for it to extrapolate correctly. This is the
   *core generalisation probe*.
3. **Contrast with `talkie-web-13b-base`** — the same prompts on the modern
   twin establish the upper bound for what the architecture/FLOPs can do
   with modern training data.
4. **Per-year forgetting curve** — plot Talkie's per-token surprisal on
   expected completions as a function of the `year` field to recover a
   "knowledge cutoff" signature.

## Expansion plan (for experiment runner)

The experiment runner can extend this seed with auto-generated items derived from:

- **NYT "On This Day"** descriptions parsed by date (the Talkie team
  already used ~5,000 such items for their "historical surprise" plot).
- **MMLU subjects that are date-anchored**: many `_history`, `_world_religions`,
  `us_foreign_policy`, `astronomy`, `college_computer_science` items mention
  post-1930 facts. Filter MMLU questions by automated date-extraction.
- **HumanEval (164 items)**: every problem requires Python, which postdates 1930
  by ~60 years — the entire benchmark is an OOD generalisation probe.
- **Wikidata snapshots** filtered to entities/events with timestamps > 1930.

## Notes

- The `expected_modern` text is for *scoring contrast*; the canonical experimental
  measure is comparative (Talkie-1930 vs. Talkie-web), not absolute correctness.
- "Anachronistic" questions (e.g., asking Talkie who Hitler is) are the *easy* case
  for proving non-generalisation. The harder, more interesting case is whether
  in-context demonstrations let Talkie learn truly new content (e.g., Python).
- Talkie's blog reports their own HumanEval and NYT experiments — this seed file
  is intended to *augment* those with additional categories (neologisms, mathematical
  results, etc.) that future experiments can sweep over.
