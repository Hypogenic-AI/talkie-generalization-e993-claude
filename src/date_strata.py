"""Heuristic date-stratification of MMLU items.

Given an MMLU question + 4 choices we tag it with a "decidability stratum":

* ``post1930`` — the answer demonstrably depends on a fact, technology,
  vocabulary item, or person that postdates 1930.
* ``pre1930``  — the question and all choices are framed entirely in
  pre-1930 vocabulary; the answer would in principle be derivable from a
  pre-1930 corpus.
* ``unknown``  — neither test fires (the majority).

We use a conservative two-pass detector:

1. **Year tagger.** Any 4-digit number ``19xx`` with xx > 30 or ``20xx`` /
   ``21xx`` flags ``post1930``. Any 4-digit ``17xx`` / ``18xx`` / ``19xx``
   with xx ≤ 30 contributes a pre-1930 hint, but only ``pre1930`` *if* no
   post-1930 hits exist.
2. **Keyword tagger.** A small whitelist of post-1930 entities
   (technologies, organisations, scientific concepts, neologisms) flags
   ``post1930``. This catches items like "What is the maximum theoretical
   throughput of a 100-Gigabit Ethernet link?" that have no year token.

The detector is intentionally conservative — items it cannot decide are
labelled ``unknown`` and excluded from the stratified comparison. This keeps
the partition CLEAN at the cost of throwing away items.

We do *not* ask Talkie/GPT-4.1 to do the dating: that would entangle the
detector with the models we're evaluating. Pure-Python regex is fine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Year regex — matches isolated 4-digit years 1500-2099.
# ---------------------------------------------------------------------------
# Year regex with negative lookarounds to skip numeric values like "0.1665"
# (decimal-context) or "1665.5" (continued number) that are not actual years.
# Also require either start-of-string or whitespace/punct (NOT digit, decimal,
# percent) immediately before, and end-of-string / non-digit / non-decimal /
# non-% after.
_YEAR_RE = re.compile(
    r"(?<![\d.,%])"                                     # not after digit / decimal / %
    r"(1[5-9][0-9]{2}|20[0-9]{2}|21[0-9]{2})"
    r"(?![\d.,%])"                                      # not before digit / decimal / %
)

# ---------------------------------------------------------------------------
# Post-1930 keyword list. Each is a hand-picked unambiguously post-1930 term.
# Match is case-insensitive and word-boundary aware. Keep terms short and
# unambiguous; do NOT add anything that has a pre-1930 sense in English (e.g.,
# "computer" — used pre-1930 to mean "a person who computes").
# ---------------------------------------------------------------------------
POST1930_KEYWORDS: tuple[str, ...] = (
    # technology / computing
    "internet", "world wide web", "wi-fi", "wifi", "smartphone", "iphone",
    "android", "tcp/ip", "ipv4", "ipv6", "ethernet", "javascript", "python",
    "linux", "unix", "microsoft", "google", "facebook", "twitter", "youtube",
    "amazon.com", "blockchain", "bitcoin",
    "machine learning", "deep learning", "neural network",
    "artificial intelligence", "transistor", "semiconductor", "microprocessor",
    # physics / chemistry / biology
    "double helix", "quark", "gluon", "higgs boson",
    "neutron star", "black hole", "big bang", "quasar", "pulsar",
    "polio vaccine", "covid", "covid-19", "sars-cov", "hiv/aids", "crispr",
    "human genome project",
    "stem cell",
    # politics / events / organisations
    "united nations", "european union", "world war ii", "world war 2",
    "world war two", "cold war", "vietnam war", "korean war",
    "great depression", "manhattan project", "moon landing", "apollo 11",
    "soviet union", "ussr", "berlin wall", "9/11", "september 11",
    "fall of the berlin wall", "9-11",
    # culture / neologisms
    "rock and roll", "rock 'n' roll", "hip hop",
    "spaceflight",
    "atomic bomb", "nuclear weapon", "hydrogen bomb",
    "large hadron collider",
    # corporate / brand
    "spacex", "openai", "anthropic",
)
# All-uppercase tokens that are unambiguous post-1930 acronyms.
# These are matched CASE-SENSITIVELY because lowercase variants overlap with
# common English words ("nato"/"NATO", "aids"/"AIDS", etc.).
POST1930_UPPER_ACRONYMS: tuple[str, ...] = (
    "DNA", "RNA", "AIDS", "HIV", "NATO", "USSR", "CERN", "LHC", "NASA",
)
_KEYWORD_RES = [re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE) for k in POST1930_KEYWORDS]
_UPPER_RES = [re.compile(rf"\b{re.escape(k)}\b") for k in POST1930_UPPER_ACRONYMS]


@dataclass
class DateTag:
    stratum: str           # "post1930" | "pre1930" | "unknown"
    triggers: list[str]    # textual triggers that fired the rule


def tag_text(text: str, year_text: str | None = None) -> DateTag:
    """Tag a text. Year detection is restricted to ``year_text`` if provided
    (otherwise to ``text``). Keyword detection always uses ``text``.

    Restricting year detection to (typically) the question text avoids false
    positives from numeric *answer-choice* values like ``1800`` in arithmetic
    items.
    """
    if year_text is None:
        year_text = text
    triggers: list[str] = []

    post_year = False
    pre_year = False
    for m in _YEAR_RE.finditer(year_text):
        y = int(m.group(0))
        if y > 1930:
            post_year = True
            triggers.append(f"year:{y}")
        elif y <= 1930:
            pre_year = True
            triggers.append(f"year:{y}")

    # Keyword triggers (case-insensitive whitelist).
    keyword_hit = False
    for k, r in zip(POST1930_KEYWORDS, _KEYWORD_RES):
        if r.search(text):
            keyword_hit = True
            triggers.append(f"kw:{k}")
    # Upper-case acronym triggers.
    for k, r in zip(POST1930_UPPER_ACRONYMS, _UPPER_RES):
        if r.search(text):
            keyword_hit = True
            triggers.append(f"acr:{k}")

    if post_year or keyword_hit:
        return DateTag("post1930", triggers)
    if pre_year:
        return DateTag("pre1930", triggers)
    return DateTag("unknown", triggers)


def tag_mmlu_item(question: str, choices: list[str]) -> DateTag:
    """Tag an MMLU item.

    Years are extracted from the question only (so that numeric values in
    answer choices don't create year false-positives). Keywords are matched
    against question + all choices.
    """
    full = question + " " + " ".join(choices)
    return tag_text(full, year_text=question)
