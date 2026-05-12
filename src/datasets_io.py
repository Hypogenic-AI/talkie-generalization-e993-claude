"""Loaders for the three datasets used in this study."""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------

@dataclass
class HumanEvalItem:
    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    test: str


def load_humaneval(path: Path | None = None) -> list[HumanEvalItem]:
    if path is None:
        path = REPO / "datasets" / "HumanEval.jsonl"
    out: list[HumanEvalItem] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(HumanEvalItem(
                task_id=d["task_id"],
                prompt=d["prompt"],
                entry_point=d["entry_point"],
                canonical_solution=d["canonical_solution"],
                test=d["test"],
            ))
    return out


# ---------------------------------------------------------------------------
# Post-1930 probes
# ---------------------------------------------------------------------------

@dataclass
class Probe:
    id: str
    year: int
    category: str
    concept: str
    prompt: str
    expected_modern: str
    kind: str  # "completion" or "icl"


def load_probes(path: Path | None = None) -> list[Probe]:
    if path is None:
        path = REPO / "datasets" / "post1930_probes" / "probes.jsonl"
        if not path.exists():
            path = REPO / "datasets" / "post1930_probes" / "seed_probes.jsonl"
    out: list[Probe] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(Probe(
                id=d["id"], year=int(d["year"]), category=d["category"],
                concept=d["concept"], prompt=d["prompt"],
                expected_modern=d["expected_modern"], kind=d["kind"],
            ))
    return out


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

@dataclass
class MMLUItem:
    subject: str
    question: str
    choices: list[str]   # length 4: [A, B, C, D]
    answer: str          # one of "A", "B", "C", "D"


_LETTERS = ["A", "B", "C", "D"]


def load_mmlu(test_dir: Path | None = None) -> list[MMLUItem]:
    if test_dir is None:
        test_dir = REPO / "datasets" / "mmlu" / "data" / "test"
    out: list[MMLUItem] = []
    for csv_path in sorted(test_dir.glob("*_test.csv")):
        subject = csv_path.stem.replace("_test", "")
        with csv_path.open() as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                q, a, b, c, d, ans = row[:6]
                ans = ans.strip().upper()
                if ans not in _LETTERS:
                    continue
                out.append(MMLUItem(
                    subject=subject,
                    question=q,
                    choices=[a, b, c, d],
                    answer=ans,
                ))
    return out


def stratified_subsample_mmlu(
    items: list[MMLUItem], n_per_subject: int, seed: int = 0
) -> list[MMLUItem]:
    """Take up to n_per_subject items per subject (deterministic by seed)."""
    rng = random.Random(seed)
    by_subject: dict[str, list[MMLUItem]] = {}
    for it in items:
        by_subject.setdefault(it.subject, []).append(it)
    subset: list[MMLUItem] = []
    for subject, lst in sorted(by_subject.items()):
        rng.shuffle(lst)
        subset.extend(lst[:n_per_subject])
    return subset


# ---------------------------------------------------------------------------
# Convenience: probe expansion (used in build_probes.py)
# ---------------------------------------------------------------------------

def write_probes(items: list[Probe], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in items:
            f.write(json.dumps({
                "id": p.id, "year": p.year, "category": p.category,
                "concept": p.concept, "prompt": p.prompt,
                "expected_modern": p.expected_modern, "kind": p.kind,
            }) + "\n")
