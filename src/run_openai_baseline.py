"""Baseline GPT-4.1 via OpenAI API.

Runs the same probe + MMLU-subsample evaluations as the Talkie scripts but
using the OpenAI Chat Completions API. Used for the meta-attribution
analysis: GPT-4.1 has unknown training cutoff/composition, so we treat
its outputs as a "frontier modern LLM, indistinguishable memorisation /
generalisation" baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.date_strata import tag_mmlu_item
from src.datasets_io import load_mmlu, stratified_subsample_mmlu, load_probes


_LETTERS = ["A", "B", "C", "D"]
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def call_openai(messages, model="gpt-4.1", max_tokens=128, temperature=0.0,
                logprobs=False, top_logprobs=0):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    # gpt-4.1 supports temperature 0; gpt-5 does not. Add only if supported.
    if not model.startswith("gpt-5"):
        payload["temperature"] = temperature
    if logprobs:
        payload["logprobs"] = True
        if top_logprobs:
            payload["top_logprobs"] = top_logprobs
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    for attempt in range(4):
        try:
            r = requests.post(_OPENAI_URL, headers=headers, json=payload,
                              timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt
                print(f"  HTTP {r.status_code}, sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:200]}")
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f"  request error {e}, sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("OpenAI API failed after retries")


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def run_probes(model: str, output: Path):
    probes = load_probes()
    print(f"[probes-openai] running {len(probes)} probes on {model}", flush=True)

    records: list[dict] = []
    t_start = time.time()
    for i, p in enumerate(probes):
        msg = [
            {"role": "system",
             "content": "You are a text-completion engine. "
                        "Continue the user's text naturally and minimally; "
                        "give just the completion, no preamble."},
            {"role": "user", "content": p.prompt},
        ]
        try:
            resp = call_openai(msg, model=model, max_tokens=64)
            completion = resp["choices"][0]["message"]["content"]
        except Exception as e:
            completion = ""
            err = f"{type(e).__name__}: {e}"
        else:
            err = None

        em_norm = p.expected_modern.strip().lower()
        gen_norm = completion.strip().lower()
        contains_expected = em_norm in gen_norm

        records.append({
            "id": p.id, "year": p.year, "category": p.category,
            "kind": p.kind, "concept": p.concept,
            "prompt": p.prompt, "expected_modern": p.expected_modern,
            "completion": completion, "error": err,
            "contains_expected": contains_expected,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(probes)}] elapsed={time.time()-t_start:.0f}s",
                  flush=True)

    # Aggregate
    def agg(filt):
        s = [r for r in records if filt(r)]
        if not s:
            return {"n": 0}
        em = sum(1 for r in s if r["contains_expected"])
        return {"n": len(s), "n_em": em, "em_rate": em / len(s)}

    summary = {
        "model": model,
        "n_probes": len(records),
        "wall_seconds": time.time() - t_start,
        "agg_post1930_completion": agg(
            lambda r: r["year"] >= 1930 and r["kind"] == "completion"),
        "agg_pre1930_completion":  agg(
            lambda r: r["year"] < 1930 and r["kind"] == "completion"),
        "agg_icl":                 agg(lambda r: r["kind"] == "icl"),
    }
    out = {"summary": summary, "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

def run_mmlu(model: str, output: Path, n_per_subject: int, seed: int):
    items = load_mmlu()
    sub = stratified_subsample_mmlu(items, n_per_subject, seed=seed)
    print(f"[mmlu-openai] running {len(sub)} items on {model}", flush=True)

    records: list[dict] = []
    n_correct = 0
    t_start = time.time()
    for i, it in enumerate(sub):
        choices_text = "\n".join(f"{l}. {c}" for l, c in zip(_LETTERS, it.choices))
        user = (f"Question: {it.question}\n{choices_text}\n\n"
                "Reply with the single letter (A, B, C, or D) of the best answer.")
        msg = [{"role": "user", "content": user}]
        try:
            resp = call_openai(msg, model=model, max_tokens=8)
            txt = resp["choices"][0]["message"]["content"].strip()
            # Pick the first A/B/C/D character.
            pred = next((c for c in txt if c.upper() in _LETTERS), None)
            pred = pred.upper() if pred else None
            err = None
        except Exception as e:
            pred = None
            err = f"{type(e).__name__}: {e}"
            txt = ""

        tag = tag_mmlu_item(it.question, it.choices)
        rec = {
            "subject": it.subject,
            "question": it.question,
            "choices": it.choices,
            "answer": it.answer,
            "predicted": pred,
            "raw_response": txt,
            "correct": (pred == it.answer),
            "stratum": tag.stratum,
            "stratum_triggers": tag.triggers[:5],
            "error": err,
        }
        records.append(rec)
        if rec["correct"]:
            n_correct += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(sub)}] acc={n_correct/(i+1):.3f} "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)

    def acc(filt):
        s = [r for r in records if filt(r)]
        if not s: return {"n": 0}
        c = sum(1 for r in s if r["correct"])
        return {"n": len(s), "acc": c / len(s)}

    summary = {
        "model": model,
        "n_items": len(records),
        "overall_acc": n_correct / len(records),
        "wall_seconds": time.time() - t_start,
        "stratum_post1930": acc(lambda r: r["stratum"] == "post1930"),
        "stratum_pre1930":  acc(lambda r: r["stratum"] == "pre1930"),
        "stratum_unknown":  acc(lambda r: r["stratum"] == "unknown"),
    }
    out = {"summary": summary, "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["probes", "mmlu"])
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_per_subject", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.output)
    if args.task == "probes":
        run_probes(args.model, out)
    else:
        run_mmlu(args.model, out, args.n_per_subject, args.seed)


if __name__ == "__main__":
    main()
