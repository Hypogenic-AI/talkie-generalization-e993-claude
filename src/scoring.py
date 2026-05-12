"""Log-probability scoring helpers for Talkie models.

The upstream `TalkieModel.forward` returns logits at the last sequence position
only (used for sampling). For evaluation we need either:

* per-token log-likelihoods of a known completion conditioned on a prompt
  (used for MMLU letter-choice scoring and probe surprisal), or
* greedy completion of a prompt (used for HumanEval and probe top-1 EM).

This module wires both onto the existing model without modifying upstream
files: ``forward_all_logits`` re-implements the model's forward to return
``[B, S, V]`` logits, and the public helpers handle prompt encoding,
token-by-token reduction, and bf16/AMP context management.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal forward helpers
# ---------------------------------------------------------------------------

def _forward_all_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Mirror of ``TalkieModel.forward`` that returns ALL-position logits.

    Returns a ``[B, S, V]`` float32 tensor. Replicates the residual / RoPE
    logic from ``code/talkie/src/talkie/model.py`` exactly.
    """
    _, seq_len = input_ids.shape
    cos_sin = model.cos[:, :seq_len], model.sin[:, :seq_len]

    x = model.embed(input_ids)
    x = F.rms_norm(x, (x.shape[-1],))
    e_x = x
    for block in model.blocks:
        x = block(e_x, x, cos_sin)
    x = F.rms_norm(x, (x.shape[-1],))
    return F.linear(x, model.lm_head_gain(model.lm_head)).float()


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    """Outcome of scoring `completion` given `prompt`."""

    sum_logp: float          # total log-likelihood of the completion (nats)
    n_tokens: int            # number of completion tokens scored
    mean_logp: float         # sum_logp / n_tokens
    bits_per_token: float    # -mean_logp / ln(2)
    completion_tokens: list[int]


def score_completion(
    talkie,                 # talkie.Talkie instance
    prompt: str,
    completion: str,
    max_context: int = 1900,
) -> ScoreResult:
    """Score the log-likelihood of `completion` given `prompt`.

    Single forward pass over the concatenation. We extract the conditional
    log-prob of each completion token from the per-position logits.

    The prompt is truncated from the LEFT if the total length exceeds
    ``max_context`` (the model's RoPE buffer is 2048 by default).
    """
    tokenizer = talkie.tokenizer
    model = talkie.model
    device = talkie.device

    p_tokens = tokenizer.encode(prompt, allowed_special="all")
    c_tokens = tokenizer.encode(completion, allowed_special="all")

    # Left-truncate the prompt if necessary so that the full sequence fits.
    full = p_tokens + c_tokens
    if len(full) > max_context:
        excess = len(full) - max_context
        p_tokens = p_tokens[excess:]
        full = p_tokens + c_tokens

    if not c_tokens:
        return ScoreResult(0.0, 0, 0.0, 0.0, [])

    x = torch.tensor(full, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad(), talkie._autocast:
        logits_all = _forward_all_logits(model, x)  # [1, S, V]

    # log P(c_i | prompt + c_<i) is at position (len(p) - 1 + i) of the
    # logits tensor (logits[t] predicts token t+1).
    n_p = len(p_tokens)
    # Logits used to predict each completion token:
    pred_logits = logits_all[0, n_p - 1 : n_p - 1 + len(c_tokens)]
    log_probs_full = F.log_softmax(pred_logits, dim=-1)  # [n_c, V]
    target = torch.tensor(c_tokens, dtype=torch.long, device=device)
    token_logps = log_probs_full.gather(1, target.unsqueeze(1)).squeeze(1)  # [n_c]
    sum_lp = float(token_logps.sum().item())

    return ScoreResult(
        sum_logp=sum_lp,
        n_tokens=len(c_tokens),
        mean_logp=sum_lp / len(c_tokens),
        bits_per_token=-sum_lp / (len(c_tokens) * math.log(2.0)),
        completion_tokens=c_tokens,
    )


def score_choices(
    talkie,
    prompt: str,
    choices: list[str],
    max_context: int = 1900,
) -> list[ScoreResult]:
    """Score each candidate completion in `choices` against the same prompt."""
    return [score_completion(talkie, prompt, c, max_context=max_context) for c in choices]


def score_single_token_choices(
    talkie,
    prompt: str,
    choice_strings: list[str],
    max_context: int = 1900,
) -> list[float]:
    """Fast-path: score single-token choices with ONE forward pass.

    Each ``choice_strings[i]`` is encoded with the tokenizer; we take the
    *first* token of each encoding and read its log-probability from the
    model's next-token distribution at the end of ``prompt``. This is
    correct for MMLU letter-choice scoring where each choice (" A", " B",
    " C", " D") tokenises to a single token in the Talkie tokenizer.

    Returns a list of float log-probabilities, one per choice.
    """
    tokenizer = talkie.tokenizer
    model = talkie.model
    device = talkie.device

    p_tokens = tokenizer.encode(prompt, allowed_special="all")
    if len(p_tokens) > max_context:
        p_tokens = p_tokens[-max_context:]

    # First token of each choice.
    first_ids: list[int] = []
    for c in choice_strings:
        ids = tokenizer.encode(c, allowed_special="all")
        if not ids:
            raise ValueError(f"empty token encoding for choice {c!r}")
        first_ids.append(ids[0])

    x = torch.tensor(p_tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad(), talkie._autocast:
        last_logits = model.forward(x)[0]  # [V]
    log_probs = F.log_softmax(last_logits.float(), dim=-1)
    return [float(log_probs[i].item()) for i in first_ids]


# ---------------------------------------------------------------------------
# Greedy completion (for HumanEval, free-form probe responses)
# ---------------------------------------------------------------------------

def greedy_complete(
    talkie,
    prompt: str,
    max_tokens: int = 256,
    stop_strings: Iterable[str] | tuple[str, ...] = ("\n\n\n",),
    max_context: int = 1900,
) -> str:
    """Deterministic greedy decoding (temperature=0 equivalent).

    The native :py:meth:`Talkie.generate` always adds Gumbel noise, so we
    bypass it for byte-exact reproducibility.
    """
    tokenizer = talkie.tokenizer
    model = talkie.model
    device = talkie.device
    stop_ids = talkie._stop_ids

    p_tokens = tokenizer.encode(prompt, allowed_special="all")
    if len(p_tokens) >= max_context:
        p_tokens = p_tokens[-max_context:]

    tokens_tensor = torch.tensor(p_tokens, dtype=torch.long, device=device).unsqueeze(0)
    out_tokens: list[int] = []
    out_text = ""

    stop_strings = tuple(s for s in stop_strings if s)

    with torch.no_grad(), talkie._autocast:
        for _ in range(max_tokens):
            logits = model.forward(tokens_tensor)  # [1, V]
            next_id = int(torch.argmax(logits[0]).item())
            if next_id in stop_ids:
                break
            tokens_tensor = torch.cat(
                [tokens_tensor,
                 torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )
            out_tokens.append(next_id)
            # Decoding incrementally is fine for tiktoken.
            out_text = tokenizer.decode(out_tokens)
            if stop_strings:
                # Match either as substring OR as line-prefix at start of output
                # (catches the case where the model regenerates `def foo():`
                # immediately, with no leading newline).
                cuts = []
                for s in stop_strings:
                    if s in out_text:
                        cuts.append(out_text.find(s))
                    if s.startswith("\n") and out_text.startswith(s[1:]):
                        cuts.append(0)
                if cuts:
                    cut = min(cuts)
                    out_text = out_text[:cut]
                    break
            if tokens_tensor.shape[1] >= 2048:
                break

    return out_text
