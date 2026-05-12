"""Safe-ish Python sandbox for grading HumanEval completions.

Runs ``code + test`` (a `check(candidate)` invocation) in a separate process
under a wall-clock timeout. Failure modes — syntax errors, exceptions inside
``check``, timeouts — all map to ``passed=False`` with a recorded reason.

We use ``multiprocessing`` so the parent process is unaffected by SIGKILL on
timeout, and we redirect stdout/stderr inside the child to keep noise out of
the eval log.
"""

from __future__ import annotations

import io
import contextlib
import multiprocessing as mp
from dataclasses import dataclass


@dataclass
class GradeResult:
    passed: bool
    reason: str


def _worker(code: str, conn) -> None:
    """Execute *code* (function definition + check call) in a child process."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            globs: dict = {"__name__": "__main__"}
            exec(code, globs)
        conn.send((True, "ok"))
    except BaseException as e:  # noqa: BLE001 — we want everything
        conn.send((False, f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def grade_humaneval(prompt: str, completion: str, test: str,
                    entry_point: str, timeout: float = 5.0) -> GradeResult:
    """Run the HumanEval ``check(candidate)`` against the model's completion.

    The standard HumanEval pattern is to concatenate the prompt (which ends
    mid-function-body or with a ``def`` header) with the model's completion.
    Then we append the test code, then ``check(<entry_point>)``.
    """
    full_code = (
        prompt + completion
        + "\n" + test
        + f"\ncheck({entry_point})\n"
    )

    parent_conn, child_conn = mp.Pipe(duplex=False)
    ctx = mp.get_context("fork")
    p = ctx.Process(target=_worker, args=(full_code, child_conn))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        if p.is_alive():
            p.kill()
            p.join(1.0)
        return GradeResult(False, "TIMEOUT")
    if not parent_conn.poll(timeout=0.5):
        return GradeResult(False, "NO_OUTPUT")
    ok, msg = parent_conn.recv()
    return GradeResult(ok, msg)
