"""Coarse-to-fine grid baseline: a Point question becomes two decider Choice questions.

Q1 picks one of n x n coarse cells over the frame; Q2, given a coarse cell, picks one of
n x n fine cells inside it (n = 15: 225 options each, about tile resolution on a Brood War map).
Training uses the true coarse cell in Q2 (teacher forcing). Inference runs two passes:
Q1, then Q2 for the predicted coarse cell.

The two answers define a piecewise-constant density over the normalized frame:
  density(u, v) = p(coarse) * p(fine | coarse) * n^4
so NLL is comparable with the mixture head's (same normalized space).
"""

from __future__ import annotations

import math

from decider.point import Frame, PointExample

from decider.data.core import Example, Q

N = 15


def cell_of(u: float, v: float, n: int = N) -> tuple[int, int]:
    """(row, col) of a normalized point; row follows y, col follows x. Edges clamp inward."""
    return min(n - 1, max(0, int(v * n))), min(n - 1, max(0, int(u * n)))


def labels(n: int = N) -> list:
    return [f"{r},{c}" for r in range(n) for c in range(n)]


def cell_bounds(frame: Frame, r: int, c: int, n: int = N, parent=None) -> tuple:
    """Frame-unit bounds (x_lo, x_hi, y_lo, y_hi) of coarse cell (r, c), or of fine cell (r, c) inside `parent`."""
    if parent is None:
        u0, v0, du, dv = 0.0, 0.0, 1 / n, 1 / n
    else:
        pr, pc = parent
        u0, v0, du, dv = pc / n, pr / n, 1 / n ** 2, 1 / n ** 2
    x_lo, y_lo = frame.denormalize(u0 + c * du, v0 + r * dv)
    x_hi, y_hi = frame.denormalize(u0 + (c + 1) * du, v0 + (r + 1) * dv)
    return x_lo, x_hi, y_lo, y_hi


def g(v: float) -> str:
    return f"{v:.4g}"


def coarse_question(text: str, frame: Frame, n: int = N) -> str:
    dx, dy = (frame.x1 - frame.x0) / n, (frame.y1 - frame.y0) / n
    return (f"{text}\nAnswer with a cell of a {n} by {n} grid over the frame ({frame.describe()}). "
            f"Cell \"row,col\" covers x from {g(frame.x0)} + col*{g(dx)} to {g(frame.x0)} + (col+1)*{g(dx)} and "
            f"y from {g(frame.y0)} + row*{g(dy)} to {g(frame.y0)} + (row+1)*{g(dy)}.")


def fine_question(text: str, frame: Frame, coarse: tuple, n: int = N) -> str:
    x_lo, x_hi, y_lo, y_hi = cell_bounds(frame, *coarse, n)
    return (f"{text}\nThe answer lies in the cell spanning x from {g(x_lo)} to {g(x_hi)} and y from {g(y_lo)} to {g(y_hi)}. "
            f"Answer with a cell of a {n} by {n} grid over that cell, \"row,col\" as before.")


def to_grid_example(e: PointExample, n: int = N) -> Example:
    """Training form: for each Point question, a coarse question and a teacher-forced fine question."""
    qs = []
    opts = labels(n)
    for q in e.qs:
        u, v = q.frame.normalize(*q.target)
        cr, cc = cell_of(u, v, n)
        fr, fc = cell_of((u * n - cc), (v * n - cr), n)
        qs.append(Q(coarse_question(q.text, q.frame, n), opts, cr * n + cc))
        qs.append(Q(fine_question(q.text, q.frame, (cr, cc), n), opts, fr * n + fc))
    return Example(e.context, qs, e.task)


def coarse_example(e: PointExample, n: int = N) -> Example:
    return Example(e.context, [Q(coarse_question(q.text, q.frame, n), labels(n), 0) for q in e.qs], e.task)


def fine_example(e: PointExample, coarse_cells: list, n: int = N) -> Example:
    return Example(e.context, [Q(fine_question(q.text, q.frame, cc, n), labels(n), 0)
                               for q, cc in zip(e.qs, coarse_cells)], e.task)


def fine_target(u: float, v: float, n: int = N) -> tuple:
    cr, cc = cell_of(u, v, n)
    return (cr, cc), cell_of(u * n - cc, v * n - cr, n)


def centre(coarse: tuple, fine: tuple, n: int = N) -> tuple[float, float]:
    """Normalized centre of fine cell `fine` inside coarse cell `coarse`."""
    (cr, cc), (fr, fc) = coarse, fine
    return (cc + (fc + 0.5) / n) / n, (cr + (fr + 0.5) / n) / n


def log_density(p_coarse: float, p_fine: float, n: int = N) -> float:
    return math.log(max(p_coarse, 1e-12)) + math.log(max(p_fine, 1e-12)) + 4 * math.log(n)
