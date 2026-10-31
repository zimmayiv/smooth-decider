import math
import random

from decider.grid import N, cell_bounds, cell_of, centre, fine_target, log_density, to_grid_example
from decider.point import Frame, PointExample, PointQ


def test_uniform_guess_has_density_one():
    assert math.isclose(log_density(1 / N ** 2, 1 / N ** 2), 0.0, abs_tol=1e-12)


def test_fine_cell_centre_is_within_one_fine_cell_of_the_target():
    rng = random.Random(0)
    for _ in range(2000):
        u, v = rng.random(), rng.random()
        coarse, fine = fine_target(u, v)
        cu, cv = centre(coarse, fine)
        assert abs(cu - u) <= 0.5 / N ** 2 + 1e-12 and abs(cv - v) <= 0.5 / N ** 2 + 1e-12


def test_grid_example_golds_match_the_target_cells_and_bounds():
    f = Frame(0, 8192, 0, 4096, "pixels", y_down=True)
    e = PointExample("ctx", [PointQ("Go there.", f, (5000.0, 1000.0))], "t")
    g = to_grid_example(e)
    (cr, cc), (fr, fc) = fine_target(*f.normalize(5000.0, 1000.0))
    assert g.qs[0].gold == cr * N + cc and g.qs[1].gold == fr * N + fc
    x_lo, x_hi, y_lo, y_hi = cell_bounds(f, fr, fc, parent=(cr, cc))
    assert x_lo <= 5000 <= x_hi and y_lo <= 1000 <= y_hi
    assert f"x from {cell_bounds(f, cr, cc)[0]:.4g}" in g.qs[1].text
    assert len(g.qs[0].options) == N * N and cell_of(1.0, 1.0) == (N - 1, N - 1)
