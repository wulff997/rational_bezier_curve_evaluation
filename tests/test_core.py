import math
import unittest

from rational_bezier_curve_evaluation.core import (
    evaluate_rational_bezier,
    evaluate_nurbs_basis,
    find_span,
    basis_funs,
)


class TestRationalBezier(unittest.TestCase):
    def _close(self, a, b, tol=1e-9):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertLessEqual(abs(x - y), tol)

    def test_linear_curve_endpoint0(self):
        pts = [(0.0, 0.0), (2.0, 4.0)]
        w = [1.0, 1.0]
        self._close(evaluate_rational_bezier(pts, w, 0.0), [0.0, 0.0])

    def test_linear_curve_endpoint1(self):
        pts = [(0.0, 0.0), (2.0, 4.0)]
        w = [1.0, 1.0]
        self._close(evaluate_rational_bezier(pts, w, 1.0), [2.0, 4.0])

    def test_linear_curve_midpoint(self):
        pts = [(0.0, 0.0), (2.0, 4.0)]
        w = [1.0, 1.0]
        self._close(evaluate_rational_bezier(pts, w, 0.5), [1.0, 2.0])

    def test_weight_reduces_to_start(self):
        pts = [(0.0, 0.0), (2.0, 4.0)]
        w = [10.0, 1.0]
        # Standard rational Bezier at t=0.5: numerator/denominator gives
        # (0.5*w0*P0 + 0.5*w1*P1) / (0.5*w0 + 0.5*w1) = [1,2]/5.5.
        self._close(evaluate_rational_bezier(pts, w, 0.5), [1.0/5.5, 2.0/5.5])

    def test_weights_equal_one_matches_bezier(self):
        pts = [(0.0, 0.0), (1.0, 2.0), (3.0, 0.0)]
        w = [1.0, 1.0, 1.0]
        t = 0.3
        bx = (1 - t) ** 2 * pts[0][0] + 2 * (1 - t) * t * pts[1][0] + t ** 2 * pts[2][0]
        by = (1 - t) ** 2 * pts[0][1] + 2 * (1 - t) * t * pts[1][1] + t ** 2 * pts[2][1]
        self._close(evaluate_rational_bezier(pts, w, t), [bx, by])

    def test_single_control_point(self):
        self._close(evaluate_rational_bezier([(5.0, 7.0)], [2.0], 0.5), [5.0, 7.0])

    def test_zero_weight_at_end_clamps_to_other_endpoint(self):
        pts = [(0.0, 0.0), (10.0, 0.0)]
        w = [0.0, 1.0]
        self._close(evaluate_rational_bezier(pts, w, 0.0), [10.0, 0.0])
        self._close(evaluate_rational_bezier(pts, w, 0.5), [10.0, 0.0])
        self._close(evaluate_rational_bezier(pts, w, 1.0), [10.0, 0.0])

    def test_zero_weight_at_t_one_clamps_to_start(self):
        pts = [(0.0, 0.0), (10.0, 0.0)]
        w = [1.0, 0.0]
        self._close(evaluate_rational_bezier(pts, w, 1.0), [0.0, 0.0])

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            evaluate_rational_bezier([(0.0, 0.0), (1.0, 1.0)], [1.0], 0.5)

    def test_empty_points_raises(self):
        with self.assertRaises(ValueError):
            evaluate_rational_bezier([], [], 0.5)

    def test_returns_list_of_floats(self):
        result = evaluate_rational_bezier([(0.0, 1.0)], [1.0], 0.0)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(v, float) for v in result))


class TestNurbsBasis(unittest.TestCase):
    def _close(self, a, b, tol=1e-9):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertLessEqual(abs(x - y), tol)

    def test_linear_basis_at_start(self):
        knots = [0.0, 0.0, 1.0, 1.0]
        N = evaluate_nurbs_basis(1, knots, 0.0)
        self._close(N, [1.0, 0.0])

    def test_linear_basis_at_end(self):
        knots = [0.0, 0.0, 1.0, 1.0]
        N = evaluate_nurbs_basis(1, knots, 1.0)
        self._close(N, [0.0, 1.0])

    def test_linear_basis_at_midpoint(self):
        knots = [0.0, 0.0, 1.0, 1.0]
        N = evaluate_nurbs_basis(1, knots, 0.5)
        self._close(N, [0.5, 0.5])

    def test_quadratic_basis_at_midpoint(self):
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        N = evaluate_nurbs_basis(2, knots, 0.5)
        self._close(N, [0.25, 0.5, 0.25])

    def test_basis_partition_of_unity(self):
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            N = evaluate_nurbs_basis(2, knots, t)
            self._close([sum(N)], [1.0])

    def test_invalid_degree_raises(self):
        with self.assertRaises(ValueError):
            evaluate_nurbs_basis(-1, [0.0, 1.0], 0.5)

    def test_knot_vector_too_short_raises(self):
        with self.assertRaises(ValueError):
            evaluate_nurbs_basis(2, [0.0, 1.0], 0.5)

    def test_find_span_clamps_to_last(self):
        knots = [0.0, 0.0, 1.0, 1.0]
        self.assertEqual(find_span(1, knots, 2, 1.0), 1)

    def test_find_span_clamps_to_first(self):
        knots = [0.0, 0.0, 1.0, 1.0]
        self.assertEqual(find_span(1, knots, 2, 0.0), 1)


if __name__ == "__main__":
    unittest.main()
