# rational_bezier_curve_evaluation

Evaluates non-uniform rational B-spline and rational Bezier curves using De Casteljau's algorithm. Pure Python, standard library only.

## Usage

```python
from rational_bezier_curve_evaluation import evaluate_rational_bezier, evaluate_nurbs_basis

# Rational Bezier curve: two control points, weights
pts = [(0.0, 0.0), (2.0, 4.0)]
wts = [1.0, 1.0]
x, y = evaluate_rational_bezier(pts, wts, 0.5)  # -> [1.0, 2.0]

# NURBS basis functions for a clamped quadratic knot vector
knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
N = evaluate_nurbs_basis(2, knots, 0.5)  # -> [0.25, 0.5, 0.25]
```

## Why this exists

This library exists to evaluate rational Bezier and NURBS curves without pulling in a numeric or geometry stack. The trade-off is precision and expressiveness: we use plain Python `float` arithmetic, so callers who need CAD-grade robustness against cancellation should wrap these calls in their own error tracking. The De Casteljau split keeps the code short and the edge cases local.

## Exported names

- `evaluate_rational_bezier(points, weights, t) -> list[float]`
- `evaluate_nurbs_basis(degree, knot_vector, t) -> list[float]`

## Edge cases you will hit

A zero weight at an endpoint produces a `0/0` collapse. The implementation resolves this by returning the surviving (finite) weighted control point, so a curve that starts or ends at a zero-weight control point clamps to the opposite control point rather than returning `NaN`. Zero weights at interior control points are handled naturally by the separate numerator/denominator passes. Callers who depend on a different convention for singular weights should post-process the result themselves.
