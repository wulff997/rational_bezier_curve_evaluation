import math
from typing import Sequence, List


def evaluate_rational_bezier(
    points: Sequence[Sequence[float]],
    weights: Sequence[float],
    t: float,
) -> List[float]:
    """Evaluate a rational Bezier curve at parameter t via De Casteljau.

    The numerator (weighted control point) and denominator (weight) are
    interpolated separately, so a zero weight at a *non-endpoint* control
    point is handled correctly: the denominator and numerator pass through
    the same interpolated zero weight, and the math remains consistent for
    the curve segments that are defined. A zero weight at an endpoint t=0 or
    t=1 propagates a 0/0, which we resolve to the finite, weighted control
    point at the other end of the collapsing segment.
    """
    if not points:
        raise ValueError("points must be a non-empty sequence")
    n = len(points)
    if len(weights) != n:
        raise ValueError("weights length must match points length")
    if n == 1:
        return [float(v) for v in points[0]]

    # Separate the de Casteljau passes so a zero weight (non-endpoint)
    # naturally produces 0/0 -> resolved to the surviving control point.
    num: List[List[float]] = [
        [float(v) * float(weights[i]) for v in points[i]] for i in range(n)
    ]
    den: List[float] = [float(weights[i]) for i in range(n)]

    for level in range(1, n):
        size = n - level
        for i in range(size):
            num[i] = [
                (1.0 - t) * num[i][j] + t * num[i + 1][j]
                for j in range(len(num[i]))
            ]
            den[i] = (1.0 - t) * den[i] + t * den[i + 1]

    d = den[0]
    if d == 0.0:
        # 0/0 collapse: return the surviving (non-zero weight) control point.
        for i in range(n):
            if float(weights[i]) != 0.0:
                return [float(v) for v in points[i]]
        return [0.0] * len(points[0])
    return [v / d for v in num[0]]


def evaluate_nurbs_basis(
    degree: int,
    knot_vector: Sequence[float],
    t: float,
) -> List[float]:
    """Evaluate the (rational) B-spline basis functions of a given degree.

    Returns the n_basis non-zero basis values, where n_basis = degree + 1.
    The caller is responsible for identifying the corresponding control
    points using the returned knot span index via :func:`find_span`.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    n = len(knot_vector) - degree - 1
    if n < 1:
        raise ValueError(
            "knot_vector too short: need at least degree+2 knots"
        )

    span = find_span(degree, knot_vector, n, t)
    return basis_funs(span, t, degree, knot_vector)


def find_span(
    degree: int,
    knot_vector: Sequence[float],
    n_points: int,
    t: float,
) -> int:
    """Return the knot span index for parameter t.

    Handles the t == last-knot case by clamping to n_points - 1, matching the
    convention used by Piegl & Tiller in *The NURBS Book*.
    """
    if t >= knot_vector[n_points]:
        return n_points - 1
    if t <= knot_vector[degree]:
        return degree
    low = degree
    high = n_points
    while high - low > 1:
        mid = (low + high) // 2
        if t < knot_vector[mid]:
            high = mid
        else:
            low = mid
    return low


def basis_funs(
    span: int,
    t: float,
    degree: int,
    knot_vector: Sequence[float],
) -> List[float]:
    """Compute the non-zero B-spline basis functions at t.

    The array length is always degree + 1.
    """
    N: List[float] = [0.0] * (degree + 1)
    left: List[float] = [0.0] * (degree + 1)
    right: List[float] = [0.0] * (degree + 1)
    N[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = t - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - t
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            if denom == 0.0:
                # Repeated knot: avoid division by zero; one of the
                # basis functions at this collapse is undefined but
                # the limit takes the other coefficient unmodified.
                continue
            tmp = N[r] / denom
            N[r] = saved + right[r + 1] * tmp
            saved = left[j - r] * tmp
        N[j] = saved
    return N
