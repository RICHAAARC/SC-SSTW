"""Small linear algebra helpers for 2D-state feasibility tests."""

from __future__ import annotations

import math


Vector = list[float]
Matrix = list[list[float]]


def transpose(matrix: Matrix) -> Matrix:
    return [list(row) for row in zip(*matrix, strict=True)]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    right_t = transpose(right)
    return [[dot(row, col) for col in right_t] for row in left]


def matvec(matrix: Matrix, vector: Vector) -> Vector:
    return [dot(row, vector) for row in matrix]


def dot(left: Vector | tuple[float, ...], right: Vector | tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def add(left: Vector, right: Vector) -> Vector:
    return [a + b for a, b in zip(left, right, strict=True)]


def sub(left: Vector, right: Vector) -> Vector:
    return [a - b for a, b in zip(left, right, strict=True)]


def scale(value: float, vector: Vector) -> Vector:
    return [value * item for item in vector]


def squared_norm(vector: Vector | tuple[float, ...]) -> float:
    return dot(vector, vector)


def invert_2x2(matrix: Matrix, *, ridge: float = 0.0) -> Matrix:
    a = matrix[0][0] + ridge
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1] + ridge
    det = a * d - b * c
    if abs(det) < 1e-12:
        raise ValueError("singular 2x2 matrix")
    return [[d / det, -b / det], [-c / det, a / det]]


def least_squares(features: Matrix, targets: Matrix, *, ridge: float = 1e-8) -> Matrix:
    """Solve min ||features * beta - targets|| with small ridge.

    Returns beta with shape feature_count x target_count.
    """

    xt = transpose(features)
    xtx = matmul(xt, features)
    for index in range(len(xtx)):
        xtx[index][index] += ridge
    xty = matmul(xt, targets)
    inverse = invert_3x3(xtx)
    return matmul(inverse, xty)


def invert_3x3(matrix: Matrix) -> Matrix:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    cofactors = [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ]
    det = a * cofactors[0][0] + b * cofactors[1][0] + c * cofactors[2][0]
    if abs(det) < 1e-12:
        raise ValueError("singular 3x3 matrix")
    return [[value / det for value in row] for row in cofactors]


def condition_number_2d_columns(matrix: Matrix) -> float:
    """Condition number of A with two columns via eigenvalues of A^T A."""

    ata = matmul(transpose(matrix), matrix)
    trace = ata[0][0] + ata[1][1]
    det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0]
    disc = max(0.0, trace * trace - 4.0 * det)
    lambda_max = 0.5 * (trace + math.sqrt(disc))
    lambda_min = 0.5 * (trace - math.sqrt(disc))
    if lambda_min <= 0.0:
        return math.inf
    return math.sqrt(lambda_max / lambda_min)
