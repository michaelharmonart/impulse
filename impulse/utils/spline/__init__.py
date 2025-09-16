"""
Spline rigging and math utilities.

This package provides a collection of functions for working with NURBS splines,
including core mathematical algorithms.
"""

from .math import (
    generate_knots,
    get_point_on_spline,
    get_tangent_on_spline,
    get_weights_along_spline,
    is_periodic_knot_vector,
    point_on_spline_weights,
    resample,
    tangent_on_spline_weights,
)
from .matrix_spline import (
    MatrixSpline,
    closest_point_on_matrix_spline,
    matrix_spline_from_curve,
    matrix_spline_from_transforms,
    pin_to_matrix_spline,
)
from .maya_query import get_cv_weights, get_cvs, get_knots, maya_to_standard_knots

__all__ = [
    # Math
    "generate_knots",
    "is_periodic_knot_vector",
    "point_on_spline_weights",
    "get_weights_along_spline",
    "tangent_on_spline_weights",
    "get_point_on_spline",
    "get_tangent_on_spline",
    "resample",
    # Matrix Spline
    "MatrixSpline",
    "closest_point_on_matrix_spline",
    "pin_to_matrix_spline",
    "matrix_spline_from_curve",
    "matrix_spline_from_transforms",
    # Maya Query
    "get_knots",
    "maya_to_standard_knots",
    "get_cvs",
    "get_cv_weights",
]
