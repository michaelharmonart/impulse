from dataclasses import dataclass

from impulse.utils.color.core import blend_colors_by_weight
from impulse.utils.spline.math import point_on_spline_weights


@dataclass
class GradientStop:
    position: float
    color: tuple[float, float, float]


@dataclass
class Gradient:
    stops: tuple[GradientStop, ...]
    degree: int


OKLCH_HEATMAP_GRADIENT = Gradient(
    stops=(
        GradientStop(0.0, (0, 0.05, -80)),
        GradientStop(0.01, (0.1, 0.15, -60)),
        GradientStop(0.2, (0.7, 0.15, 0)),
        GradientStop(1.0, (1.0, 0.15, 90)),
    ),
    degree=2,
)


def get_gradient_knots(gradient: Gradient):
    degree = gradient.degree
    stop_positions = [stop.position for stop in gradient.stops]
    clamp_start = [stop_positions[0]] * (degree + 1)
    clamp_end = [stop_positions[-1]] * (degree + 1)

    num_internal = len(gradient.stops) - degree - 1
    internal_knots = list(
        (sum(stop_positions[j : j + degree]) / degree) for j in range(1, num_internal + 1)
    )
    clamped_knots = clamp_start + internal_knots + clamp_end
    return [float(knot) for knot in clamped_knots]


def sample_spline_gradient(gradient: Gradient, position: float) -> tuple[float, float, float]:
    gradient_stop_colors = [stop.color for stop in gradient.stops]
    gradient_knots = get_gradient_knots(gradient)
    color_weights = point_on_spline_weights(
        cvs=gradient_stop_colors,
        t=position,
        knots=gradient_knots,
        degree=gradient.degree,
        normalize=False,
    )
    return blend_colors_by_weight(
        colors=(color_weight[0] for color_weight in color_weights),
        weights=(color_weight[1] for color_weight in color_weights),
    )
