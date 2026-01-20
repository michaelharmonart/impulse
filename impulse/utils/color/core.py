from typing import Sequence


def add_colors(
    color1: tuple[float, float, float], color2: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (color1[0] + color2[0], color1[1] + color2[1], color1[2] + color2[2])


def scale_color(color: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    return (color[0] * scalar, color[1] * scalar, color[2] * scalar)


def clamp_value(value: float) -> float:
    max(0.0, min(1.0, value))


def clamp_color(color: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Clamps each component of a color to the range [0.0, 1.0].

    Args:
        color: A tuple of three floats (e.g., RGB or any color space).
    Returns:
        A tuple with each component clamped to the [0.0, 1.0] range.
    """
    return (clamp_value(color[0]), clamp_value(color[1]), clamp_value(color[2]))


def blend_colors_by_weight(colors: Sequence[tuple[float, float, float]], weights: Sequence[float]):
    final_color = (0.0, 0.0, 0.0)
    for color, weight in zip(colors, weights):
        final_color = add_colors(scale_color(color, weight), final_color)
    return final_color
