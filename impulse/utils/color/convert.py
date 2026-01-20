import math

from impulse.utils.color.core import clamp_color


def linear_srgb_to_rec2020(color: tuple[float, float, float]) -> tuple[float, float, float]:
    SRGB_TO_REC2020 = (
        (0.6274, 0.3293, 0.0433),
        (0.0691, 0.9195, 0.0114),
        (0.0164, 0.0880, 0.8956),
    )
    r2020_linear: tuple[float, float, float] = (
        SRGB_TO_REC2020[0][0] * color[0]
        + SRGB_TO_REC2020[0][1] * color[1]
        + SRGB_TO_REC2020[0][2] * color[2],
        SRGB_TO_REC2020[1][0] * color[0]
        + SRGB_TO_REC2020[1][1] * color[1]
        + SRGB_TO_REC2020[1][2] * color[2],
        SRGB_TO_REC2020[2][0] * color[0]
        + SRGB_TO_REC2020[2][1] * color[1]
        + SRGB_TO_REC2020[2][2] * color[2],
    )
    return r2020_linear


def linear_srgb_to_oklab(color: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Converts a linear sRGB color to the Oklab color space.

    Args:
        color (RGB): The input color in linear sRGB space.

    Returns:
        color (OkLab): The corresponding color in Oklab space.
    """
    l: float = 0.4122214708 * color[0] + 0.5363325363 * color[1] + 0.0514459929 * color[2]
    m: float = 0.2119034982 * color[0] + 0.6806995451 * color[1] + 0.1073969566 * color[2]
    s: float = 0.0883024619 * color[0] + 0.2817188376 * color[1] + 0.6299787005 * color[2]

    l_: float = math.copysign(abs(l) ** (1 / 3), l)
    m_: float = math.copysign(abs(m) ** (1 / 3), m)
    s_: float = math.copysign(abs(s) ** (1 / 3), s)

    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def oklab_to_linear_srgb(
    color: tuple[float, float, float], clamp: bool = True
) -> tuple[float, float, float]:
    """
    Converts a Oklab color to the linear sRGB color space.
    Args:
        color (OkLab): The input color in the Oklab space.
        clamp: When True the values of the color will be in a 0-1 range.
    Returns:
        color (OkLab): The corresponding color in linear sRGB space.
    """
    l_: float = color[0] + 0.3963377774 * color[1] + 0.2158037573 * color[2]
    m_: float = color[0] - 0.1055613458 * color[1] - 0.0638541728 * color[2]
    s_: float = color[0] - 0.0894841775 * color[1] - 1.2914855480 * color[2]

    l: float = l_ * l_ * l_
    m: float = m_ * m_ * m_
    s: float = s_ * s_ * s_

    rgb: tuple[float, float, float] = (
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )
    return (
        max(0.0, min(1.0, rgb[0])),
        max(0.0, min(1.0, rgb[1])),
        max(0.0, min(1.0, rgb[2])),
    )


def lab_to_lch(color: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Converts a Lab color to the LCh color space.

    Args:
        color: The input color in Lab space (L, a, b).

    Returns:
        color: The corresponding color in LCh space (L, C, H). Hue is measured in degrees.
    """

    l: float = color[0]
    a: float = color[1]
    b: float = color[2]

    c: float = math.sqrt(a * a + b * b)
    h: float = math.degrees(math.atan2(b, a))
    if h < 0:
        h += 360.0
    return (l, c, h)


def lch_to_lab(color: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Converts an LCh color to the Lab color space.

    Args:
        color (tuple[float, float, float]): The input color in LCh space (L, C, H).
            Hue is measured in degrees.

    Returns:
        tuple[float, float, float]: The corresponding color in Lab space (L, a, b).
    """
    l: float = color[0]
    c: float = color[1]
    h: float = math.radians(color[2])

    a: float = c * math.cos(h)
    b: float = c * math.sin(h)

    return (l, a, b)


def linear_to_srgb_color(linear_color: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Convert a linear MColor to sRGB space.

    Args:
        linear_color: Linear color with RGBA channels in [0,1].

    Returns:
        tuple[float, float, float]: sRGB converted color.
    """

    def convert_channel(c: float) -> float:
        if c <= 0.0031308:
            return 12.92 * c
        else:
            return 1.055 * (pow(base=c, exp=(1.0 / 2.4))) - 0.055

    r = convert_channel(linear_color[0])
    g = convert_channel(linear_color[1])
    b = convert_channel(linear_color[2])

    # Clamp results between 0 and 1 to avoid out of gamut
    return (
        max(0.0, min(1.0, r)),
        max(0.0, min(1.0, g)),
        max(0.0, min(1.0, b)),
    )


def srgb_to_linear_color(srgb_color: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Convert an sRGB MColor to linear color space.

    Args:
        srgb_color: sRGB color with RGBA channels in [0,1].

    Returns:
        tuple[float, float, float]: Linear color.
    """

    def convert_channel(c: float) -> float:
        if c <= 0.0404482362771082:
            return c / 12.92
        else:
            return ((c + 0.055) / 1.055) ** 2.4

    r = convert_channel(srgb_color[0])
    g = convert_channel(srgb_color[1])
    b = convert_channel(srgb_color[2])

    # Clamp between 0 and 1
    return (
        max(0.0, min(1.0, r)),
        max(0.0, min(1.0, g)),
        max(0.0, min(1.0, b)),
    )
