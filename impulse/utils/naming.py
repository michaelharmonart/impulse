import re

def flip_side(name: str, from_side: str = "L", to_side: str = "R") -> str:
    """
    Replaces side token in the name from 'L' to 'R' or vice versa,
    only when it's a distinct token (e.g., 'Front_Leg_L' becomes 'Front_Leg_R').

    Args:
        name: The original name.
        from_side: The side token to replace.
        to_side: The replacement side token.

    Returns:
        The renamed string.
    """
    # Regex: matches '_L' or '_R' at end or before non-word boundary
    # (?<=_) — positive lookbehind, ensures that the side marker (e.g. L) is preceded by an underscore.
    # ({from_side}) — captures the actual side marker (L, R, etc.)
    # (?=_|$) — positive lookahead, ensures that the side marker is followed by either another underscore or the end of the string.

    pattern = rf'(?<=_)({from_side})(?=_|$)'
    return re.sub(pattern, f'{to_side}', name)

def get_side(name: str, sides: list[str] = ["L", "R", "M"]) -> str | None:
    """
    Extracts the side token from a control name. eg. Front_Leg_L_CTL will return "L"

    Args:
        name: The control name.
        sides: A list of valid side tokens to check for (default: ["L", "R", "M"]).

    Returns:
        The side token found in the name, or None if not found.
    """
    # Create a pattern that matches any of the sides preceded by "_" and followed by "_" or end of string
    side_pattern = rf'(?<=_)({"|".join(sides)})(?=_|$)'
    match = re.search(side_pattern, name)
    return match.group(0) if match else None