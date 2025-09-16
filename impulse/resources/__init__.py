from pathlib import Path


def get_resource_root() -> Path:
    """
    Returns the path of the root resources folder.
    """
    return Path(__file__).parent