from __future__ import annotations

from typing import Any

import maya.cmds as cmds


class Attribute:
    """Base class for all Maya attributes."""

    def __init__(self, attr_path: str):
        self.attr_path = attr_path

    def __str__(self) -> str:
        """Return the attribute path when used as a string."""
        return self.attr_path

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.attr_path}')"

    def get(self) -> Any:
        """Get the value of this attribute."""
        return cmds.getAttr(self.attr_path)

    def set(self, value: Any) -> None:
        """Set the value of this attribute."""
        cmds.setAttr(self.attr_path, value)

    @property
    def value(self) -> Any:
        """Get the value of this attribute."""
        return self.get()

    @value.setter
    def value(self, val: Any) -> None:
        """Set the value of this attribute."""
        self.set(val)

    def connect_from(self, source_attr: str | Attribute) -> None:
        """Connect another attribute to this one."""
        source = str(source_attr)  # Works with both strings and Attribute objects
        cmds.connectAttr(source, self.attr_path)

    def connect_to(self, dest_attr: str | Attribute) -> None:
        """Connect this attribute to another one."""
        dest = str(dest_attr)
        cmds.connectAttr(self.attr_path, dest)

    def exists(self) -> bool:
        """Check if this attribute exists."""
        return cmds.objExists(self.attr_path)


class ScalarAttribute(Attribute):
    """A Maya attribute of a scalar type."""

    def __init__(self, attr_path: str):
        super().__init__(attr_path)

    def get(self) -> float:
        """Get the value of this attribute."""
        return cmds.getAttr(self.attr_path)

    def set(self, value: float | int) -> None:
        """Set the value of this attribute."""
        cmds.setAttr(self.attr_path, value)

    @property
    def value(self) -> float:
        """Get the value of this attribute."""
        return self.get()

    @value.setter
    def value(self, val: float | int) -> None:
        """Set the value of this attribute."""
        self.set(val)


class IntegerAttribute(ScalarAttribute):
    """A Maya attribute of an integer type."""

    def __init__(self, attr_path: str):
        super().__init__(attr_path)

    def get(self) -> int:
        """Get the value of this attribute."""
        return int(cmds.getAttr(self.attr_path))

    def set(self, value: int) -> None:
        """Set the value of this attribute."""
        cmds.setAttr(self.attr_path, value)

    @property
    def value(self) -> int:
        """Get the value of this attribute."""
        return self.get()

    @value.setter
    def value(self, val: int) -> None:
        """Set the value of this attribute."""
        self.set(val)


class MatrixAttribute(Attribute):
    """A Maya attribute of the matrix type."""

    def __init__(self, attr_path: str):
        super().__init__(attr_path)


class Vector3Attribute(Attribute):
    """A Maya attribute of the type double3 (XYZ)"""

    def __init__(self, attr_path: str):
        super().__init__(attr_path)

        self.x = ScalarAttribute(f"{attr_path}X")
        self.y = ScalarAttribute(f"{attr_path}Y")
        self.z = ScalarAttribute(f"{attr_path}Z")


class Vector4Attribute(Attribute):
    """A Maya attribute of the type double4 (XYZW)"""

    def __init__(self, attr_path: str):
        super().__init__(attr_path)

        self.x = ScalarAttribute(f"{attr_path}X")
        self.y = ScalarAttribute(f"{attr_path}Y")
        self.z = ScalarAttribute(f"{attr_path}Z")
        self.w = ScalarAttribute(f"{attr_path}W")


class IndexableAttribute(Attribute):
    """A Maya attribute that supports indexing with bracket notation."""

    def __getitem__(self, index: int) -> Attribute:
        """Return the indexed attribute path: attr.input[0], attr.input[1], etc."""
        return Attribute(attr_path=f"{self.attr_path}[{index}]")

    def get_size(self) -> int:
        """Get the number of elements in this array."""
        return cmds.getAttr(self.attr_path, size=True)

    def get_indices(self) -> list[int]:
        """Get all existing indices in this array."""
        return cmds.getAttr(self.attr_path, multiIndices=True) or []
