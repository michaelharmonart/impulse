from typing import Final

import maya.cmds as cmds

from impulse.maya_api.attribute import (
    Attribute,
    IndexableAttribute,
    IntegerAttribute,
    MatrixAttribute,
    ScalarAttribute,
    Vector3Attribute,
    Vector4Attribute,
)

API_VERSION: Final[int] = cmds.about(apiVersion=True)
TARGET_API_VERSION = 20242000


def is_maya2026_or_newer() -> bool:
    return API_VERSION >= 20260000


def is_target_2026_or_newer() -> bool:
    return TARGET_API_VERSION >= 20260000


class Node:
    """Base class for all Maya nodes."""

    NODE_TYPES: dict[str, dict[str, str]] = {
        "multiply": {"standard": "multiply", "DL": "multiplyDL"},
        "subtract": {"standard": "subtract", "DL": "subtractDL"},
        "sum": {"standard": "sum", "DL": "sumDL"},
        "divide": {"standard": "divide", "DL": "divideDL"},
        "clampRange": {"standard": "clampRange", "DL": "clampRangeDL"},
        "distanceBetween": {"standard": "distanceBetween", "DL": "distanceBetweenDL"},
        "crossProduct": {"standard": "crossProduct", "DL": "crossProductDL"},
        "length": {"standard": "length", "DL": "lengthDL"},
        "rowFromMatrix": {"standard": "rowFromMatrix", "DL": "rowFromMatrixDL"},
        "multiplyPointByMatrix": {
            "standard": "multiplyPointByMatrix",
            "DL": "multiplyPointByMatrixDL",
        },
    }

    def __init__(self, node_type: str, name: str | None = None) -> None:
        """
        Initialize a Maya node with version compatibility.

        Args:
            node_type: The base Maya node type (e.g., "multiply", "sum")
            name: Optional custom name for the node
        """
        self.node_type: str = node_type
        self.name: str = self._create_node(node_type, name=name or node_type)
        self._setup_attributes()

    def _create_node(self, node_type: str, name: str) -> str:
        """Create the Maya node with appropriate version handling."""
        if node_type in self.NODE_TYPES:
            types = self.NODE_TYPES[node_type]
            if is_maya2026_or_newer() and not is_target_2026_or_newer():
                maya_node_type = types["DL"]
            else:
                maya_node_type = types["standard"]
        else:
            maya_node_type = node_type

        return cmds.createNode(maya_node_type, name=name)

    def _setup_attributes(self) -> None:
        """Override in subclasses to define node-specific attributes."""
        pass

    def delete(self) -> None:
        """Delete this node."""
        if cmds.objExists(self.name):
            cmds.delete(self.name)

    def exists(self) -> bool:
        """Check if this node exists in Maya."""
        return cmds.objExists(self.name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class ClampRangeNode(Node):
    """Maya clampRange node with enhanced interface."""

    def __init__(self, name: str = "clampRange") -> None:
        super().__init__("clampRange", name)

    def _setup_attributes(self) -> None:
        self.input = ScalarAttribute(f"{self.name}.input")
        self.minimum = ScalarAttribute(f"{self.name}.minimum")
        self.maximum = ScalarAttribute(f"{self.name}.maximum")
        self.output = ScalarAttribute(f"{self.name}.output")


class CrossProductNode(Node):
    """Maya crossProduct node with enhanced interface."""

    def __init__(self, name: str = "crossProduct") -> None:
        super().__init__("crossProduct", name)

    def _setup_attributes(self) -> None:
        self.input1 = Vector3Attribute(f"{self.name}.input1")
        self.input2 = Vector3Attribute(f"{self.name}.input2")
        self.output = Vector3Attribute(f"{self.name}.output")


class DistanceBetweenNode(Node):
    """Maya distanceBetween node with enhanced interface."""

    def __init__(self, name: str = "distanceBetween") -> None:
        super().__init__("distanceBetween", name)

    def _setup_attributes(self) -> None:
        self.point1 = Vector3Attribute(f"{self.name}.point1")
        self.point2 = Vector3Attribute(f"{self.name}.point2")
        self.input_matrix1 = MatrixAttribute(f"{self.name}.inMatrix1")
        self.input_matrix2 = MatrixAttribute(f"{self.name}.inMatrix2")
        self.distance = ScalarAttribute(f"{self.name}.distance")


class DivideNode(Node):
    """Maya divide node with enhanced interface."""

    def __init__(self, name: str = "divide") -> None:
        super().__init__("divide", name)

    def _setup_attributes(self) -> None:
        self.input1 = ScalarAttribute(f"{self.name}.input1")
        self.input2 = ScalarAttribute(f"{self.name}.input2")
        self.output = ScalarAttribute(f"{self.name}.output")


class LengthNode(Node):
    """Maya length node with enhanced interface."""

    def __init__(self, name: str = "length") -> None:
        super().__init__("length", name)

    def _setup_attributes(self) -> None:
        self.input = Vector3Attribute(f"{self.name}.input")
        self.output = ScalarAttribute(f"{self.name}.output")


class MultiplyNode(Node):
    """Maya multiply node with enhanced interface."""

    def __init__(self, name: str = "multiply") -> None:
        super().__init__("multiply", name)

    def _setup_attributes(self) -> None:
        self.input: IndexableAttribute = IndexableAttribute(f"{self.name}.input")
        self.output: ScalarAttribute = ScalarAttribute(f"{self.name}.output")


class MultiplyPointByMatrixNode(Node):
    """Maya multiplyPointByMatrix node with enhanced interface."""

    def __init__(self, name: str = "multiplyPointByMatrix") -> None:
        super().__init__("multiplyPointByMatrix", name)

    def _setup_attributes(self) -> None:
        self.input_point = Vector3Attribute(f"{self.name}.input")
        self.input_matrix = MatrixAttribute(f"{self.name}.matrix")
        self.output = Vector3Attribute(f"{self.name}.output")


class RowFromMatrixNode(Node):
    """Maya rowFromMatrix node with enhanced interface."""

    def __init__(self, name: str = "rowFromMatrix") -> None:
        super().__init__("rowFromMatrix", name)

    def _setup_attributes(self) -> None:
        self.input = IntegerAttribute(f"{self.name}.input")
        self.matrix = MatrixAttribute(f"{self.name}.matrix")
        self.output = Vector4Attribute(f"{self.name}.output")


class SubtractNode(Node):
    """Maya subtract node with enhanced interface."""

    def __init__(self, name: str = "subtract") -> None:
        super().__init__("subtract", name)

    def _setup_attributes(self) -> None:
        self.input1: ScalarAttribute = ScalarAttribute(f"{self.name}.input1")
        self.input2: ScalarAttribute = ScalarAttribute(f"{self.name}.input2")
        self.output: ScalarAttribute = ScalarAttribute(f"{self.name}.output")


class SumNode(Node):
    """Maya sum node with enhanced interface."""

    def __init__(self, name: str = "sum") -> None:
        super().__init__("sum", name)

    def _setup_attributes(self) -> None:
        self.input: IndexableAttribute = IndexableAttribute(f"{self.name}.input")
        self.output: ScalarAttribute = ScalarAttribute(f"{self.name}.output")

