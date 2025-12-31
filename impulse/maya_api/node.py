from typing import Final, cast

import maya.cmds as cmds
from impulse.maya_api.attribute import (
    Attribute,
    BooleanAttribute,
    EnumAttribute,
    IndexableAttribute,
    IndexableBlendMatrixTargetAttribute,
    IndexableMatrixAttribute,
    IndexableWtMatrixAttribute,
    IntegerAttribute,
    MatrixAttribute,
    ScalarAttribute,
    Vector3Attribute,
    Vector4Attribute, QuatAttribute, AimMatrixAxisAttribute,
)

API_VERSION: Final[int] = cast(int, cmds.about(apiVersion=True))
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
        "sin": {"standard": "sin", "DL": "sinDL"},
        "cos": {"standard": "cos", "DL": "cosDL"},
        "divide": {"standard": "divide", "DL": "divideDL"},
        "clampRange": {"standard": "clampRange", "DL": "clampRangeDL"},
        "distanceBetween": {"standard": "distanceBetween", "DL": "distanceBetweenDL"},
        "crossProduct": {"standard": "crossProduct", "DL": "crossProductDL"},
        "length": {"standard": "length", "DL": "lengthDL"},
        "lerp": {"standard": "lerp", "DL": "lerpDL"},
        "rowFromMatrix": {"standard": "rowFromMatrix", "DL": "rowFromMatrixDL"},
        "multiplyPointByMatrix": {
            "standard": "multiplyPointByMatrix",
            "DL": "multiplyPointByMatrixDL",
        },
        "multiplyVectorByMatrix": {
            "standard": "multiplyVectorByMatrix",
            "DL": "multiplyVectorByMatrixDL",
        },
        "normalize": {"standard": "normalize", "DL": "normalizeDL"},
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


class AimMatrixNode(Node):
    """Maya aimMatrix node with enhanced interface."""

    def __init__(self, name: str = "aimMatrix") -> None:
        super().__init__("aimMatrix", name)

    def _setup_attributes(self) -> None:
        self.input_matrix = MatrixAttribute(f"{self.name}.inputMatrix")
        self.primary = AimMatrixAxisAttribute(f"{self.name}.primary", "primary")
        self.secondary = AimMatrixAxisAttribute(f"{self.name}.secondary", "secondary")
        self.output_matrix = MatrixAttribute(f"{self.name}.outputMatrix")


class AxisFromMatrixNode(Node):
    """Maya axisFromMatrix node with enhanced interface."""

    def __init__(self, name: str = "axisFromMatrix") -> None:
        super().__init__("axisFromMatrix", name)

    def _setup_attributes(self) -> None:
        self.input = MatrixAttribute(f"{self.name}.input")
        self.axis = EnumAttribute(f"{self.name}.axis")
        self.output = Vector3Attribute(f"{self.name}.output")


class BlendMatrixNode(Node):
    """Maya blendMatrix node with enhanced interface."""

    def __init__(self, name: str = "blendMatrix") -> None:
        super().__init__("blendMatrix", name)

    def _setup_attributes(self) -> None:
        self.input_matrix = MatrixAttribute(f"{self.name}.inputMatrix")
        self.pose_space_matrix = MatrixAttribute(f"{self.name}.postSpaceMatrix")
        self.pre_space_matrix = MatrixAttribute(f"{self.name}.preSpaceMatrix")
        self.target = IndexableBlendMatrixTargetAttribute(f"{self.name}.target")
        self.output_matrix = MatrixAttribute(f"{self.name}.outputMatrix")


class ClampRangeNode(Node):
    """Maya clampRange node with enhanced interface."""

    def __init__(self, name: str = "clampRange") -> None:
        super().__init__("clampRange", name)

    def _setup_attributes(self) -> None:
        self.input = ScalarAttribute(f"{self.name}.input")
        self.minimum = ScalarAttribute(f"{self.name}.minimum")
        self.maximum = ScalarAttribute(f"{self.name}.maximum")
        self.output = ScalarAttribute(f"{self.name}.output")


class CosNode(Node):
    """Maya cos node with enhanced interface."""

    def __init__(self, name: str = "cos") -> None:
        super().__init__("cos", name)

    def _setup_attributes(self) -> None:
        self.input: ScalarAttribute = ScalarAttribute(f"{self.name}.input")
        self.output: ScalarAttribute = ScalarAttribute(f"{self.name}.output")


class CrossProductNode(Node):
    """Maya crossProduct node with enhanced interface."""

    def __init__(self, name: str = "crossProduct") -> None:
        super().__init__("crossProduct", name)

    def _setup_attributes(self) -> None:
        self.input1 = Vector3Attribute(f"{self.name}.input1")
        self.input2 = Vector3Attribute(f"{self.name}.input2")
        self.output = Vector3Attribute(f"{self.name}.output")



class DecomposeMatrixNode(Node):
    """Maya decomposeMatrix node with enhanced interface."""

    def __init__(self, name: str = "decomposeMatrix") -> None:
        super().__init__("decomposeMatrix", name)

    def _setup_attributes(self) -> None:
        self.input_matrix = MatrixAttribute(f"{self.name}.inputMatrix")
        self.input_rotate_order = EnumAttribute(f"{self.name}.inputRotateOrder")
        self.output_quat = Vector4Attribute(f"{self.name}.outputQuat")
        self.output_rotate = Vector3Attribute(f"{self.name}.outputRotate")
        self.output_scale = Vector3Attribute(f"{self.name}.outputScale")
        self.output_shear = Vector3Attribute(f"{self.name}.outputShear")
        self.output_translate = Vector3Attribute(f"{self.name}.outputTranslate")


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


class FourByFourMatrixNode(Node):
    """Maya fourByFourMatrix node with enhanced interface."""

    def __init__(self, name: str = "fourByFourMatrix") -> None:
        super().__init__("fourByFourMatrix", name)

    def _setup_attributes(self) -> None:
        self.in_00 = ScalarAttribute(f"{self.name}.in00")
        self.in_01 = ScalarAttribute(f"{self.name}.in01")
        self.in_02 = ScalarAttribute(f"{self.name}.in02")
        self.in_03 = ScalarAttribute(f"{self.name}.in03")
        self.in_10 = ScalarAttribute(f"{self.name}.in10")
        self.in_11 = ScalarAttribute(f"{self.name}.in11")
        self.in_12 = ScalarAttribute(f"{self.name}.in12")
        self.in_13 = ScalarAttribute(f"{self.name}.in13")
        self.in_20 = ScalarAttribute(f"{self.name}.in20")
        self.in_21 = ScalarAttribute(f"{self.name}.in21")
        self.in_22 = ScalarAttribute(f"{self.name}.in22")
        self.in_23 = ScalarAttribute(f"{self.name}.in23")
        self.in_30 = ScalarAttribute(f"{self.name}.in30")
        self.in_31 = ScalarAttribute(f"{self.name}.in31")
        self.in_32 = ScalarAttribute(f"{self.name}.in32")
        self.in_33 = ScalarAttribute(f"{self.name}.in33")
        self.output = MatrixAttribute(f"{self.name}.output")


class LengthNode(Node):
    """Maya length node with enhanced interface."""

    def __init__(self, name: str = "length") -> None:
        super().__init__("length", name)

    def _setup_attributes(self) -> None:
        self.input = Vector3Attribute(f"{self.name}.input")
        self.output = ScalarAttribute(f"{self.name}.output")


class LerpNode(Node):
    """Maya lerp node with enhanced interface."""

    def __init__(self, name: str = "lerp") -> None:
        super().__init__("lerp", name)

    def _setup_attributes(self) -> None:
        self.input1 = ScalarAttribute(f"{self.name}.input1")
        self.input2 = ScalarAttribute(f"{self.name}.input2")
        self.weight = ScalarAttribute(f"{self.name}.weight")
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


class MultiplyVectorByMatrixNode(Node):
    """Maya multiplyVectorByMatrix node with enhanced interface."""

    def __init__(self, name: str = "multiplyVectorByMatrix") -> None:
        super().__init__("multiplyVectorByMatrix", name)

    def _setup_attributes(self) -> None:
        self.input_vector = Vector3Attribute(f"{self.name}.input")
        self.input_matrix = MatrixAttribute(f"{self.name}.matrix")
        self.output = Vector3Attribute(f"{self.name}.output")


class MultMatrixNode(Node):
    """Maya multMatrix node with enhanced interface."""

    def __init__(self, name: str = "multMatrix") -> None:
        super().__init__("multMatrix", name)

    def _setup_attributes(self) -> None:
        self.matrix_in = IndexableMatrixAttribute(f"{self.name}.matrixIn")
        self.matrix_sum = MatrixAttribute(f"{self.name}.matrixSum")


class NormalizeNode(Node):
    """Maya normalize node with enhanced interface."""

    def __init__(self, name: str = "normalize") -> None:
        super().__init__("normalize", name)

    def _setup_attributes(self) -> None:
        self.input = Vector3Attribute(f"{self.name}.input")
        self.output = Vector3Attribute(f"{self.name}.output")


class QuatInvertNode(Node):
    """Maya quatInvert node with enhanced interface."""

    def __init__(self, name: str = "quatInvert") -> None:
        super().__init__("quatInvert", name)

    def _setup_attributes(self) -> None:
        self.input_quat = QuatAttribute(f"{self.name}.inputQuat")
        self.output_quat = QuatAttribute(f"{self.name}.outputQuat")


class QuatProdNode(Node):
    """Maya quatProd node with enhanced interface."""

    def __init__(self, name: str = "quatProd") -> None:
        super().__init__("quatProd", name)

    def _setup_attributes(self) -> None:
        self.input1_quat = QuatAttribute(f"{self.name}.input1Quat")
        self.input2_quat = QuatAttribute(f"{self.name}.input2Quat")
        self.output_quat = QuatAttribute(f"{self.name}.outputQuat")

class QuatToEulerNode(Node):
    """Maya quatToEuler node with enhanced interface."""

    def __init__(self, name: str = "quatToEuler") -> None:
        super().__init__("quatToEuler", name)

    def _setup_attributes(self) -> None:
        self.input_quat = QuatAttribute(f"{self.name}.inputQuat")
        self.input_rotate_order = EnumAttribute(f"{self.name}.inputRotateOrder")
        self.output_rotate = Vector3Attribute(f"{self.name}.outputRotate")

class PickMatrixNode(Node):
    """Maya pickMatrix node with enhanced interface."""

    def __init__(self, name: str = "pickMatrix") -> None:
        super().__init__("pickMatrix", name)

    def _setup_attributes(self) -> None:
        self.input_matrix = MatrixAttribute(f"{self.name}.inputMatrix")
        self.use_translate = BooleanAttribute(f"{self.name}.useTranslate")
        self.use_rotate = BooleanAttribute(f"{self.name}.useRotate")
        self.use_scale = BooleanAttribute(f"{self.name}.useScale")
        self.use_shear = BooleanAttribute(f"{self.name}.useShear")
        self.output_matrix = MatrixAttribute(f"{self.name}.outputMatrix")


class RowFromMatrixNode(Node):
    """Maya rowFromMatrix node with enhanced interface."""

    def __init__(self, name: str = "rowFromMatrix") -> None:
        super().__init__("rowFromMatrix", name)

    def _setup_attributes(self) -> None:
        self.input = IntegerAttribute(f"{self.name}.input")
        self.matrix = MatrixAttribute(f"{self.name}.matrix")
        self.output = Vector4Attribute(f"{self.name}.output")


class SinNode(Node):
    """Maya sin node with enhanced interface."""

    def __init__(self, name: str = "sin") -> None:
        super().__init__("sin", name)

    def _setup_attributes(self) -> None:
        self.input: ScalarAttribute = ScalarAttribute(f"{self.name}.input")
        self.output: ScalarAttribute = ScalarAttribute(f"{self.name}.output")


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


class WtAddMatrixNode(Node):
    """Maya wtAddMatrix node with enhanced interface."""

    def __init__(self, name: str = "wtAddMatrix") -> None:
        super().__init__("wtAddMatrix", name)

    def _setup_attributes(self) -> None:
        self.weight_matrix: IndexableWtMatrixAttribute = IndexableWtMatrixAttribute(
            f"{self.name}.wtMatrix"
        )
        self.matrix_sum: MatrixAttribute = MatrixAttribute(f"{self.name}.matrixSum")
