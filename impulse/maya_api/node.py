from typing import Final
import maya.cmds as cmds

API_VERSION: Final[int] = cmds.about(apiVersion=True)
TARGET_API_VERSION = 20242000

def is_maya2026_or_newer() -> bool:
    return API_VERSION >= 20260000

def is_target_2026_or_newer()-> bool:
    return TARGET_API_VERSION >= 20260000

class Node:
    def __init__(self) -> None:
        pass


class RowFromMatrixNode:
    def __init__(self, name: str = "rowFromMatrix") -> None:
        if is_maya2026_or_newer() and not is_target_2026_or_newer():
            node_name: str = cmds.createNode("rowFromMatrixDL", name=name)
        else:
            node_name: str = cmds.createNode("rowFromMatrix", name=name)
        self.input: str = f"{node_name}.input"
        self.name: str = node_name
        self.matrix: str = f"{node_name}.matrix"
        self.output: str = f"{node_name}.output"

class MultiplyNode(Node):
    def __init__(self, name: str = "multiply") -> None:
        if is_maya2026_or_newer() and not is_target_2026_or_newer():
            node_name: str = cmds.createNode("multiplyDL", name=name)
        else:
            node_name: str = cmds.createNode("multiply", name=name)
        self.name: str = node_name
        self.input: str = f"{self.name}.input"
        self.output: str = f"{self.name}.output"

    def connect_input(self, input_attr: str, input_number: int) -> None:
        """
        Connects an attribute to this node's input, handling the difference
        between Maya <2026 (input1/input2) and Maya 2026+ (multi input array).

        Args:
            input_attr (str): Attribute to connect (e.g. "ctrl.tx").
            input_number (int): Input index (1–2 pre-2026, 1+ in 2026+).
        """
        if input_number < 1:
            raise RuntimeError("Input number starts at 1 not 0")
        cmds.connectAttr(input_attr, f"{self.name}.input[{input_number - 1}]")

    def set_input(self, input_number: int, value: float) -> None:
        if input_number < 1:
            raise RuntimeError("Input number starts at 1 not 0")
        cmds.setAttr(f"{self.name}.input[{input_number - 1}]", value)


class PointMatrixMultiplyNode:
    def __init__(self, name: str = "multiplyPointByMatrix") -> None:
        if is_maya2026_or_newer() and not is_target_2026_or_newer():
            node_name: str = cmds.createNode("multiplyPointByMatrixDL", name=name)
        else:
            node_name: str = cmds.createNode("multiplyPointByMatrix", name=name)
        self.input_point: str = f"{node_name}.input"
        self.name: str = node_name
        self.input_matrix: str = f"{node_name}.matrix"
        self.output: str = f"{node_name}.output"

class SumNode:
    def __init__(self, name: str = "sum") -> None:
        if is_maya2026_or_newer() and not is_target_2026_or_newer():
            node_name: str = cmds.createNode("sumDL", name=name)
        else:
            node_name: str = cmds.createNode("sum", name=name)
        self.name: str = node_name    
        self.input: str = f"{node_name}.input"
        self.output: str = f"{node_name}.output"

class ClampRangeNode:
    def __init__(self, name: str = "clampRange") -> None:
        if is_maya2026_or_newer() and not is_target_2026_or_newer():
            node_name: str = cmds.createNode("clampRangeDL", name=name)
        else:
            node_name: str = cmds.createNode("clampRange", name=name)
        self.name: str = node_name    
        self.input: str = f"{node_name}.input"
        self.maximum: str = f"{node_name}.maximum"
        self.minimum: str = f"{node_name}.minimum"
        self.output: str = f"{node_name}.output"

class DistanceBetweenNode:
    def __init__(self, name: str = "distanceBetween") -> None:
        if is_maya2026_or_newer() and not is_target_2026_or_newer():
            node_name: str = cmds.createNode("distanceBetweenDL", name=name)
        else:
            node_name: str = cmds.createNode("distanceBetween", name=name)
        self.name: str = node_name    
        self.input_point1: str = f"{node_name}.point1"
        self.input_point2: str = f"{node_name}.point2"
        self.input_matrix1: str = f"{node_name}.inMatrix1"
        self.input_matrix2: str = f"{node_name}.inMatrix2"
        self.distance: str = f"{node_name}.distance"

class DivideNode:
    def __init__(self, name: str = "divide") -> None:
        if is_maya2026_or_newer() and not is_target_2026_or_newer():
            node_name: str = cmds.createNode("divideDL", name=name)
        else:
            node_name: str = cmds.createNode("divide", name=name)
        self.name: str = node_name    
        self.input1: str = f"{node_name}.input1"
        self.input2: str = f"{node_name}.input2"
        self.output: str = f"{node_name}.output"

class CrossProductNode:
    def __init__(self, name: str = "crossProduct") -> None:
        if is_maya2026_or_newer() and not is_target_2026_or_newer():
            node_name: str = cmds.createNode("crossProductDL", name=name)
        else:
            node_name: str = cmds.createNode("crossProduct", name=name)
        self.name: str = node_name    
        self.input1: str = f"{node_name}.input1"
        self.input2: str = f"{node_name}.input2"
        self.output: str = f"{node_name}.output"

class LengthNode:
    def __init__(self, name: str = "length") -> None:
        if is_maya2026_or_newer() and not is_target_2026_or_newer():
            node_name: str = cmds.createNode("lengthDL", name=name)
        else:
            node_name: str = cmds.createNode("length", name=name)
        self.name: str = node_name    
        self.input: str = f"{node_name}.input"
        self.output: str = f"{node_name}.output"