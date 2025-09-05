import maya.cmds as cmds


def is_maya2026_or_newer() -> bool:
    return cmds.about(apiVersion=True) >= 20260000


class MultiplyNode:
    def __init__(self, name: str = "multiply") -> None:
        if is_maya2026_or_newer:
            mult_node: str = cmds.createNode("multiply", name=name)
        else:
            mult_node: str = cmds.createNode("multDoubleLinear", name=name)
        self.name: str = mult_node
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
        if not is_maya2026_or_newer and input_number > 2:
            raise RuntimeError("can't connect more than two inputs to a multDoubleLinear node.")
        if is_maya2026_or_newer:
            cmds.connectAttr(input_attr, f"{self.name}.input[{input_number - 1}]")
        else:
            cmds.connectAttr(input_attr, f"{self.name}.input{input_number}")

    def set_input(self, input_number: int, value: float) -> None:
        if input_number < 1:
            raise RuntimeError("Input number starts at 1 not 0")
        if not is_maya2026_or_newer and input_number > 2:
            raise RuntimeError("can't connect more than two inputs to a multDoubleLinear node.")
        if is_maya2026_or_newer:
            cmds.setAttr(f"{self.name}.input[{input_number - 1}]", value)
        else:
            cmds.setAttr(f"{self.name}.input{input_number}", value)

class PointMatrixMultiplyNode:
    def __init__(self, name: str = "multiplyPointByMatrix") -> None:
        if is_maya2026_or_newer:
            mult_node: str = cmds.createNode("multiplyPointByMatrix", name=name)
            self.input_point: str = f"{mult_node}.input"
        else:
            mult_node: str = cmds.createNode("pointMatrixMult", name=name)
            self.input_point: str = f"{mult_node}.inPoint"
        self.name: str = mult_node
        self.input_matrix: str = f"{mult_node}.matrix"
        self.output: str = f"{mult_node}.output"