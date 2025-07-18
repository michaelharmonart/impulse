import json
from typing import Literal
import maya.cmds as cmds

from impulse.utils.transform import RotationOrder, match_transform, matrix_constraint
from impulse.utils import pin as pin
from impulse.utils import math as math
from enum import Enum
import os

from impulse.utils.naming import flip_side, get_side

CONTROL_DIR: str = os.path.dirname(os.path.realpath(__file__)) + "/control_shapes"


class ControlShape(Enum):
    """Enum for available control shapes with file names."""

    CIRCLE = "circle"
    SQUARE = "square"
    CUBE = "cube"
    SPHERE = "sphere"
    LOCATOR = "locator"
    DIAMOND = "diamond"
    TRIANGLE = "triangle"

    @property
    def filename(self) -> str:
        """returns the filename of the json file representing the control shape."""
        return self.value

class Direction(Enum):
    """Enum for available control directions"""

    X = 0
    Y = 1
    Z = 2


def get_shapes(transform: str) -> list[str]:
    # list the shapes of node
    shape_list: list[str] = cmds.listRelatives(
        transform, shapes=True, noIntermediate=True, children=True
    )

    if shape_list:
        return shape_list
    else:
        raise RuntimeError(f"{transform} has no child shape nodes")


def get_cv_positions(curve_shape: str) -> list[tuple[float, float, float]]:
    """
    Gets the positions of all CVs for a given curve shape.
    Args:
        curve_shape(str): Name of curve shape node.
    Returns:
        list: A list of CV positions as tuples
    """
    curve_info = cmds.createNode("curveInfo", name="temp_curveInfo")
    cmds.connectAttr(f"{curve_shape}.local", f"{curve_info}.inputCurve")
    cv_list: list[tuple[float, float, float]] = cmds.getAttr(f"{curve_info}.controlPoints[*]")
    cmds.delete(curve_info)
    position_list = [(position[0], position[1], position[2]) for position in cv_list]
    return position_list


def get_cv_weights(curve_shape: str) -> list[float]:
    """
    Gets the weights of all CVs for a given curve shape.
    Args:
        curve_shape(str): Name of curve shape node.
    Returns:
        list: A list of CV weight values.
    """
    curve_info = cmds.createNode("curveInfo", name="temp_curveInfo")
    cmds.connectAttr(f"{curve_shape}.local", f"{curve_info}.inputCurve")
    weights: list[float] = cmds.getAttr(f"{curve_info}.weights[*]")
    cmds.delete(curve_info)
    return weights


def get_knots(curve_shape: str) -> list[float]:
    """
    Gets the knot vector for a given curve shape.
    Args:
        curve_shape(str): Name of curve shape node.
    Returns:
        list: A list of knot values. (aka knot vector)
    """
    curve_info = cmds.createNode("curveInfo", name="temp_curveInfo")
    cmds.connectAttr(f"{curve_shape}.local", f"{curve_info}.inputCurve")
    knots: list[float] = cmds.getAttr(f"{curve_info}.knots[*]")
    cmds.delete(curve_info)
    return knots


def get_curve_info(curve: str):
    curve_dict = {}
    for curve in get_shapes(transform=curve):
        degree = cmds.getAttr(curve + ".degree")
        form = cmds.getAttr(curve + ".form")
        cv_positions: list[tuple[float, float, float]] = get_cv_positions(curve_shape=curve)
        cv_weights: list[float] = get_cv_weights(curve_shape=curve)
        knots: list[float] = get_knots(curve_shape=curve)
        draw_on_top: bool = cmds.getAttr(f"{curve}.alwaysDrawOnTop")
        curve_info = {
            "degree": degree,
            "form": form,
            "cv_positions": cv_positions,
            "cv_weights": cv_weights,
            "knots": knots,
            "draw_on_top": draw_on_top,
        }
        curve_dict[curve] = curve_info
    return curve_dict


def write_curve(control: str | None = None, name: str | None = None, force: bool = False):
    """
    Saves selected or defined curve to shape library.

    Args:
        control(str): Name of control to save. If None, uses the current selection.
    Returns:
        list: A list of CV weight values.
    """
    # make sure we either define a curve or have one selected
    # also make sure we're using the transform node
    if not control:
        selection: list[str] = cmds.ls(selection=True)
        if len(selection) == 0:
            raise RuntimeError(
                "Unable to write control shape to file, no control transform was defined, and no control is selected."
            )
        control: str = selection[0]

    # if a name is not defined, use the curves name instead
    if not name:
        name: str = control

    json_path = f"{CONTROL_DIR}/{name}.json"

    # If the file exists, only write it if force = True, or after asking for confirmation.
    if os.path.isfile(path=json_path):
        if force:
            pass
        else:
            confirm: str = cmds.confirmDialog(
                title="File Overwrite",
                message=f"{json_path} already exists and will be overwritten, are you sure you want to write the file?",
                button=["Yes", "No"],
                defaultButton="Yes",
                cancelButton="No",
                dismissString="No",
            )
            if confirm == "Yes":
                pass
            else:
                return

    # get curve data
    curve_data = get_curve_info(curve=control)
    json_dump = json.dumps(obj=curve_data, indent=4)

    # write shape
    with open(file=json_path, mode="w") as json_file:
        json_file.write(json_dump)
        json_file.close()

    return


_loaded_control_shapes = {}


def get_curve_data(curve_shape: ControlShape | str) -> dict:
    """
    Args:
        curve_shape(ControlShape): Name of the control shape to retrieve.
    Returns:
        dict: Curve data.
    """
    if isinstance(curve_shape, str):
        curve_shape: ControlShape = ControlShape[curve_shape.strip().upper()]
    if curve_shape not in _loaded_control_shapes:
        # check if curve dict is a file and convert it to dictionary if it is
        file_path = f"{CONTROL_DIR}/{curve_shape.filename}.json"
        if not os.path.isfile(file_path):
            cmds.error(
                f"Shape does not exist in library. You must write out the file {file_path} before reading."
            )

        with open(file_path, "r") as json_file:
            json_data = json_file.read()
            _loaded_control_shapes[curve_shape] = json.loads(json_data)
    return _loaded_control_shapes[curve_shape]


def create_curve(curve_shape: ControlShape | str = ControlShape.CIRCLE) -> str:
    """
    Creates a curve from the specified item in the shape library.

    Args:
        curve_shape(ControlShape): Name of the control shape to generate.
    Returns:
        str: Name of the generated curve transform.
    """
    if isinstance(curve_shape, str):
        curve_shape: ControlShape = ControlShape[curve_shape.strip().upper()]
    curve_data = get_curve_data(curve_shape=curve_shape)
    curve_transform: str
    for index, shape in enumerate(curve_data):
        info = curve_data[shape]
        positions: list[tuple[float, float, float]] = info["cv_positions"]
        degree: int = info["degree"]
        periodic: bool = True if info["form"] == 2 else False
        knots: list[float] = info["knots"]
        weights: list[float] = info["cv_weights"]
        position_weights: list[tuple[float, float, float, float]] = [
            (position[0], position[1], position[2], weights[index])
            for index, position in enumerate(positions)
        ]
        if index == 0:
            curve_transform: str = cmds.curve(
                name=curve_shape.value,
                pointWeight=position_weights,
                knot=knots,
                periodic=periodic,
                degree=degree,
            )
            curve_shape_node: str = get_shapes(curve_transform)[0]
            curve_shape_node = cmds.rename(curve_shape_node, f"{curve_shape.value}Shape")
        else:
            child_curve_transform: str = cmds.curve(
                pointWeight=position_weights, knot=knots, periodic=periodic, degree=degree
            )
            curve_shape_node: str = get_shapes(child_curve_transform)[0]
            curve_shape_node = cmds.rename(curve_shape_node, f"{curve_shape.value}Shape{index}")
            cmds.parent(curve_shape_node, curve_transform, shape=True, relative=True)
            cmds.delete(child_curve_transform)
    cmds.select(curve_transform)
    return curve_transform


def combine_curves(main_curve: str | None = None, other_curves: list[str] | None = None):
    """
    This is a utility that can be used to combine curve shapes under one transform

    Args:
        curve(str): main transform that shapes will be parented under
        shapes(list): a list of other shapes to combine under main transform
    Returns:
        str: Name of the merged curve transform.
    """
    selection: list[str] = cmds.ls(selection=True)
    if not main_curve:
        main_curve: str = selection[0]
    cmds.makeIdentity(main_curve, apply=True)

    if not other_curves:
        if len(selection) > 1:
            other_curves = cmds.ls(selection=True)
        else:
            other_curves = cmds.listRelatives(main_curve, children=True)

    all_shapes = []
    for curve in other_curves:
        shape_list = get_shapes(transform=curve)
        if shape_list:
            all_shapes += shape_list

    for shape in all_shapes:
        transform = cmds.listRelatives(shape, parent=True)
        cmds.makeIdentity(transform, apply=True)
        if cmds.listRelatives(shape, parent=True)[0] == main_curve:
            continue
        cmds.parent(shape, main_curve, shape=True, relative=True)
        if not cmds.listRelatives(transform, allDescendents=True):
            cmds.delete(transform)


def get_tagged_controls(side: str | None = None) -> list[str]:
    """
    Returns all transform nodes tagged as controllers via a connected controller node.

    Returns:
        list: A list of transform node names that are tagged as controllers.
    """
    controller_nodes: list[str] = cmds.ls(type="controller")
    tagged_controls: list[str] = []
    for control_node in controller_nodes:
        connected: list[str] = cmds.listConnections(
            f"{control_node}.controllerObject", source=True, destination=False
        )
        if side:
            if get_side(control_node) == side:
                tagged_controls.append(connected[0])
        elif connected:
            tagged_controls.append(connected[0])

    return tagged_controls


def write_control_shapes(filepath: str, force: bool = False) -> None:
    """
    Writes a JSON file representing the control curves of all controls in the scene at the filepath specified.

    Args:
        filepath: The path and filename and extension to save under.
        force: If True, will automatically overwrite any existing file at the filepath specified.

    """

    # If the file exists, only write it if force = True, or after asking for confirmation.
    if os.path.isfile(path=filepath):
        if force:
            pass
        else:
            confirm: str = cmds.confirmDialog(
                title="File Overwrite",
                message=f"{filepath} already exists and will be overwritten, are you sure you want to write the file?",
                button=["Yes", "No"],
                defaultButton="Yes",
                cancelButton="No",
                dismissString="No",
            )
            if confirm == "Yes":
                pass
            else:
                return

    control_dict = {}

    controls: list[str] = get_tagged_controls()
    for control in controls:
        curve_info = get_curve_info(curve=control)
        control_dict[control] = curve_info

    json_path: str = filepath
    json_dump: str = json.dumps(obj=control_dict, indent=4)

    with open(file=json_path, mode="w") as json_file:
        json_file.write(json_dump)
        json_file.close()
    return


def apply_control_file(filepath: str) -> None:
    if not os.path.isfile(path=filepath):
        raise RuntimeError(f"{filepath} is not a valid file. Unable to load controls")
    current_control_dict = {}
    controls = get_tagged_controls()
    for control in controls:
        current_control_dict[control] = True

    control_dict = {}
    with open(filepath, "r") as json_file:
        json_data = json_file.read()
        control_dict = json.loads(json_data)
    for control in control_dict:
        if control in current_control_dict:
            shapes: list[str] = get_shapes(transform=control)
            for shape in shapes:
                cmds.delete(shape)
            curve_data = control_dict[control]
            for index, shape in enumerate(curve_data):
                info = curve_data[shape]
                positions: list[tuple[float, float, float]] = info["cv_positions"]
                degree: int = info["degree"]
                periodic: bool = True if info["form"] == 2 else False
                knots: list[float] = info["knots"]
                weights: list[float] = info["cv_weights"]
                draw_on_top: bool = info["draw_on_top"]
                position_weights: list[tuple[float, float, float, float]] = [
                    (position[0], position[1], position[2], weights[index])
                    for index, position in enumerate(positions)
                ]

                child_curve_transform: str = cmds.curve(
                    pointWeight=position_weights, knot=knots, periodic=periodic, degree=degree
                )
                curve_shape_node: str = get_shapes(child_curve_transform)[0]
                curve_shape_node = cmds.rename(curve_shape_node, shape)
                cmds.setAttr(f"{curve_shape_node}.alwaysDrawOnTop", 1 if draw_on_top else 0)
                cmds.parent(curve_shape_node, control, shape=True, relative=True)
                cmds.delete(child_curve_transform)


def mirror_control_shapes() -> None:
    controls: list[str] = get_tagged_controls(side="L")
    for control in controls:
        shapes: list[str] = get_shapes(transform=control)
        duplicated: str = cmds.duplicate(control)[0]
        duplicated_shapes: list[str] = get_shapes(transform=duplicated)
        flipped_control: str = flip_side(control)
        for shape, duplicated_shape in zip(shapes, duplicated_shapes):
            flipped_shape: str = flip_side(name=shape)
            cmds.delete(flipped_shape)
            duplicated_shape: str = cmds.rename(duplicated_shape, flipped_shape)
            cmds.parent(duplicated_shape, flipped_control, shape=True, relative=True)
        cmds.delete(duplicated)


class Control:
    def __init__(self, control_transform: str, offset_transform: str, name: str | None):
        self.control_transform = control_transform
        self.offset_transform = offset_transform
        if name:
            self.name = name
        else:
            self.name = control_transform


def draw_on_top(control: Control, enable: bool = True) -> None:
    shapes: list[str] = get_shapes(control.control_transform)
    value: Literal[1, 0] = 1 if enable else 0
    for shape in shapes:
        cmds.setAttr(f"{shape}.alwaysDrawOnTop", value)


def get_controller_tag(control: Control) -> str | None:
    tags = cmds.listConnections(f"{control.control_transform}.message", type="controller")
    return tags[0] if tags else None


def tag_as_controller(control: Control, parent: str | None = None):
    """
    Tag a Control as a controller (both for Maya's Evaluation Graph and for Impulse).
    Optionally connect the control tag to a parent controller.
    """
    # Check if already tagged
    tag: str | None = get_controller_tag(control)
    if tag:
        return tag

    tag: str = cmds.createNode("controller", name=f"{control.name}_CTL_TAG")
    cmds.connectAttr(f"{control.control_transform}.message", f"{tag}.controllerObject")

    if parent:
        parent_tag: str | None = get_controller_tag(parent)
        if parent_tag:
            cmds.connectAttr(f"{parent_tag}.message", f"{tag}.parent")
    return tag


def lock_pivots(transform: str) -> None:
    """
    Lock the rotate and scale pivot attributes of the given transform.
    """
    cmds.setAttr(f"{transform}.rotateAxis", lock=True)
    cmds.setAttr(f"{transform}.rotatePivot", lock=True)
    cmds.setAttr(f"{transform}.rotatePivotTranslate", lock=True)

    cmds.setAttr(f"{transform}.scalePivot", lock=True)
    cmds.setAttr(f"{transform}.scalePivotTranslate", lock=True)


def make_control(
    name: str,
    parent: str = None,
    position: tuple[float, float, float] | None = None,
    target_transform: str | None = None,
    direction: Direction = Direction.Y,
    opposite_direction: bool = False,
    size: float = 1,
    dimensions: tuple[float, float, float] = (1, 1, 1),
    control_shape: ControlShape | str = ControlShape.CIRCLE,
    offset: float = 0,
    rotation_order: RotationOrder | None = None,
) -> Control:
    """
    Create a control curve in Maya at a given position, scale it, offset it,
    and parent it under the specified transform.

    Args:
        position: (x, y, z) coordinates for the control's final position.
        target_transform: when set, the control will be generated to match the world space transform of the given transform node.
        parent: Name of the parent transform.
        direction: Direction control shape will face.
        size: Scaling factor for the control curve.
        dimensions: Dimensions of the resulting control curve.
        control_shape: The type of control shape to create.
        offset: Vertical offset applied immediately after creation.

    Returns:
        The name of the created control transform.
    """
    if not rotation_order:
        if target_transform:
            target_rotation_order = cmds.getAttr(f"{target_transform}.rotateOrder")
            rotation_order: RotationOrder = RotationOrder(
                value=target_rotation_order
                if target_rotation_order is not None
                else RotationOrder.YXZ
            )
        else:
            rotation_order = RotationOrder.YXZ
    if isinstance(control_shape, str):
        control_shape: ControlShape = ControlShape[control_shape.strip().upper()]
    # Generate a curve and apply rotation order.
    control_transform: str = create_curve(curve_shape=control_shape)
    cmds.setAttr(f"{control_transform}.rotateOrder", rotation_order.value)

    # Adjust the control's scale, apply an offset, reset transforms, reposition.
    control_transform: str = cmds.rename(control_transform, f"{name}_CTL")
    scaled_dimensions = [size * dimension for dimension in dimensions]
    cmds.scale(*scaled_dimensions, control_transform, relative=False)
    cmds.move(0, offset, 0, control_transform, relative=True)
    cmds.makeIdentity(control_transform, apply=True)
    cmds.xform(control_transform, pivots=(0, 0, 0))

    # Comfort feature: make it so it's not possible to have negative scale
    min_scale: float = 0.01
    cmds.transformLimits(
        control_transform,
        enableScaleX=(True, False),
        scaleX=(min_scale, 1),
        enableScaleY=(True, False),
        scaleY=(min_scale, 1),
        enableScaleZ=(True, False),
        scaleZ=(min_scale, 1),
    )

    if direction == Direction.X:
        if opposite_direction:
            cmds.rotate(0, 0, 90)
        else:
            cmds.rotate(0, 0, -90)
    elif direction == Direction.Z:
        if opposite_direction:
            cmds.rotate(-90, 0, 0)
        else:
            cmds.rotate(90, 0, 0)
    cmds.makeIdentity(control_transform, apply=True)
    cmds.xform(control_transform, pivots=(0, 0, 0))
    lock_pivots(transform=control_transform)

    offset_transform: str = cmds.group(control_transform, name=f"{name}_OFFSET")
    cmds.xform(offset_transform, pivots=(0, 0, 0))
    cmds.setAttr(f"{offset_transform}.rotateOrder", rotation_order.value)

    if parent:
        cmds.parent(offset_transform, parent, relative=True)

    if target_transform:
        match_transform(transform=offset_transform, target_transform=target_transform)
    elif position:
        cmds.move(position[0], position[1], position[2], relative=True, worldSpace=True)
    control = Control(
        control_transform=control_transform, offset_transform=offset_transform, name=name
    )
    tag_as_controller(control)
    return control


def make_surface_control(
    name: str,
    surface: str,
    parent: str = None,
    position: tuple[float, float, float] = (0, 0, 0),
    target_transform: str = None,
    uv_position: tuple[float, float] = None,
    u_attribute: str = None,
    v_attribute: str = None,
    control_sensitivity: tuple[float, float] = (1, 1),
    direction: Direction = Direction.Y,
    opposite_direction: bool = False,
    size: float = 1,
    control_shape: ControlShape = ControlShape.CIRCLE,
    offset: float = 0.1,
) -> Control:
    """
    Create a control that moves only along a given surface (plus an offset).

    Args:
        surface: The surface (mesh or NURBS) that the control will move along.
        parent: Name of the transform to parent control to.
        position: World space position that will be projected onto the surface to set the default control position in UV space.
        target_transform: when set, the control will be generated to match the position and angle of the given transform node.
        u_attribute: Attribute to use as U offset instead of the transform or position.
        v_attribute: Attribute to use as V offset instead of the transform or position.
        control_sensitivity: multiplier for UV space movement from the control transform (needed since the surface UV space will be 0-1 matter how large it is)
        direction: Direction control shape will face.
        size: Scaling factor for the control curve.
        control_shape: The type of control shape to create.
        offset: Vertical offset applied immediately after creation.
        use_opm: Use offset parent matrix instead of offset transform group (cleaner hierarchy)

    Returns:
        The name of the created control transform.
    """

    # Generate a curve
    control_transform = create_curve(curve_shape=control_shape)

    # Adjust the control's scale, apply an offset, reset transforms, and reposition.
    cmds.select(control_transform)
    control_transform = cmds.rename(control_transform, f"{name}_CTL")
    cmds.scale(size, size, size, relative=True)
    cmds.move(0, offset, 0, relative=True)
    cmds.makeIdentity(apply=True)
    cmds.xform(control_transform, pivots=(0, 0, 0))

    if direction == Direction.X:
        if opposite_direction:
            cmds.rotate(0, 0, 90)
        else:
            cmds.rotate(0, 0, -90)
    elif direction == Direction.Z:
        if opposite_direction:
            cmds.rotate(-90, 0, 0)
        else:
            cmds.rotate(90, 0, 0)
    cmds.makeIdentity(apply=True)
    cmds.xform(control_transform, pivots=(0, 0, 0))
    lock_pivots(transform=control_transform)

    control_static_group = cmds.group(control_transform, name=f"{name}_STATIC")
    offset_matrix_compose = cmds.createNode(
        "composeMatrix", name=f"{control_transform}_offsetMatrixCompose"
    )
    cmds.connectAttr(
        f"{control_transform}.translate.translateX",
        f"{offset_matrix_compose}.inputTranslate.inputTranslateX",
    )
    cmds.connectAttr(
        f"{control_transform}.translate.translateZ",
        f"{offset_matrix_compose}.inputTranslate.inputTranslateZ",
    )
    matrix_inverse = cmds.createNode("inverseMatrix", name=f"{control_transform}_inverseMatrix")
    cmds.connectAttr(f"{offset_matrix_compose}.outputMatrix", f"{matrix_inverse}.inputMatrix")
    cmds.connectAttr(f"{matrix_inverse}.outputMatrix", f"{control_static_group}.offsetParentMatrix")

    offset_transform = cmds.group(control_static_group, name=f"{name}_OFFSET")
    shapes = cmds.listRelatives(surface, shapes=True) or []
    if not shapes:
        cmds.error(f"No shape node found on object: {surface}")
    shape = shapes[0]

    surface_type = cmds.objectType(shape)
    if surface_type == "mesh":
        cp_node_type = "closestPointOnMesh"
        attr_world = ".worldMesh[0]"
        cp_input = ".inMesh"
    elif surface_type == "nurbsSurface":
        cp_node_type = "closestPointOnSurface"
        attr_world = ".worldSpace[0]"
        cp_input = ".inputSurface"
    else:
        raise RuntimeError(f"{surface} is of type {surface_type}, but should be NURBS or a mesh.")

    # Create temp node to get the UV of the closest point on the surface to the default location of this control.
    rotation_y = 0
    cp_node = cmds.createNode(cp_node_type, name=f"{name}_closestPoint_TEMP")
    cmds.connectAttr(f"{shape}{attr_world}", f"{cp_node}{cp_input}")
    if surface_type == "mesh":
        cmds.connectAttr(f"{shape}.worldMatrix[0]", f"{cp_node}.inputMatrix")

    x_attribute = f"{control_transform}.translate.translateX"
    z_attribute = f"{control_transform}.translate.translateZ"
    if target_transform:
        position = cmds.xform(target_transform, worldSpace=True, query=True, translation=True)
        rotate_matrix = cmds.createNode("composeMatrix", name=f"{name}_rotationMatrixCompose")
        cmds.connectAttr(
            f"{offset_transform}.rotate.rotateY", f"{rotate_matrix}.inputRotate.inputRotateY"
        )
        translate_matrix = cmds.createNode("composeMatrix", name=f"{name}_translateMatrixCompose")
        cmds.connectAttr(
            f"{control_transform}.translate.translateX",
            f"{translate_matrix}.inputTranslate.inputTranslateX",
        )
        cmds.connectAttr(
            f"{control_transform}.translate.translateZ",
            f"{translate_matrix}.inputTranslate.inputTranslateZ",
        )
        multiplied_matrix = cmds.createNode("multMatrix", name=f"{name}_multMatrix")
        cmds.connectAttr(f"{translate_matrix}.outputMatrix", f"{multiplied_matrix}.matrixIn[0]")
        cmds.connectAttr(f"{rotate_matrix}.outputMatrix", f"{multiplied_matrix}.matrixIn[1]")
        decompose_matrix = cmds.createNode("decomposeMatrix", name=f"{name}_decomposeMatrix")
        cmds.connectAttr(f"{multiplied_matrix}.matrixSum", f"{decompose_matrix}.inputMatrix")
        x_attribute = f"{decompose_matrix}.outputTranslate.outputTranslateX"
        z_attribute = f"{decompose_matrix}.outputTranslate.outputTranslateZ"

    cmds.setAttr(f"{cp_node}.inPosition", position[0], position[1], position[2])
    if uv_position:
        default_u: float = uv_position[0]
        default_v: float = uv_position[1]
    else:
        default_u: float = cmds.getAttr(f"{cp_node}.result.parameterU")
        default_v: float = cmds.getAttr(f"{cp_node}.result.parameterV")
    min_max_u: tuple[float, float] = (0, 1)
    min_max_v: tuple[float, float] = (0, 1)
    if surface_type == "nurbsSurface":
        min_max_u = cmds.getAttr(f"{shape}.minMaxRangeU")[0]
        min_max_v = cmds.getAttr(f"{shape}.minMaxRangeV")[0]
    cmds.delete(cp_node)
    u_range: float = min_max_u[1] - min_max_u[0]
    v_range: float = min_max_v[1] - min_max_v[0]
    uv_ratio: float = u_range / v_range

    uv_pin_node = pin.make_uv_pin(
        object_to_pin=offset_transform, surface=surface, u=default_u, v=default_v, normalize=False
    )
    if target_transform:
        temp_locator = cmds.group(empty=True, parent=offset_transform)
        cmds.parentConstraint(target_transform, temp_locator, maintainOffset=False)
        rotation_y = cmds.getAttr(f"{temp_locator}.rotate.rotateY")
        cmds.delete(temp_locator)
        cmds.xform(offset_transform, rotation=(0, rotation_y, 0), worldSpace=False)

    multiplier = cmds.createNode("multiplyDivide", name=f"{name}_sensitivityMultiply")
    cmds.connectAttr(x_attribute, f"{multiplier}.input1.input1X")
    cmds.connectAttr(z_attribute, f"{multiplier}.input1.input1Z")
    cmds.setAttr(f"{multiplier}.input2.input2X", control_sensitivity[0] * u_range)
    cmds.setAttr(f"{multiplier}.input2.input2Z", -control_sensitivity[1] * v_range * uv_ratio)
    u_adder = cmds.createNode("addDoubleLinear", name=f"{name}_uOffsetAdd")
    v_adder = cmds.createNode("addDoubleLinear", name=f"{name}_vOffsetAdd")
    cmds.connectAttr(f"{multiplier}.output.outputX", f"{u_adder}.input1")
    cmds.connectAttr(f"{multiplier}.output.outputZ", f"{v_adder}.input1")
    if u_attribute:
        cmds.connectAttr(u_attribute, f"{u_adder}.input2")
    else:
        cmds.setAttr(f"{u_adder}.input2", default_u)
    if v_attribute:
        cmds.connectAttr(v_attribute, f"{v_adder}.input2")
    else:
        cmds.setAttr(f"{v_adder}.input2", default_v)

    u_clamp = cmds.createNode("clampRange", name=f"{name}_uClamp")
    v_clamp = cmds.createNode("clampRange", name=f"{name}_vClamp")
    cmds.connectAttr(f"{u_adder}.output", f"{u_clamp}.input")
    cmds.connectAttr(f"{v_adder}.output", f"{v_clamp}.input")
    cmds.setAttr(f"{u_clamp}.minimum", min_max_u[0])
    cmds.setAttr(f"{u_clamp}.maximum", min_max_u[1])
    cmds.setAttr(f"{v_clamp}.minimum", min_max_v[0])
    cmds.setAttr(f"{v_clamp}.maximum", min_max_v[1])

    cmds.connectAttr(f"{u_clamp}.output", f"{uv_pin_node}.coordinate[0].coordinateU")
    cmds.connectAttr(f"{v_clamp}.output", f"{uv_pin_node}.coordinate[0].coordinateV")

    if parent:
        cmds.parent(offset_transform, parent, relative=False)
    control = Control(
        control_transform=control_transform, offset_transform=offset_transform, name=name
    )
    tag_as_controller(control)
    return control


def connect_control(
    control: Control,
    driven_name: str,
    keep_offset: bool = False,
) -> None:
    matrix_constraint(
        source_transform=control.control_transform,
        constrain_transform=driven_name,
        keep_offset=keep_offset,
    )
