import maya.cmds as cmds
import maya.mel as mel
from . import pin as pin
from . import math as math
from enum import Enum
import warnings

class ControlShape(Enum):
    """Enum for available control shapes."""
    CIRCLE = 0
    TRIANGLE = 1
    SQUARE = 2
    PILL = 3

class Direction(Enum):
    """Enum for available control directions"""
    X = 0
    Y = 1
    Z = 2

# Mapping from control shape to its corresponding MEL command string.
MEL_COMMANDS = {
    ControlShape.CIRCLE: (
        """//ML Control Curve: circle
string $ml_tempCtrlName = `createNode transform -n "circle_#"`;
createNode nurbsCurve -p $ml_tempCtrlName;
setAttr -k off ".v";
setAttr ".cc" -type "nurbsCurve" 
3 8 2 no 3
13 -2 -1 0 1 2 3 4 5 6 7 8
 9 10
11
0.78361162489122504 4.7982373409884682e-17 -0.78361162489122382
-1.2643170607829326e-16 6.7857323231109134e-17 -1.1081941875543879
-0.78361162489122427 4.7982373409884713e-17 -0.78361162489122427
-1.1081941875543879 1.9663354616187859e-32 -3.2112695072372299e-16
-0.78361162489122449 -4.7982373409884694e-17 0.78361162489122405
-3.3392053635905195e-16 -6.7857323231109146e-17 1.1081941875543881
0.78361162489122382 -4.7982373409884719e-17 0.78361162489122438
1.1081941875543879 -3.6446300679047921e-32 5.9521325992805852e-16
0.78361162489122504 4.7982373409884682e-17 -0.78361162489122382
-1.2643170607829326e-16 6.7857323231109134e-17 -1.1081941875543879
-0.78361162489122427 4.7982373409884713e-17 -0.78361162489122427
;"""
    ),
    ControlShape.TRIANGLE: (
        """//ML Control Curve: triangle
string $ml_tempCtrlName = `createNode transform -n "triangle_#"`;
createNode nurbsCurve -p $ml_tempCtrlName;
setAttr -k off ".v";
setAttr ".cc" -type "nurbsCurve" 
3 8 2 no 3
13 -2 -1 0 1 2 3 4 5 6 7 8
 9 10
11
0.60438580191109736 -5.6848212903126036e-06 1.9306698734455003
-8.390276879065607e-15 -5.6848212900327269e-06 2.0300870557164479
-0.60438580191111402 -5.6848212897087013e-06 1.9306698734455012
-2.0300870557162964 -5.684821288567725e-06 1.6091162660778342e-13
-1.9306698734453507 -5.6848212881886989e-06 -1.9306698734451799
-1.0045870072155691e-14 -5.6848212891311874e-06 -2.0300870557161272
1.93066987344533 -5.6848212901178249e-06 -1.9306698734451804
2.0300870557162782 -5.6848212905961893e-06 1.5954439019415462e-13
0.60438580191109736 -5.6848212903126036e-06 1.9306698734455003
-8.390276879065607e-15 -5.6848212900327269e-06 2.0300870557164479
-0.60438580191111402 -5.6848212897087013e-06 1.9306698734455012
;"""
    ),
    ControlShape.PILL: (
        """//ML Control Curve: pill
string $ml_tempCtrlName = `createNode transform -n "pill_#"`;
createNode nurbsCurve -p $ml_tempCtrlName;
setAttr -k off ".v";
setAttr ".cc" -type "nurbsCurve" 
3 16 2 no 3
21 -0.125 -0.0625 0 0.0625 0.125 0.1875 0.25 0.3125 0.375 0.4375 0.5
 0.5625 0.625 0.6875 0.75 0.8125 0.875 0.93750000000000011 1 1.0625 1.125
19
1.172939121998376 1.0774315860976344e-16 0.75127324711188403
1.1729402050761661 1.1696140604314265e-16 -8.2866285639933504e-16
1.1729391219983754 1.0774315860976344e-16 -0.71992924706777273
1.1729361791220005 8.2695869434192015e-17 -1.1683230822955293
0.76125060477226536 4.4629742975787312e-17 -1.8477644801561781
1.5646587498839395e-05 9.5807716501497136e-22 -2.0736340129126583
-0.76128837655064607 -4.4632055830161894e-17 -1.847764480156181
-1.1727690100482948 -8.2692163614709417e-17 -1.1683230822955237
-1.1727719541927091 -1.077502452723395e-16 -0.71992924706777384
-1.1727730344679179 -1.1694463949820226e-16 1.8381251164062978e-16
-1.1727719541927084 -1.0775024527233942e-16 0.75127324711188448
-1.1727690100482939 -8.269216361470954e-17 1.215219870343615
-0.74509258124420297 -4.463205583016166e-17 1.8390002733665078
1.5646587500656142e-05 9.5807716494482651e-22 2.0752540628040212
0.74505480946583147 4.4629742975787534e-17 1.8390002733665074
1.1729361791220014 8.2695869434192064e-17 1.2152198703436123
1.172939121998376 1.0774315860976344e-16 0.75127324711188403
1.1729402050761661 1.1696140604314265e-16 -8.2866285639933504e-16
1.1729391219983754 1.0774315860976344e-16 -0.71992924706777273
;"""
    ),
    ControlShape.SQUARE: (
        """//ML Control Curve: square
string $ml_tempCtrlName = `createNode transform -n "square_#"`;
createNode nurbsCurve -p $ml_tempCtrlName;
setAttr ".cc" -type "nurbsCurve" 
3 8 2 no 3
13 -2 -1 0 1 2 3 4 5 6 7 8
    9 10
11
1.9033843953925922 1.1654868036822793e-16 -1.9033843953925924
1.241723479304563e-16 1.241723479304563e-16 -2.0278883351005348
-1.9033843953925922 1.165486803682279e-16 -1.903384395392592
-2.0278883351005357 -8.3314952071146944e-33 9.4409652395123731e-17
-1.9033843953925922 -1.165486803682279e-16 1.9033843953925922
-2.0313496146335863e-16 -1.241723479304564e-16 2.0278883351005361
1.9033843953925922 -1.165486803682279e-16 1.903384395392592
2.0278883351005357 -3.1701950895796529e-32 4.7607815804023312e-16
1.9033843953925922 1.1654868036822793e-16 -1.9033843953925924
1.241723479304563e-16 1.241723479304563e-16 -2.0278883351005348
-1.9033843953925922 1.165486803682279e-16 -1.903384395392592
;"""
    ),
}


def generate_control(
        name: str,
        parent: str = None,
        position: tuple[float, float, float] = (0,0,0),
        direction: Direction = Direction.Y,
        opposite_direction: bool = False,
        size: float = 1,
        control_shape: ControlShape = ControlShape.CIRCLE,
        offset: float = 0.1,
) -> str:
    """
    Create a control curve in Maya at a given position, scale it, offset it,
    and parent it under the specified transform.
    
    Args:
        position: (x, y, z) coordinates for the control's final position.
        parent: Name of the parent transform.
        direction: Direction control shape will face.
        size: Scaling factor for the control curve.
        control_shape: The type of control shape to create.
        offset: Vertical offset applied immediately after creation.
        use_opm: Use offset parent matrix instead of offset transform group (cleaner hierarchy) 
    
    Returns:
        The name of the created control transform.
    """
    
    # Retrieve the MEL command for the desired control shape.
    mel_command = MEL_COMMANDS.get(control_shape)
    if mel_command is None:
        raise ValueError(f"Unsupported control shape: {control_shape}")

    # Execute the MEL command to create the control curve.
    mel.eval(mel_command)
    
    # Retrieve the created control (assumes the curve is currently selected).
    selection = cmds.ls(selection=True)
    if not selection:
        raise RuntimeError("No control created; check the MEL command output.")
    
    # Get the transform node from the selected curve.
    control_transform = cmds.listRelatives(selection[0], parent=True)
    if not control_transform:
        raise RuntimeError("Control creation failed; no parent transform found.")
    control_transform = control_transform[0]

    # Adjust the control's scale, apply an offset, reset transforms, and reposition.
    cmds.select(control_transform)
    control_transform = cmds.rename(control_transform, f"{name}_CTL")
    cmds.scale(size, size, size, relative=True)
    cmds.move(0, offset, 0, relative=True)
    cmds.makeIdentity(apply=True)
    cmds.xform(control_transform, pivots=(0, 0, 0))

    if direction == Direction.X:
        if opposite_direction:
            cmds.rotate(0,0,90) 
        else:
            cmds.rotate(0,0,-90)
    elif direction == Direction.Z:
        if opposite_direction:
            cmds.rotate(-90,0,0) 
        else:
            cmds.rotate(90,0,0)
    cmds.makeIdentity(apply=True)
    cmds.xform(control_transform, pivots=(0, 0, 0))

    control_transform = cmds.group(control_transform, name=f"{name}_OFFSET")
    cmds.move(position[0], position[1], position[2], relative=True, worldSpace=True)
    if parent:
        cmds.parent(control_transform, parent)
    
    return control_transform

def generate_surface_control(
        name: str,
        surface: str,
        parent: str = None,
        position: tuple[float, float, float] = (0,0,0),
        match_transform: str = None,
        uv_position: tuple[float, float] = None,
        u_attribute: str = None,
        v_attribute: str = None,
        control_sensitivity: tuple[float, float] = (1,1),
        direction: Direction = Direction.Y,
        opposite_direction: bool = False,
        size: float = 1,
        control_shape: ControlShape = ControlShape.CIRCLE,
        offset: float = 0.1,
) -> str:
    """
    Create a control that moves only along a given surface (plus an offset).
    
    Args:
        surface: The surface (mesh or NURBS) that the control will move along.
        parent: Name of the transform to parent control to.
        position: World space position that will be projected onto the surface to set the default control position in UV space.
        match_transform: when set, the control will be generated to match the position and angle of the given transform node.
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
    
    

    # Retrieve the MEL command for the desired control shape.
    mel_command = MEL_COMMANDS.get(control_shape)
    if mel_command is None:
        raise ValueError(f"Unsupported control shape: {control_shape}")

    # Execute the MEL command to create the control curve.
    mel.eval(mel_command)
    
    # Retrieve the created control (assumes the curve is currently selected).
    selection = cmds.ls(selection=True)
    if not selection:
        raise RuntimeError("No control created; check the MEL command output.")
    
    # Get the transform node from the selected curve.
    control_transform = cmds.listRelatives(selection[0], parent=True)
    if not control_transform:
        raise RuntimeError("Control creation failed; no parent transform found.")
    control_transform = control_transform[0]


    # Adjust the control's scale, apply an offset, reset transforms, and reposition.
    cmds.select(control_transform)
    control_transform= cmds.rename(control_transform, f"{name}_CTL")
    cmds.scale(size, size, size, relative=True)
    cmds.move(0, offset, 0, relative=True)
    cmds.makeIdentity(apply=True)
    cmds.xform(control_transform, pivots=(0, 0, 0))

    if direction == Direction.X:
        if opposite_direction:
            cmds.rotate(0,0,90) 
        else:
            cmds.rotate(0,0,-90)
    elif direction == Direction.Z:
        if opposite_direction:
            cmds.rotate(-90,0,0) 
        else:
            cmds.rotate(90,0,0)
    cmds.makeIdentity(apply=True)
    cmds.xform(control_transform, pivots=(0, 0, 0))

    control_static_group = cmds.group(control_transform, name=f"{name}_STATIC")
    offset_matrix_compose = cmds.createNode("composeMatrix", name=f"{control_transform}_offsetMatrixCompose")
    cmds.connectAttr(f"{control_transform}.translate.translateX",f"{offset_matrix_compose}.inputTranslate.inputTranslateX")
    cmds.connectAttr(f"{control_transform}.translate.translateZ",f"{offset_matrix_compose}.inputTranslate.inputTranslateZ")
    matrix_inverse = cmds.createNode("inverseMatrix", name=f"{control_transform}_inverseMatrix")
    cmds.connectAttr(f"{offset_matrix_compose}.outputMatrix",f"{matrix_inverse}.inputMatrix")  
    cmds.connectAttr(f"{matrix_inverse}.outputMatrix",f"{control_static_group}.offsetParentMatrix") 
       
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
    
    #Create temp node to get the UV of the closest point on the surface to the default location of this control.
    rotation_y = 0
    cp_node = cmds.createNode(cp_node_type, name=f"{name}_closestPoint_TEMP")
    cmds.connectAttr(f"{shape}{attr_world}", f"{cp_node}{cp_input}")
    if surface_type == "mesh":
        cmds.connectAttr(f"{shape}.worldMatrix[0]", f"{cp_node}.inputMatrix")
    
    x_attribute = f"{control_transform}.translate.translateX"
    z_attribute = f"{control_transform}.translate.translateZ"
    if match_transform:
        position = cmds.xform(match_transform, worldSpace=True, query=True, translation=True)
        rotate_matrix = cmds.createNode("composeMatrix", name=f"{name}_rotationMatrixCompose")
        cmds.connectAttr(f"{offset_transform}.rotate.rotateY", f"{rotate_matrix}.inputRotate.inputRotateY")
        translate_matrix = cmds.createNode("composeMatrix", name=f"{name}_translateMatrixCompose")
        cmds.connectAttr(f"{control_transform}.translate.translateX", f"{translate_matrix}.inputTranslate.inputTranslateX")
        cmds.connectAttr(f"{control_transform}.translate.translateZ", f"{translate_matrix}.inputTranslate.inputTranslateZ")
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
    u_range: float = min_max_u[1]-min_max_u[0]
    v_range: float = min_max_v[1]-min_max_v[0]
    uv_ratio: float = u_range/v_range

    uv_pin_node = pin.make_uv_pin(object_to_pin=offset_transform, surface=surface, u= default_u, v=default_v, normalize=False)
    if match_transform:
        temp_locator = cmds.group(empty=True, parent=offset_transform)
        cmds.parentConstraint(match_transform, temp_locator, maintainOffset=False)
        rotation_y = cmds.getAttr(f"{temp_locator}.rotate.rotateY")
        cmds.delete(temp_locator)
        cmds.xform(offset_transform, rotation=(0,rotation_y,0), worldSpace=False)

    multiplier = cmds.createNode("multiplyDivide", name=f"{name}_sensitivityMultiply")
    cmds.connectAttr(x_attribute, f"{multiplier}.input1.input1X")
    cmds.connectAttr(z_attribute , f"{multiplier}.input1.input1Z")
    cmds.setAttr(f"{multiplier}.input2.input2X", control_sensitivity[0]*u_range)
    cmds.setAttr(f"{multiplier}.input2.input2Z", -control_sensitivity[1]*v_range*uv_ratio)
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
        cmds.parent(offset_transform, parent)
    
    return offset_transform

def connect(
        control_name: str,
        driven_name: str,
        connect_scale: bool = True,
) -> None:
    children = cmds.listRelatives(control_name, allDescendents=True, type="transform") or []
    if len(children) == 0:
        raise RuntimeError(f"{control_name} doesn't have any child transforms that could be a control shape, is it a valid control? Here are it's children: {children}")
    control_transform: str = None
    for child in children:
        if child.endswith("_CTL"):
            control_transform: str = child
    if control_transform is None:
        raise RuntimeError(f"{control_name} doesn't have a child transform ending in CTL, is it a valid control? Here are it's children: {children}")
    cmds.parentConstraint(control_transform, driven_name, weight=1)
    if connect_scale:
        cmds.scaleConstraint(control_transform, driven_name, weight=1)

def get_control_transform(
        control_name: str,
) -> str:
    children = cmds.listRelatives(control_name, allDescendents=True, type="transform") or []
    if len(children) == 0:
        raise RuntimeError(f"{control_name} doesn't have any child transforms that could be a control shape, is it a valid control? Here are it's children: {children}")
    control_transform: str = None
    for child in children:
        if child.endswith("_CTL"):
            control_transform: str = child
    if control_transform is None:
        raise RuntimeError(f"{control_name} doesn't have a child transform ending in CTL, is it a valid control? Here are it's children: {children}")
    return control_transform