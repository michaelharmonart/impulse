import maya.cmds as cmds
import maya.mel as mel
from enum import Enum

class ControlShape(Enum):
    """Enum for available control shapes."""
    CIRCLE = 0
    TRIANGLE = 1
    SQUARE = 2
    PILL = 3
    

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
        position: tuple[float, float, float],
        size: float,
        parent: str,
        control_shape: ControlShape = ControlShape.CIRCLE,
        offset: float = 0.1,
) -> str:
    """
    Create a control curve in Maya at a given position, scale it, offset it,
    and parent it under the specified transform.
    
    Args:
        position: (x, y, z) coordinates for the control's final position.
        size: Scaling factor for the control curve.
        parent: Name of the parent transform.
        control_shape: The type of control shape to create.
        offset: Vertical offset applied immediately after creation.
    
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
    cmds.scale(size, size, size, relative=True)
    cmds.move(0, offset, 0, relative=True)
    cmds.makeIdentity(apply=True)
    cmds.xform(control_transform, pivots=(0, 0, 0))
    cmds.move(position[0], position[1], position[2], relative=True, worldSpace=True)
    cmds.parent(control_transform, parent)
    
    return control_transform