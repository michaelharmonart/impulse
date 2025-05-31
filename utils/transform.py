import maya.cmds as cmds

def match_transform(transform: str, target_transform: str, scale: bool = False) -> None:
    """
    Match a transform to another in world space.

    Args:
        transform: Object to be moved to the specified transform.
        target_transform: Name of the transform to parent control to.
        scale: If true, scale will be taken into account.
    """
    temp_locator = cmds.group(empty=True, parent=cmds.listRelatives(transform, parent=True)[0])
    cmds.parentConstraint(target_transform, temp_locator, maintainOffset=False)

    translation = cmds.getAttr(f"{temp_locator}.translate")[0]
    rotation = cmds.getAttr(f"{temp_locator}.rotate")[0]
    cmds.setAttr(f"{transform}.translate", *translation)
    cmds.setAttr(f"{transform}.rotate", *rotation)
    if scale:
        cmds.scaleConstraint(target_transform, temp_locator, maintainOffset=False)
        scale = cmds.getAttr(f"{temp_locator}.scale")[0]
        cmds.setAttr(f"{transform}.scale", *scale)

    cmds.delete(temp_locator)

def matrix_constraint(source_transform: str, constrain_transform: str, keep_offset: bool = True, use_parent: bool = False) -> None:
    """
    Constrain a transform to another 

    Args:
        source_transform: joint to match.
        constrain_joint: joint to constrain.
        keep_offset: keep the offset of the constrained transform to the source at time of constraint generation.
        
    """

    

    mult_index: int = 0
    mult_matrix: str = cmds.createNode("multMatrix", name=f"{constrain_transform}_ConstraintMultMatrix")

    if keep_offset:
        offset_matrix_node: str = cmds.createNode("multMatrix", name=f"{constrain_transform}_OffsetMatrix")
        cmds.connectAttr(f"{constrain_transform}.worldMatrix[0]", f"{offset_matrix_node}.matrixIn[0]")
        cmds.connectAttr(f"{source_transform}.worldInverseMatrix[0]", f"{offset_matrix_node}.matrixIn[1]")
        offset_matrix = cmds.getAttr(f"{offset_matrix_node}.matrixSum")
        cmds.setAttr(f"{mult_matrix}.matrixIn[{mult_index}]", offset_matrix, type="matrix")
        mult_index += 1
    
    cmds.connectAttr(f"{source_transform}.worldMatrix[0]", f"{mult_matrix}.matrixIn[{mult_index}]")
    mult_index += 1
    if use_parent:
        constraint_parent: str = cmds.listRelatives(constrain_transform, parent=True)[0]
        cmds.connectAttr(f"{constraint_parent}.worldInverseMatrix[0]", f"{mult_matrix}.matrixIn[{mult_index}]")
        cmds.setAttr(f"{constrain_transform}.inheritsTransform", 0)
        mult_index += 1

    decompose_matrix: str = cmds.createNode("decomposeMatrix", name=f"{constrain_transform}_ConstrainMatrixDecompose")
    cmds.connectAttr(f"{mult_matrix}.matrixSum", f"{decompose_matrix}.inputMatrix")
    
    cmds.connectAttr(f"{decompose_matrix}.outputTranslate", f"{constrain_transform}.translate")
    cmds.connectAttr(f"{decompose_matrix}.outputRotate", f"{constrain_transform}.rotate")
    cmds.connectAttr(f"{decompose_matrix}.outputScale", f"{constrain_transform}.scale")

def constrain_transforms(source_transforms: list[str], constrain_transforms: list[str]) -> None:
    """
    Constrain a set of transforms to another

    Args:
        source_transforms: joints to match.
        constrain_transforms: joints to constrain.
    """
    if len(source_transforms) != len(constrain_transforms):
        raise RuntimeError("Number of joints in source joints and constraint joints don't match!")
    for index, transform in enumerate(constrain_transforms):
        matrix_constraint(source_transform=source_transforms[index], constrain_transform=transform)
