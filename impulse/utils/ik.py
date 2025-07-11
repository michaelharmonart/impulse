import maya.cmds as cmds

from impulse.utils.transform import match_transform, matrix_constraint


def ik_from_guides(
    guides: list[str],
    pole_vector_guide: str,
    reverse_segments: int = 0,
    stretch: bool = False,
    name: str | None = None,
    parent: str | None = None,
    suffix: str = "_IK",
) -> str:
    """
    Takes a hierarchy of guides and creates an IK chain.
    Args:
        guides: The guides that will become the IK joints.
        pole_vector: The guide for placing the pole vector.
        reverse_segments: How many of the segments on the chain should be reverse IK.
        include_end_joint: When True, the last Guide given (the end joint) will also be included as a joint in the deformation chain
        stretch: If True, will make limb stretch when going past full extension.
        name: Name for the newly created IK Chain group.
        parent: Parent for the newly created IK Chain group.
        suffix: Suffix to be added to joint names and IK chain group.
    Returns:
        list[str]: Name of the created IK joints.
    """
    reverse: bool = reverse_segments != 0

    if not name:
        name: str = f"{guides[0].rsplit('_', 1)[0]}{suffix}"
    # Create group for IK chain
    ik_group: str = cmds.group(empty=True, world=True, name=name)
    if parent:
        cmds.parent(ik_group, parent, relative=False)

    ik_guides: list[str] = guides
    reverse_guides: list[str] = []
    if reverse:
        reverse_guides: list[str] = guides[-(reverse_segments+1):]
        if reverse_segments > 0:
            ik_guides: list[str] = guides[:-(reverse_segments)]


    # Duplicating and rename the IK guides
    ik_joints: list[str] = []
    for index, guide in enumerate(ik_guides):
        ik_joint: str = cmds.duplicate(guide, name=f"{guide}{suffix}", parentOnly=True)[0]
        if index > 0:
            cmds.parent(ik_joint, ik_joints[index - 1])
        else:
            cmds.parent(ik_joint, ik_group)
        
        if reverse:
            if index == len(ik_guides) - 1:
                ik_joint = cmds.rename(ik_joint, f"{name}_Effector")
        ik_joints.append(ik_joint)
    
    # Duplicating and rename the reverse guides
    reversed_reverse_guides: list[str] = reverse_guides[::-1]
    reverse_joints: list[str] = []
    for index, guide in enumerate(reversed_reverse_guides):
        reverse_joint: str = cmds.duplicate(guide, name=f"{guide}{suffix}", parentOnly=True)[0]
        reverse_joints.append(reverse_joint)
        if index > 0:
            cmds.parent(reverse_joint, reverse_joints[index])
        else:
            cmds.parent(reverse_joint, ik_group)
    reverse_joints.reverse()

    ik_chain: list[str] = ik_joints
    if reverse:
        ik_chain: list[str] = ik_joints[:-1] + reverse_joints

    # Create IK Handle
    ik_handle: str = cmds.ikHandle(startJoint=ik_joints[0], endEffector=ik_joints[-1], name=f"{name}_ikHandle")[0]
    if reverse:
        cmds.parent(ik_handle, reverse_joints[0])
    else:
        cmds.parent(ik_handle, ik_group)

    # Create a transform for the Pole Vector and constrain ikHandle to it.
    pole_vector: str = cmds.group(empty=True, name=f"{pole_vector_guide}_IN", parent=ik_group)
    match_transform(transform=pole_vector, target_transform=pole_vector_guide)
    cmds.poleVectorConstraint(pole_vector, ik_handle)

    # If set to have stretch, create the relevant nodes and attach them.
    if stretch:
        
        # Create nodes to measure distance
        start_joint_local: str = cmds.group(empty=True, name=f"{ik_joints[0]}_LOCAL", parent=ik_group)
        matrix_constraint(ik_joints[0], start_joint_local)

        distance_node: str = cmds.createNode("distanceBetween", name=f"{name}_dist")
        cmds.connectAttr(f"{ik_joints[0]}.worldMatrix", f"{distance_node}.inMatrix1")
        cmds.connectAttr(f"{ik_joints[-1]}.worldMatrix", f"{distance_node}.inMatrix2")

        # Get rest length and a normalized distance multiplier that tells us how much longer or shorter the limb is.
        rest_distance = cmds.getAttr(f"{distance_node}.distance")
        normalize_node: str = cmds.createNode("divide", name=f"{name}_dist_norm")
        cmds.connectAttr(f"{distance_node}.distance", f"{normalize_node}.input1")
        cmds.setAttr(f"{normalize_node}.input2", rest_distance)

        # Only stretch, dont shrink the limb.
        condition_node: str = cmds.createNode("condition", name=f"{name}_dist_cond")
        cmds.setAttr(f"{condition_node}.operation", 2) # Greater than
        cmds.setAttr(f"{condition_node}.secondTerm", 1)
        cmds.connectAttr(f"{normalize_node}.output", f"{condition_node}.firstTerm")
        cmds.connectAttr(f"{normalize_node}.output", f"{condition_node}.colorIfTrueR")
        scale_factor_attr = f"{condition_node}.outColorR"

        for joint in ik_joints[1:-1]:
            # Get the rest Y position, multiply by the scale factor, and pump it into the Y position.
            joint_y_adjust = cmds.createNode("multDoubleLinear", name=f"{joint}_yAdjust")
            cmds.connectAttr(scale_factor_attr, f"{joint_y_adjust}.input1")
            rest_y = cmds.getAttr(f"{joint}.translate.translateY")
            cmds.setAttr(f"{joint_y_adjust}.input2", rest_y)
            cmds.connectAttr(f"{joint_y_adjust}.output", f"{joint}.translate.translateY")

    return ik_chain


def fk_from_guides(
    guides: list[str], name: str | None = None, parent: str | None = None, suffix: str = "_FK", side_mult: int = 1
) -> str:
    """
    Takes a hierarchy of guides and creates an FK chain.
    Args:
        guides: The guides that will become the FK joints.
        name: Name for the newly created FK Chain group.
        parent: Parent for the newly created FK Chain group.
        suffix: Suffix to be added to joint names and FK chain group.
        side_mult: Allows for scaling of the parent group for mirroring purposes.
    Returns:
        list[str]: Names of the created FK joints.
    """
    if not name:
        name: str = f"{guides[0].rsplit('_', 1)[0]}{suffix}"
    # Create group for FK chain
    fk_group: str = cmds.group(empty=True, world=True, name=name)
    cmds.scale(side_mult, 1, 1, fk_group, absolute=True)
    if parent:
        cmds.parent(fk_group, parent, relative=False)

    # Start by duplicating and renaming the guides
    fk_joints: list[str] = []
    for index, guide in enumerate(guides):
        fk_joint: str = cmds.duplicate(guide, name=f"{guide}{suffix}", parentOnly=True)[0]
        fk_joints.append(fk_joint)
        if index > 0:
            cmds.parent(fk_joint, fk_joints[index - 1])
        else:
            cmds.parent(fk_joint, fk_group)

    return fk_joints


def ik_fk_blend(ik_joint: str, fk_joint: str, blended_joint: str, blend_attr: str) -> None:
    """
    Takes two joints and blends their transforms based on an attribute. joints need the same hierarchy and orients.
    Args:
        ik_joint: The IK joint of the blend.
        fk_joint: The FK joint of the blend.
        blended_joint: The joint to give the blended transform
        blend_attr: Attribute to use to determine blending.
    """
    # Create Blend Matrix node and connect it
    blend_matrix: str = cmds.createNode("blendMatrix", name=f"{blended_joint}_BlendMatrix")
    cmds.connectAttr(f"{ik_joint}.matrix", f"{blend_matrix}.inputMatrix")
    cmds.connectAttr(f"{fk_joint}.matrix", f"{blend_matrix}.target[0].targetMatrix")
    cmds.connectAttr(blend_attr, f"{blend_matrix}.target[0].weight")

    # Create the Decomposed Matrix and connect its input
    decompose_matrix: str = cmds.createNode("decomposeMatrix", name=f"{blended_joint}_BlendMatrixDecompose")
    cmds.connectAttr(f"{blend_matrix}.outputMatrix", f"{decompose_matrix}.inputMatrix")

    # Reset blended joint orient and connect the joint to the decomposeMatrix values
    cmds.setAttr(f"{blended_joint}.jointOrient", 0, 0, 0, type="float3")

    cmds.connectAttr(f"{decompose_matrix}.outputRotate", f"{blended_joint}.rotate")
    cmds.connectAttr(f"{decompose_matrix}.outputTranslate", f"{blended_joint}.translate")
    cmds.connectAttr(f"{decompose_matrix}.outputScale", f"{blended_joint}.scale")
    cmds.connectAttr(f"{decompose_matrix}.outputShear", f"{blended_joint}.shear")


def ik_fk_blend_list(ik_joints: list[str], fk_joints: list[str], blended_joints: list[str], blend_attr: str) -> None:
    """
    Takes two lists of joints and blends their transforms based on an attribute. joints need the same hierarchy and orients.
    Args:
        ik_joints: The IK joints of the blend.
        fk_joints: The FK joints of the blend.
        blended_joints: The joints to give the blended transform
        blend_attr: Attribute to use to determine blending.
    """
    if not (len(ik_joints) == len(fk_joints) == len(blended_joints)):
        raise RuntimeError("Number of fk ik and blend joints don't match!")
    for index, blended_joint in enumerate(blended_joints):
        ik_fk_blend(
            ik_joint=ik_joints[index], fk_joint=fk_joints[index], blended_joint=blended_joint, blend_attr=blend_attr
        )


def blend_selected(blend_attr: str) -> None:
    selection: list[str] = cmds.ls(selection=True)
    ik_fk_blend(ik_joint=selection[0], fk_joint=selection[1], blended_joint=selection[2], blend_attr=blend_attr)
