import maya.cmds as cmds

from impulse.maya_api import node
from impulse.utils.transform import match_transform, matrix_constraint


class IkChain:
    """
    A container class representing a constructed IK chain setup.

    Attributes:
        ik_chain_joints (list[str]): The list of joint names that make up the IK chain, including any reverse segments.
        socket (str): Name of the transform that is the socket or attachment point for the chain.
        pole_vector (str): The name of the transform used as the pole vector controller.
    """

    def __init__(self, ik_chain_joints: list[str], socket: str, pole_vector: str, ik_handle: str):
        self.ik_chain_joints = ik_chain_joints
        self.socket = socket
        self.pole_vector = pole_vector
        self.ik_handle = ik_handle


def ik_from_guides(
    guides: list[str],
    pole_vector_guide: str,
    reverse_segments: int = 0,
    stretch: bool = True,
    name: str | None = None,
    parent: str | None = None,
    suffix: str = "_IK",
) -> IkChain:
    """
    Takes a hierarchy of guides and creates an IK chain.
    Args:
        guides: The guides that will become the IK joints.
        pole_vector: The guide for placing the pole vector.
        reverse_segments: How many of the segments on the chain should be reverse IK.
        stretch: If True, will make limb stretch when going past full extension.
        name: Name for the newly created IK Chain group.
        parent: Parent for the newly created IK Chain group.
        suffix: Suffix to be added to joint names and IK chain group.
    Returns:
        IkChain: Object containing relevant data about the generated Ik Chain.
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
        reverse_guides: list[str] = guides[-(reverse_segments + 1) :]
        if reverse_segments > 0:
            ik_guides: list[str] = guides[:-(reverse_segments)]

    # Duplicating and rename the IK guides
    ik_joints: list[str] = []
    for index, guide in enumerate(ik_guides):
        ik_joint: str = cmds.duplicate(guide, name=f"{guide}{suffix}", parentOnly=True)[0]
        if stretch:
            cmds.setAttr(f"{ik_joint}.segmentScaleCompensate", 1)
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

    # Create and connect Socket
    socket: str = cmds.group(empty=True, name=f"{name}_Socket", parent=ik_group)
    match_transform(transform=socket, target_transform=guides[0])
    cmds.pointConstraint(socket, ik_chain[0])

    # Create IK Handle
    ik_handle: str = cmds.ikHandle(
        startJoint=ik_joints[0], endEffector=ik_joints[-1], name=f"{name}_ikHandle"
    )[0]
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
        socket_local: str = cmds.group(empty=True, name=f"{socket}_LOCAL", parent=ik_group)
        matrix_constraint(
            source_transform=socket, constrain_transform=socket_local, keep_offset=False
        )

        handle_local: str = cmds.group(empty=True, name=f"{ik_handle}_LOCAL", parent=ik_group)
        matrix_constraint(ik_handle, handle_local, keep_offset=False)

        distance_node = node.DistanceBetweenNode(name=f"{name}_dist")
        cmds.connectAttr(f"{socket_local}.matrix", distance_node.input_matrix1)
        cmds.connectAttr(f"{handle_local}.matrix", distance_node.input_matrix2)

        # Get rest length and a normalized distance multiplier that tells us how much longer or shorter the limb is.
        rest_length: float = 0
        temp_distance_node = node.DistanceBetweenNode(name=f"{name}_temp_dist")
        for index, joint in enumerate(ik_joints):
            if index == 0:
                continue
            cmds.connectAttr(f"{joint}.matrix", temp_distance_node.input_matrix2)
            rest_length += cmds.getAttr(temp_distance_node.distance)
            cmds.disconnectAttr(f"{joint}.matrix", temp_distance_node.input_matrix2)

        normalize_node = node.DivideNode(name=f"{name}_dist_norm")
        cmds.connectAttr(distance_node.distance, normalize_node.input1)
        cmds.setAttr(normalize_node.input2, rest_length)

        # Only stretch, dont shrink the limb.
        condition_node: str = cmds.createNode("condition", name=f"{name}_dist_cond")
        cmds.setAttr(f"{condition_node}.operation", 2)  # Greater than
        cmds.setAttr(f"{condition_node}.secondTerm", 1)
        cmds.connectAttr(normalize_node.output, f"{condition_node}.firstTerm")
        cmds.connectAttr(normalize_node.output, f"{condition_node}.colorIfTrueR")
        scale_factor_attr = f"{condition_node}.outColorR"

        for joint in ik_joints[0:-1]:
            cmds.connectAttr(scale_factor_attr, f"{joint}.scale.scaleY")

    return IkChain(
        ik_chain_joints=ik_chain, socket=socket, pole_vector=pole_vector, ik_handle=ik_handle
    )


def fk_from_guides(
    guides: list[str],
    name: str | None = None,
    parent: str | None = None,
    suffix: str = "_FK",
    side_mult: int = 1,
    include_last: bool = True,
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
    filtered_guides: list[str] = guides
    if not include_last:
        filtered_guides: list[str] = guides[:-1]
    fk_joints: list[str] = []
    for index, guide in enumerate(filtered_guides):
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
    decompose_matrix: str = cmds.createNode(
        "decomposeMatrix", name=f"{blended_joint}_BlendMatrixDecompose"
    )
    cmds.connectAttr(f"{blend_matrix}.outputMatrix", f"{decompose_matrix}.inputMatrix")
    cmds.connectAttr(f"{blended_joint}.rotateOrder", f"{decompose_matrix}.inputRotateOrder")

    # Reset blended joint orient and connect the joint to the decomposeMatrix values
    cmds.setAttr(f"{blended_joint}.jointOrient", 0, 0, 0, type="float3")

    cmds.connectAttr(f"{decompose_matrix}.outputRotate", f"{blended_joint}.rotate")
    cmds.connectAttr(f"{decompose_matrix}.outputTranslate", f"{blended_joint}.translate")
    cmds.connectAttr(f"{decompose_matrix}.outputScale", f"{blended_joint}.scale")
    cmds.connectAttr(f"{decompose_matrix}.outputShear", f"{blended_joint}.shear")


def ik_fk_blend_list(
    ik_joints: list[str], fk_joints: list[str], blended_joints: list[str], blend_attr: str
) -> None:
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
            ik_joint=ik_joints[index],
            fk_joint=fk_joints[index],
            blended_joint=blended_joint,
            blend_attr=blend_attr,
        )


def blend_selected(blend_attr: str) -> None:
    selection: list[str] = cmds.ls(selection=True)
    ik_fk_blend(
        ik_joint=selection[0],
        fk_joint=selection[1],
        blended_joint=selection[2],
        blend_attr=blend_attr,
    )
