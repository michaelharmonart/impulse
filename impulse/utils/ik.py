import maya.cmds as cmds


def ik_fk_blend(ik_joint: str, fk_joint: str, blended_joint:str, blend_attr: str) -> None:
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
    cmds.setAttr(f"{blended_joint}.jointOrient", 0,0,0, type="float3")

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
    if not(len(ik_joints) == len(fk_joints) == len(blended_joints)):
        raise RuntimeError("Number of fk ik and blend joints don't match!")
    for index, blended_joint in enumerate(blended_joints):
        ik_fk_blend(ik_joint=ik_joints[index], fk_joint=fk_joints[index], blended_joint=blended_joint, blend_attr=blend_attr)

def blend_selected(blend_attr: str) -> None:
    selection: list[str] = cmds.ls(selection=True)
    ik_fk_blend(ik_joint=selection[0], fk_joint=selection[1], blended_joint=selection[2], blend_attr=blend_attr)