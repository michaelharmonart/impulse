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
    translate_blend: str = cmds.createNode("blendColors", name=f"{blended_joint}_BlendTranslate")
    cmds.connectAttr(blend_attr, f"{translate_blend}.blender")
    cmds.connectAttr(f"{ik_joint}.translate", f"{translate_blend}.color2")
    cmds.connectAttr(f"{fk_joint}.translate", f"{translate_blend}.color1")
    cmds.connectAttr(f"{translate_blend}.output", f"{blended_joint}.translate")

    rotation_blend: str = cmds.createNode("blendColors", name=f"{blended_joint}_BlendRotation")
    cmds.connectAttr(blend_attr, f"{rotation_blend}.blender")
    cmds.connectAttr(f"{ik_joint}.rotate", f"{rotation_blend}.color2")
    cmds.connectAttr(f"{fk_joint}.rotate", f"{rotation_blend}.color1")
    cmds.connectAttr(f"{rotation_blend}.output", f"{blended_joint}.rotate")

    scale_blend: str = cmds.createNode("blendColors", name=f"{blended_joint}_BlendScale")
    cmds.connectAttr(blend_attr, f"{scale_blend}.blender")
    cmds.connectAttr(f"{ik_joint}.scale", f"{scale_blend}.color2")
    cmds.connectAttr(f"{fk_joint}.scale", f"{scale_blend}.color1")
    cmds.connectAttr(f"{scale_blend}.output", f"{blended_joint}.scale")

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