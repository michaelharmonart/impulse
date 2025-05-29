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