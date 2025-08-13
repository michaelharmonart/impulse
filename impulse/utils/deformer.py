import maya.cmds as cmds


def apply_proximity_wrap(driver: str, target: str) -> str:
    wrap_node = cmds.deformer(target, type="proximityWrap")[0]
    cmds.proximityWrap(wrap_node, edit=True, addDrivers=[driver])
