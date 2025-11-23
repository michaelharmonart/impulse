from impulse.maya_api import node
from impulse.utils.transform import matrix_constraint
import maya.cmds as cmds

def create_iris_drivers(name: str, loops: int = 8, iris_loops: int = 2, iris_scale: int = 30, debug=False):
    part_name = name
    group = cmds.group(empty=True, name=f"{part_name}_GRP")
    cmds.addAttr(group, longName="irisScale", attributeType="double")
    scale_attr = f"{group}.irisScale"
    cmds.setAttr(scale_attr, iris_scale)
    cmds.setAttr(scale_attr, keyable=True)

    scale_sensitivity = node.MultiplyNode(name=f"{part_name}_irisScale_Sensitivity")
    cmds.connectAttr(scale_attr, scale_sensitivity.input[0])
    cmds.setAttr(scale_sensitivity.input[1], 1/100)

    scale_map = node.LerpNode(name=f"{part_name}_irisScale_Map")
    cmds.setAttr(scale_map.input1, 90)
    cmds.setAttr(scale_map.input2, 0)
    cmds.connectAttr(scale_sensitivity.output, scale_map.weight)

    def_group = cmds.group(empty=True, name=f"{part_name}_DEF", parent=group)
    mch_group = cmds.group(empty=True, name=f"{part_name}_MCH", parent=group)
    for i in range(loops):
        if i < iris_loops:
            iris_i = i
            iris=True
        else:
            iris=False
        if i >= iris_loops:
            white_i = i - iris_loops
            white=True
        else:
            white=False
        white_loops = loops - iris_loops
        driver= cmds.group(empty=True, name=f"{part_name}_Driver{i:02d}", parent=mch_group)

        if debug:
            debug_shape = cmds.circle()[0]
            cmds.parent(debug_shape, group)

        joint: str = cmds.joint(name=f"{part_name}_{i:02d}_DEF")
        cmds.parent(joint, def_group, relative=True)
        matrix_constraint(driver, joint, keep_offset=False)
        matrix_constraint(driver, debug_shape, keep_offset=False)

        remap = node.LerpNode(name=f"{driver}_Map")
        if iris:
            cmds.setAttr(remap.input1, 90)
            cmds.connectAttr(scale_map.output, remap.input2)
            cmds.setAttr(remap.weight, (iris_i+1)/(iris_loops))
        if white:
            cmds.connectAttr(scale_map.output, remap.input1)
            cmds.setAttr(remap.input2, 0)
            cmds.setAttr(remap.weight, (white_i+1)/(white_loops))
            
        sin_node = node.SinNode(name=f"{driver}_Sin")
        cos_node = node.CosNode(name=f"{driver}_Cos")
        cmds.connectAttr(remap.output, sin_node.input)
        cmds.connectAttr(remap.output, cos_node.input)
        cmds.connectAttr(sin_node.output, f"{driver}.translateZ")
        cmds.connectAttr(cos_node.output, f"{driver}.scaleX")
        cmds.connectAttr(cos_node.output, f"{driver}.scaleY")
        #cmds.connectAttr(cos_node.output, f"{driver}.scaleZ")



