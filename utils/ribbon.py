"""
Module for generating ribbon rigs in Maya.

This module provides functions to:
    - Create UVPin nodes for pinning objects to surfaces.
    - Generate a ribbon rig from a NURBS surface with control joints and deformation joints.
    - Interpolate between two ribbon setups.
    
It leverages custom control curves from the control_gen module.
"""

import maya.cmds as cmds
from . import control as control


def make_uv_pin (object_to_pin: str, surface: str, u: float, v: float,) -> str:
    """
    Create a UVPin node that pins an object to a given surface at specified UV coordinates.

    Args:
        object_to_pin: The name of the object to be pinned.
        surface: The name of the surface (mesh or NURBS) to pin to.
        u: The U coordinate.
        v: The V coordinate.

    Returns:
        The name of the created UVPin node.
    """
    # Retrieve shape nodes from the surface.
    shapes = cmds.listRelatives(surface, children=True, shapes=True) or []
    if not shapes:
        cmds.error(f"No shape nodes found on surface: {surface}")
    
    # Choose the primary shape (non-intermediate if available) and check for an existing intermediate shape.
    primary_shape = next((s for s in shapes if not cmds.getAttr(f"{s}.intermediateObject")), shapes[0])
    shape_origin = next((s for s in shapes if cmds.getAttr(f"{s}.intermediateObject")), None) 

    # Determine attribute names based on surface type.
    surface_type = cmds.objectType(primary_shape)
    if surface_type == "mesh":
        attr_input = ".inMesh"
        attr_world = ".worldMesh[0]"
        attr_output = ".outMesh"
    elif surface_type == "nurbsSurface":
        attr_input = ".create"
        attr_world = ".worldSpace[0]"
        attr_output = ".local"
    else:
        cmds.error(f"Unsupported surface type: {surface_type}")

    # If no intermediate shape exists, create one.
    if shape_origin is None:
        duplicated = cmds.duplicate(primary_shape)[0]
        shape_origin_list = cmds.listRelatives(duplicated, children=True, shapes=True)
        if not shape_origin_list:
            cmds.error("Could not create intermediate shape.")
        shape_origin = shape_origin_list[0]
        cmds.parent(shape_origin, surface, shape=True, relative=True)
        cmds.delete(duplicated)
        new_name = f"{primary_shape}Orig"
        shape_origin = cmds.rename(shape_origin, new_name)
        # If there is an incoming connection, reconnect it.
        in_conn = cmds.listConnections(f"{primary_shape}{attr_input}", plugs=True, connections=True, destination=True)
        if in_conn:
            cmds.connectAttr(in_conn[1], f"{shape_origin}{attr_input}")
        cmds.connectAttr(f"{shape_origin}{attr_world}", f"{primary_shape}{attr_input}", force=True)
        cmds.setAttr(f"{shape_origin}.intermediateObject", 1)
    
    # Create the UVPin node and connect it.
    uv_pin = cmds.createNode("uvPin", name=f"{object_to_pin}_uvPin")
    cmds.connectAttr(f"{primary_shape}{attr_world}", f"{uv_pin}.deformedGeometry")
    cmds.connectAttr(f"{shape_origin}{attr_output}", f"{uv_pin}.originalGeometry")
    cmds.xform(object_to_pin, translation=[0, 0, 0], rotation=[0, 0, 0])
    cmds.setAttr(f"{uv_pin}.normalAxis", 1)
    cmds.setAttr(f"{uv_pin}.tangentAxis", 0)
    cmds.setAttr(f"{uv_pin}.normalizedIsoParms", 0)
    cmds.setAttr(f"{uv_pin}.coordinate[0]", u, v, type="float2")
    cmds.connectAttr(f"{uv_pin}.outputMatrix[0]", f"{object_to_pin}.offsetParentMatrix")
    return uv_pin


def generate_ribbon (
        nurbs_surface_name: str, 
        number_of_joints: int = None,
        number_of_interpolation_joints: int = None,
        cyclic: bool = False,
        swap_uv: bool = False,
        local_space: bool = False, 
        control_joints: bool = True, 
        number_of_controls: int = None,
        half_controls: bool = True,
        hide_joints: bool = True,
        hide_surfaces: bool = False,
) -> None:
    """
    Generate a ribbon rig from a NURBS surface with joints and optional control curves.

    Args:
        nurbs_surface_name: Name of the input NURBS surface.
        number_of_joints: Number of deformation joints along the ribbon; if None, assume two per control joint
        cyclic: Whether the ribbon is cyclic (closed loop).
        swap_uv: If True, swap U and V parameters when evaluating surface positions.
        local_space: If False, disable transform inheritance on generated objects.
        control_joints: If True, create control joints with control curves.
        number_of_controls: Number of control joints; if None, assume one per isoparm (one per unit in UV space)
        half_controls: If True and number_of_controls is None, assume one per two isoparms.
        hide_joints: If True, hide generated deformation joints.
        hide_surfaces: If True, hide the original and duplicated surfaces.
    """
    # Retrieve the surface shape and ensure it is a NURBS surface.
    surface_shapes = cmds.listRelatives(nurbs_surface_name, shapes=True) or []
    if not surface_shapes:
        cmds.error(f"No shape node found for {nurbs_surface_name}")
    surface_shape = surface_shapes[0]
    if cmds.nodeType(surface_shape) != "nurbsSurface":
        cmds.error(f"Node {surface_shape} is not a nurbsSurface.")
                   
    # Get ribbon dimensions from surface attributes.
    if swap_uv:
        ribbon_length = cmds.getAttr(f"{surface_shape}.spansUV")[0][1]
        ribbon_width = cmds.getAttr(f"{surface_shape}.spansUV")[0][0]
    else:
        ribbon_length = cmds.getAttr(f"{surface_shape}.spansUV")[0][0]
        ribbon_width = cmds.getAttr(f"{surface_shape}.spansUV")[0][1]

    # Determine the number of controls if not provided.
    if number_of_controls is None:
        number_of_controls = int(ribbon_length)
        if half_controls:
            number_of_controls //= 2
    
    # Determine the number of joints if not provided.
    if number_of_joints is None:
        number_of_joints = int(ribbon_length*2)

    # Duplicate the surface to serve as the ribbon and organize it.
    ribbon_object = cmds.duplicate(nurbs_surface_name, name=f"{nurbs_surface_name}_ribbon")[0]
    ribbon_group = cmds.group(ribbon_object, name=f"{ribbon_object}_GRP")
    ctl_group = cmds.group(empty=True, parent=ribbon_group, name=f"{ribbon_object}_CTL")
    
    if hide_surfaces:
        cmds.hide(nurbs_surface_name)
        cmds.hide(ribbon_object)
    if not local_space:
        cmds.setAttr(f"{ribbon_object}.inheritsTransform", 0)

    # Helper to create a control joint given a position.
    def create_control_joint(idx: int, total_controls: int) -> str:
        control_spacing = ribbon_length / total_controls
        u_val = control_spacing * idx
        v_val = ribbon_width / 2.0
        if swap_uv:
            u_val, v_val = v_val, u_val
        pos = cmds.pointOnSurface(ribbon_object, position=True, parameterU=u_val, parameterV=v_val)
        cmds.select(ribbon_group, replace=True)
        joint_name = cmds.joint(position=pos, radius=1, name=f"{ribbon_object}_ControlJoint{idx+1}_JNT")
        ctl_name = control.generate_control(name=f"{ribbon_object}_ControlJoint{idx+1}", position=pos, size=0.4, parent=ribbon_group)
        cmds.parent(ctl_name, ctl_group)
        # Create temporary locator for UV pinning.
        cmds.select(ctl_group)
        temp_locator = cmds.joint(position=pos)
        control.connect(ctl_name, joint_name)
        make_uv_pin(object_to_pin=temp_locator, surface=ribbon_object, u=u_val, v=v_val)
        if not local_space:
            cmds.setAttr(f"{temp_locator}.inheritsTransform", 0)
        cmds.matchTransform(ctl_name, temp_locator)
        cmds.delete(temp_locator)
        return joint_name
    
    # Create control joints if requested.
    control_range = range(number_of_controls) if cyclic else range(number_of_controls + 1)
    if control_joints:
        for i in control_range:
            joint_name = create_control_joint(i, number_of_controls)
            #if hide_joints:
            #    cmds.hide(joint_name)
    
    # Helper to create a deformation joint.
    def create_deformation_joint(idx: int, total_joints: int) -> None:
        divisor = total_joints if cyclic else (total_joints - 1)
        u_val = (idx / divisor) * ribbon_length
        v_val = ribbon_width / 2.0
        if swap_uv:
            u_val, v_val = v_val, u_val
        pos = cmds.pointOnSurface(ribbon_object, position=True, parameterU=u_val, parameterV=v_val)
        cmds.select(ribbon_group, replace=True)
        joint_name = cmds.joint(position=pos, radius=0.5, name=f"{ribbon_object}_point{idx+1}_DEF")
        if hide_joints:
            cmds.hide(joint_name)
        ctl_name = control.generate_control(name = f"{ribbon_object}_point{idx+1}", position=pos, size=0.2, parent=ribbon_group)
        make_uv_pin(object_to_pin=ctl_name, surface=ribbon_object, u=u_val, v=v_val)
        cmds.makeIdentity(ctl_name, apply=False)
        cmds.parent(ctl_name, ctl_group)
        control.connect(ctl_name, joint_name)
        if not local_space:
            cmds.setAttr(f"{ctl_name}.inheritsTransform", 0)
   
    # Helper to create a interpolation joint.
    def create_interpolation_joint(idx: int, total_joints: int) -> None:
        divisor = total_joints if cyclic else (total_joints - 1)
        u_val = (idx / divisor) * ribbon_length
        v_val = ribbon_width / 2.0
        if swap_uv:
            u_val, v_val = v_val, u_val
        pos = cmds.pointOnSurface(ribbon_object, position=True, parameterU=u_val, parameterV=v_val)
        cmds.select(ribbon_group, replace=True)
        joint_name = cmds.joint(position=pos, radius=0.5, name=f"{ribbon_object}_point{idx+1}_INT")
        cmds.hide(joint_name)
        make_uv_pin(object_to_pin=joint_name, surface=ribbon_object, u=u_val, v=v_val)
        if not local_space:
            cmds.setAttr(f"{joint_name}.inheritsTransform", 0)

    # Create deformation joints.
    for i in range(number_of_joints):
        create_deformation_joint(i, number_of_joints)

    # Create interpolation joints.
    if number_of_interpolation_joints:
        for i in range(number_of_interpolation_joints):
            create_interpolation_joint(i, number_of_interpolation_joints)


def ribbon_from_selected(
        cyclic: bool = True,
        half_controls: bool = False,
        number_of_joints: int = None,
        number_of_interpolation_joints: int = None,
        swap_uv=False,
) -> None:
    """
    Generate a ribbon rig from each currently selected object.
    """
    selected_objects = cmds.ls(selection=True) or []
    for obj in selected_objects:
        generate_ribbon(
                obj, 
                cyclic=cyclic, 
                half_controls=half_controls, 
                number_of_joints=number_of_joints, 
                number_of_interpolation_joints=number_of_interpolation_joints,
                swap_uv=swap_uv,
        )


def ribbon_interpolate (
        primary_ribbon_group: str,
        secondary_ribbon_group: str,
        interpolation_object: str,
        number_of_loops: int = 4,
        use_interpolation_joints = False,
) -> None:
    """
    Set up interpolation between two ribbon groups along a mesh.

    For each corresponding joint (ending with 'DEF') in the two ribbon groups,
    this function creates blending nodes that interpolate between their UV coordinates
    on the provided mesh.

    Args:
        primary_ribbon_group: The first ribbon group's name.
        secondary_ribbon_group: The second ribbon group's name.
        interpolation_object: The object to which the interpolation is based.
        number_of_loops: Number of blending rows to create.
    """
    shapes = cmds.listRelatives(interpolation_object, shapes=True) or []
    if not shapes:
        cmds.error(f"No shape node found on object: {interpolation_object}")
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
        cmds.error(f"Unsupported surface type: {surface_type}")
    
    def get_def_joints(ribbon_group: str) -> list[str]:
        children = cmds.listRelatives(ribbon_group, children=True, type="transform") or []
        if use_interpolation_joints:
            return [child for child in children if child.endswith("INT")]
        else:
            return [child for child in children if child.endswith("DEF")]

    primary_joints = get_def_joints(primary_ribbon_group)
    secondary_joints = get_def_joints(secondary_ribbon_group)
    interpolation_joint_suffix: str = "INT" if use_interpolation_joints else "DEF"

    if len(primary_joints) != len(secondary_joints):
        cmds.error("Both ribbons must have the same number of joints.")
    
    interp_group = cmds.group(empty=True, name=primary_ribbon_group.replace("GRP", "_Interpolate_GRP"))
    cmds.select(interp_group)
    total_joints = len(primary_joints)
    row_groups: list[str] = []
    for i in range(number_of_loops):
        blend_value = (1 / (number_of_loops + 1)) * (i + 1)
        row_group = cmds.group(name=primary_ribbon_group.replace("GRP", f"Row{i+1}_GRP"), empty=True, parent=interp_group)
        row_groups.append(row_group)
        cmds.addAttr(row_group, longName="rowBlend", attributeType="float")
        cmds.setAttr(f"{row_group}.rowBlend", blend_value)

    for joint_idx in range(total_joints):
        # Create nodes to get world-space positions for primary joint.
        primary_pos_node = cmds.createNode("pointMatrixMult", name=primary_joints[joint_idx].replace(interpolation_joint_suffix, "Position"))
        cmds.connectAttr(f"{primary_joints[joint_idx]}.parentMatrix", f"{primary_pos_node}.inMatrix")
        cmds.connectAttr(f"{primary_joints[joint_idx]}.translate", f"{primary_pos_node}.inPoint")
    
        primary_cp_node = cmds.createNode(cp_node_type, name=primary_joints[joint_idx].replace(interpolation_joint_suffix, "ClosestPoint"))
        cmds.connectAttr(f"{shape}{attr_world}", f"{primary_cp_node}{cp_input}")
        cmds.connectAttr(f"{primary_pos_node}.output", f"{primary_cp_node}.inPosition")
        if surface_type == "mesh":
            cmds.connectAttr(f"{shape}.worldMatrix[0]", f"{primary_cp_node}.inputMatrix")

        # Create nodes for the secondary joint.
        secondary_pos_node = cmds.createNode("pointMatrixMult", name=secondary_joints[joint_idx].replace(interpolation_joint_suffix, "Position"))
        cmds.connectAttr(f"{secondary_joints[joint_idx]}.parentMatrix", f"{secondary_pos_node}.inMatrix")
        cmds.connectAttr(f"{secondary_joints[joint_idx]}.translate", f"{secondary_pos_node}.inPoint")

        secondary_cp_node = cmds.createNode(cp_node_type, name=secondary_joints[joint_idx].replace(interpolation_joint_suffix, "ClosestPoint"))
        cmds.connectAttr(f"{shape}{attr_world}", f"{secondary_cp_node}{cp_input}")
        cmds.connectAttr(f"{secondary_pos_node}.output", f"{secondary_cp_node}.inPosition")
        if surface_type == "mesh":
            cmds.connectAttr(f"{shape}.worldMatrix[0]", f"{secondary_cp_node}.inputMatrix")

        # For each blending row, create a control joint with blended UV attributes.
        for row_idx, row_group in enumerate(row_groups):
            cmds.select(row_group)
            blend_joint = cmds.joint(radius=1, name=primary_joints[joint_idx].replace(interpolation_joint_suffix, f"_Row{row_idx+1}_CTL"))
            cmds.setAttr(f"{blend_joint}.inheritsTransform", 0)
            uv_pin_node = make_uv_pin(object_to_pin=blend_joint, surface=interpolation_object, u=0.5, v=0.5)

            blend_u_node = cmds.createNode("blendTwoAttr", name=f"{uv_pin_node}_Blend_U")
            cmds.connectAttr(f"{primary_cp_node}.result.parameterU", f"{blend_u_node}.input[0]")
            cmds.connectAttr(f"{secondary_cp_node}.result.parameterU", f"{blend_u_node}.input[1]")
            cmds.connectAttr(f"{row_group}.rowBlend", f"{blend_u_node}.attributesBlender")

            blend_v_node = cmds.createNode("blendTwoAttr", name=f"{uv_pin_node}_Blend_V")
            cmds.connectAttr(f"{primary_cp_node}.result.parameterV", f"{blend_v_node}.input[0]")
            cmds.connectAttr(f"{secondary_cp_node}.result.parameterV", f"{blend_v_node}.input[1]")
            cmds.connectAttr(f"{row_group}.rowBlend", f"{blend_v_node}.attributesBlender")

            cmds.connectAttr(f"{blend_u_node}.output", f"{uv_pin_node}.coordinate[0].coordinateU")
            cmds.connectAttr(f"{blend_v_node}.output", f"{uv_pin_node}.coordinate[0].coordinateV")