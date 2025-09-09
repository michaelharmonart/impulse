"""
Module for generating ribbon rigs in Maya.

This module provides functions to:
    - Create UVPin nodes for pinning objects to surfaces.
    - Generate a ribbon rig from a NURBS surface with control joints and deformation joints.
    - Interpolate between two ribbon setups.

It leverages custom control curves from the control_gen module.
"""

from impulse.maya_api.node import MultiplyPointByMatrixNode
from impulse.maya_api import node
import maya.cmds as cmds

from impulse.utils import pin as pin
from impulse.utils.control import (
    Control,
    Direction,
    connect_control,
    make_control,
    make_surface_control,
)


def generate_ribbon(
    nurbs_surface_name: str,
    attach_surface: str = None,
    number_of_joints: int = None,
    number_of_interpolation_joints: int = None,
    cyclic: bool = False,
    swap_uv: bool = False,
    local_space: bool = False,
    control_joints: bool = True,
    number_of_controls: int = None,
    half_controls: bool = True,
    control_direction: Direction = None,
    control_normal_axis: str = None,
    control_tangent_axis: str = None,
    control_sensitivity: float = 1,
    hide_joints: bool = True,
    hide_surfaces: bool = False,
) -> None:
    """
    Generate a ribbon rig from a NURBS surface with joints and optional control curves.

    Args:
        nurbs_surface_name: Name of the input NURBS surface.
        attach_surface: If set, will define a surface that this ribbon will be locked to.
        number_of_joints: Number of deformation joints along the ribbon; if None, assume two per control joint
        cyclic: Whether the ribbon is cyclic (closed loop).
        swap_uv: If True, swap U and V parameters when evaluating surface positions.
        local_space: If False, disable transform inheritance on generated objects.
        control_joints: If True, create control joints with control curves.
        number_of_controls: Number of control joints; if None, assume one per isoparm (one per unit in UV space)
        half_controls: If True and number_of_controls is None, assume one per two isoparms.
        hide_joints: If True, hide generated deformation joints.
        hide_surfaces: If True, hide the original and duplicated surfaces.
        snap_surface: If set, defines a surface/mesh that the ribbon controls will snap to.
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
        number_of_joints = int(ribbon_length * 2)

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
        # Create temporary locator for UV pinning.
        cmds.select(ctl_group)
        temp_locator = cmds.joint(position=pos)

        cmds.select(ribbon_group, replace=True)
        joint_name = cmds.joint(
            position=pos, radius=1, name=f"{ribbon_object}_ControlJoint{idx + 1}_JNT"
        )
        if attach_surface:
            pin.make_uv_pin(
                object_to_pin=temp_locator,
                surface=ribbon_object,
                u=u_val,
                v=v_val,
                local_space=local_space,
                normal_axis=control_normal_axis,
                tangent_axis=control_tangent_axis,
            )
            ctl: Control = make_surface_control(
                name=f"{ribbon_object}_ControlJoint{idx + 1}",
                size=0.4,
                parent=ribbon_group,
                direction=control_direction,
                match_transform=temp_locator,
                surface=attach_surface,
                control_sensitivity=(control_sensitivity, control_sensitivity),
            )
            cmds.matchTransform(joint_name, temp_locator)
            cmds.parent(ctl.offset_transform, ctl_group)
            connect_control(control=ctl, driven_name=joint_name)
        else:
            ctl: Control = make_control(
                name=f"{ribbon_object}_ControlJoint{idx + 1}",
                position=pos,
                size=0.4,
                parent=ribbon_group,
                direction=control_direction,
            )
            cmds.parent(ctl.offset_transform, ctl_group)
            connect_control(control=ctl, driven_name=joint_name)
            pin.make_uv_pin(
                object_to_pin=temp_locator,
                surface=ribbon_object,
                u=u_val,
                v=v_val,
                local_space=local_space,
                normal_axis=control_normal_axis,
                tangent_axis=control_tangent_axis,
            )
            cmds.matchTransform(ctl.offset_transform, temp_locator)

        cmds.delete(temp_locator)
        return joint_name

    # Create control joints if requested.
    control_range = range(number_of_controls) if cyclic else range(number_of_controls + 1)
    if control_joints:
        for i in control_range:
            joint_name = create_control_joint(i, number_of_controls)
            # if hide_joints:
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
        joint_name = cmds.joint(
            position=pos, radius=0.5, name=f"{ribbon_object}_point{idx + 1}_DEF"
        )
        if hide_joints:
            cmds.hide(joint_name)

        if attach_surface:
            pin_point = cmds.group(
                empty=True, parent=ctl_group, name=f"{ribbon_object}_pin{idx + 1}"
            )
            pin.make_uv_pin(
                object_to_pin=pin_point,
                surface=ribbon_object,
                u=u_val,
                v=v_val,
                local_space=local_space,
                normal_axis=control_normal_axis,
                tangent_axis=control_tangent_axis,
            )

            shapes = cmds.listRelatives(attach_surface, shapes=True) or []
            if not shapes:
                cmds.error(f"No shape node found on object: {attach_surface}")
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
            world_position_node: MultiplyPointByMatrixNode = node.MultiplyPointByMatrixNode(name=f"{pin_point}_worldPosition")


            cmds.connectAttr(f"{pin_point}.parentMatrix", world_position_node.input_matrix)
            cmds.connectAttr(f"{pin_point}.translate", world_position_node.input_point)
            cp_node = cmds.createNode(cp_node_type, name=f"{pin_point}_closestPoint")
            cmds.connectAttr(f"{shape}{attr_world}", f"{cp_node}{cp_input}")
            cmds.connectAttr(world_position_node.output, f"{cp_node}.inPosition")

            ctl_name = make_surface_control(
                name=f"{ribbon_object}_point{idx + 1}",
                size=0.2,
                parent=ctl_group,
                direction=control_direction,
                match_transform=pin_point,
                surface=attach_surface,
                control_sensitivity=(control_sensitivity, control_sensitivity),
                u_attribute=f"{cp_node}.result.parameterU",
                v_attribute=f"{cp_node}.result.parameterV",
            )
            connect_control(ctl_name, joint_name)
            cmds.parent(pin_point, ctl_name)
        else:
            ctl: Control = make_control(
                name=f"{ribbon_object}_point{idx + 1}",
                position=pos,
                size=0.2,
                parent=ribbon_group,
                direction=control_direction,
            )
            pin.make_uv_pin(
                object_to_pin=ctl.offset_transform,
                surface=ribbon_object,
                u=u_val,
                v=v_val,
                local_space=local_space,
                normal_axis=control_normal_axis,
                tangent_axis=control_tangent_axis,
            )
            cmds.makeIdentity(ctl.offset_transform, apply=False)
            cmds.parent(ctl.offset_transform, ctl_group)
            connect_control(control=ctl, driven_name=joint_name)

    # Helper to create a interpolation joint.
    def create_interpolation_joint(idx: int, total_joints: int) -> None:
        divisor = total_joints if cyclic else (total_joints - 1)
        u_val = (idx / divisor) * ribbon_length
        v_val = ribbon_width / 2.0
        if swap_uv:
            u_val, v_val = v_val, u_val
        pos = cmds.pointOnSurface(ribbon_object, position=True, parameterU=u_val, parameterV=v_val)
        cmds.select(ribbon_group, replace=True)
        joint_name = cmds.joint(
            position=pos, radius=0.5, name=f"{ribbon_object}_point{idx + 1}_INT"
        )
        cmds.hide(joint_name)
        pin.make_uv_pin(
            object_to_pin=joint_name,
            surface=ribbon_object,
            u=u_val,
            v=v_val,
            local_space=local_space,
            normal_axis=control_normal_axis,
            tangent_axis=control_tangent_axis,
        )

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
    control_normal_axis: str = None,
    control_tangent_axis: str = None,
    number_of_joints: int = None,
    number_of_interpolation_joints: int = None,
    swap_uv=False,
    attach_surface: str = None,
    control_sensitivity: float = 1,
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
            control_normal_axis=control_normal_axis,
            control_tangent_axis=control_tangent_axis,
            attach_surface=attach_surface,
            control_sensitivity=control_sensitivity,
        )


def ribbon_interpolate(
    primary_ribbon_group: str,
    secondary_ribbon_group: str,
    interpolation_object: str,
    number_of_loops: int = 4,
    use_interpolation_joints=False,
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

    interp_group = cmds.group(
        empty=True, name=primary_ribbon_group.replace("GRP", "_Interpolate_GRP")
    )
    cmds.select(interp_group)
    total_joints = len(primary_joints)
    row_groups: list[str] = []
    for i in range(number_of_loops):
        blend_value = (1 / (number_of_loops + 1)) * (i + 1)
        row_group = cmds.group(
            name=primary_ribbon_group.replace("GRP", f"Row{i + 1}_GRP"),
            empty=True,
            parent=interp_group,
        )
        row_groups.append(row_group)
        cmds.addAttr(row_group, longName="rowBlend", attributeType="float")
        cmds.setAttr(f"{row_group}.rowBlend", blend_value)

    for joint_idx in range(total_joints):
        # Create nodes to get world-space positions for primary joint.
        
        primary_pos_node: MultiplyPointByMatrixNode = node.MultiplyPointByMatrixNode(
            name=primary_joints[joint_idx].replace(interpolation_joint_suffix, "Position")
        )
        cmds.connectAttr(
            f"{primary_joints[joint_idx]}.parentMatrix", primary_pos_node.input_matrix
        )
        cmds.connectAttr(f"{primary_joints[joint_idx]}.translate", primary_pos_node.input_point)

        primary_cp_node = cmds.createNode(
            cp_node_type,
            name=primary_joints[joint_idx].replace(interpolation_joint_suffix, "ClosestPoint"),
        )
        cmds.connectAttr(f"{shape}{attr_world}", f"{primary_cp_node}{cp_input}")
        cmds.connectAttr(primary_pos_node.output, f"{primary_cp_node}.inPosition")
        if surface_type == "mesh":
            cmds.connectAttr(f"{shape}.worldMatrix[0]", f"{primary_cp_node}.inputMatrix")

        # Create nodes for the secondary joint.

        secondary_pos_node: MultiplyPointByMatrixNode = node.MultiplyPointByMatrixNode(
            name=secondary_joints[joint_idx].replace(interpolation_joint_suffix, "Position")
        )

        cmds.connectAttr(
            f"{secondary_joints[joint_idx]}.parentMatrix", secondary_pos_node.input_matrix
        )
        cmds.connectAttr(
            f"{secondary_joints[joint_idx]}.translate", secondary_pos_node.input_point
        )

        secondary_cp_node = cmds.createNode(
            cp_node_type,
            name=secondary_joints[joint_idx].replace(interpolation_joint_suffix, "ClosestPoint"),
        )
        cmds.connectAttr(f"{shape}{attr_world}", f"{secondary_cp_node}{cp_input}")
        cmds.connectAttr(secondary_pos_node.output, f"{secondary_cp_node}.inPosition")
        if surface_type == "mesh":
            cmds.connectAttr(f"{shape}.worldMatrix[0]", f"{secondary_cp_node}.inputMatrix")

        # For each blending row, create a control joint with blended UV attributes.
        for row_idx, row_group in enumerate(row_groups):
            cmds.select(row_group)
            blend_joint = cmds.joint(
                radius=1,
                name=primary_joints[joint_idx].replace(
                    interpolation_joint_suffix, f"_Row{row_idx + 1}_CTL"
                ),
            )
            cmds.setAttr(f"{blend_joint}.inheritsTransform", 0)
            uv_pin_node = pin.make_uv_pin(
                object_to_pin=blend_joint, surface=interpolation_object, u=0.5, v=0.5
            )

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


def populate_surface(
    surface: str,
    joints_u: int = None,
    joints_v: int = None,
    cyclic_u: bool = False,
    cyclic_v: bool = False,
    control_sensitivity: float = 1,
) -> str:
    """
    Fills a given surface (NURBS or Mesh) with evenly spaced surface controls (according to UVs)

    Args:
        surface: Name of surface to generate joints for.
        joints_u: Number of joints along U, if not set, will assume one joint per isoparm.
        joints_v: Number of joints along V, if not set, will assume one joint per isoparm.
    """
    # Retrieve shape nodes from the surface.
    shapes = cmds.listRelatives(surface, children=True, shapes=True) or []
    if not shapes:
        cmds.error(f"No shape nodes found on surface: {surface}")

    # Choose the primary shape (non-intermediate if available) and check for an existing intermediate shape.
    primary_shape = next(
        (s for s in shapes if not cmds.getAttr(f"{s}.intermediateObject")), shapes[0]
    )

    # Determine attribute names based on surface type.
    surface_type = cmds.objectType(primary_shape)
    if surface_type == "mesh":
        attr_input = ".inMesh"
        attr_world = ".worldMesh[0]"
        attr_output = ".outMesh"
        surface_u = 1
        surface_v = 1
    elif surface_type == "nurbsSurface":
        attr_input = ".create"
        attr_world = ".worldSpace[0]"
        attr_output = ".local"
        surface_u = cmds.getAttr(f"{surface}.spansUV")[0][0]
        surface_v = cmds.getAttr(f"{surface}.spansUV")[0][1]
    else:
        cmds.error(f"Unsupported surface type: {surface_type}")

    group_name = cmds.group(name=f"{surface}_CTL_GRP", empty=True)

    if not joints_u:
        joints_u = surface_u
    if not joints_v:
        joints_v = surface_v

    range_u = range(joints_u) if cyclic_u else range(joints_u + 1)
    range_v = range(joints_v) if cyclic_v else range(joints_v + 1)
    divisor_u = joints_u if cyclic_u else (joints_u + 1)
    divisor_v = joints_v if cyclic_v else (joints_v + 1)

    for row in range_v:
        for column in range_u:
            ctl: Control = make_surface_control(
                name=f"{surface}_tweak_{row}_{column}",
                surface=surface,
                uv_position=((column / divisor_u) * surface_u, (row / divisor_v) * surface_v),
                control_sensitivity=(control_sensitivity, control_sensitivity),
                size=0.2,
                parent=group_name,
            )
            joint_name = cmds.joint(radius=1, name=f"{surface}_tweak_{row}_{column}_JNT")
            cmds.parent(joint_name, group_name)
            cmds.matchTransform(joint_name, ctl.control_transform)
            connect_control(control_name=ctl, driven_name=joint_name)
    return group_name
