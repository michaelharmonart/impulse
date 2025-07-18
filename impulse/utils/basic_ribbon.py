"""
Module: ribbon_generator
Description:
    This module provides a function to generate a ribbon rig in Maya based on a given NURBS surface.
    This script is depreciated, `ribbon.py` contains a more feature rich version of this script.

Usage:
    Select one or more NURBS surfaces in Maya and run this script. Each selected object will be processed
    to create a cyclic ribbon rig.
"""

import maya.cmds as cmds


def get_surface_shape(node: str) -> str:
    """
    Retrieve the first shape node of a given transform and ensure it exists.
    """
    shapes = cmds.listRelatives(node, shapes=True) or []
    if not shapes:
        raise RuntimeError(f"No shape node found for {node}.")
    return shapes[0]


def get_surface_dimensions(surface_shape: str, swap_uv: bool) -> tuple[float, float]:
    """
    Get the ribbon length and width from a NURBS surface.

    Args:
        surface_shape: The name of the surface shape.
        swap_uv: If True, swap U and V dimensions.

    Returns:
        A tuple (ribbon_length, ribbon_width).
    """
    if swap_uv:
        ribbon_length = cmds.getAttr(f"{surface_shape}.minMaxRangeV")[0][1]
        ribbon_width = cmds.getAttr(f"{surface_shape}.minMaxRangeU")[0][1]
    else:
        ribbon_length = cmds.getAttr(f"{surface_shape}.minMaxRangeU")[0][1]
        ribbon_width = cmds.getAttr(f"{surface_shape}.minMaxRangeV")[0][1]
    return ribbon_length, ribbon_width


def get_surface_point(surface: str, u: float, v: float, swap_uv: bool) -> list[float]:
    """
    Compute a point on a surface given U/V parameters.

    Args:
        surface: The surface (typically a duplicated NURBS surface) on which to compute the point.
        u: The U parameter.
        v: The V parameter.
        swap_uv: If True, swap the U and V parameters when querying the surface.

    Returns:
        The XYZ position on the surface.
    """
    if swap_uv:
        return cmds.pointOnSurface(surface, position=True, parameterU=v, parameterV=u)
    return cmds.pointOnSurface(surface, position=True, parameterU=u, parameterV=v)


def create_joint_at_point(
    parent: str, position: list[float], radius: float, joint_name: str
) -> str:
    """
    Create a joint at the specified position under a given parent.

    Args:
        parent: The transform to which the joint will belong.
        position: The XYZ position for the joint.
        radius: The joint's radius.
        joint_name: The name for the joint.

    Returns:
        The name of the created joint.
    """
    cmds.select(parent, replace=True)
    return cmds.joint(position=position, radius=radius, name=joint_name)


def pin_joint_to_uv(surface: str, joint: str, uv_string: str, local_space: bool) -> None:
    """
    Pin a joint to a surface using UV coordinates.

    Args:
        surface: The surface containing the UVs.
        joint: The joint to be pinned.
        uv_string: The UV coordinate string (e.g. "mySurface.uv[0.5][0.5]").
        local_space: If False, disable transform inheritance.
    """
    cmds.select(uv_string, replace=True)
    cmds.select(joint, add=True)
    cmds.UVPin()
    if not local_space:
        cmds.setAttr(f"{joint}.inheritsTransform", 0)


def generate_ribbon(
    nurbs_surface_name: str,
    number_of_joints: int = 8,
    cyclic: bool = False,
    swap_uv: bool = False,
    local_space: bool = False,
    control_joints: bool = True,
    number_of_controls: int | None = None,
    half_controls: bool = True,
    hide_surfaces: bool = False,
) -> None:
    """
    Generate a ribbon rig based on a given NURBS surface.

    The function duplicates the surface, creates a ribbon group, and adds control and deformation
    joints along the surface. UVPin is used to pin joints to specific UV coordinates.

    Args:
        nurbs_surface_name: The name of the NURBS surface.
        number_of_joints: The number of deformation joints to create.
        cyclic: If True, treat the ribbon as cyclic.
        swap_uv: If True, swap the U and V parameters.
        local_space: If False, disable transform inheritance on generated nodes.
        control_joints: If True, create control joints along the ribbon.
        number_of_controls: The number of control joints; if None, computed from ribbon length.
        half_controls: If number_of_controls is None, use half the ribbon length if True.
        hide_surfaces: If True, hide both the original and duplicated surfaces.
    """
    surface_shape = get_surface_shape(nurbs_surface_name)
    if cmds.nodeType(surface_shape) != "nurbsSurface":
        raise RuntimeError(f"{surface_shape} is not a nurbsSurface.")

    ribbon_length, ribbon_width = get_surface_dimensions(surface_shape, swap_uv)

    if number_of_controls is None:
        number_of_controls = int(ribbon_length)
        if half_controls:
            number_of_controls //= 2

    ribbon_object = cmds.duplicate(nurbs_surface_name, name=f"{nurbs_surface_name}_ribbon")[0]
    if hide_surfaces:
        cmds.hide(nurbs_surface_name)
        cmds.hide(ribbon_object)
    if not local_space:
        cmds.setAttr(f"{ribbon_object}.inheritsTransform", 0)
    ribbon_group = cmds.group(ribbon_object, name=f"{ribbon_object}_GRP")

    # Create control joints if requested.
    if control_joints:
        control_loop_count = number_of_controls if cyclic else number_of_controls + 1
        for i in range(control_loop_count):
            control_spacing = ribbon_length / number_of_controls
            u = control_spacing * i
            v = ribbon_width / 2.0
            position = get_surface_point(ribbon_object, u, v, swap_uv)
            joint_name = f"{ribbon_object}_{i + 1}_ControlJoint"
            create_joint_at_point(ribbon_group, position, radius=1, joint_name=joint_name)

    # Create deformation joints along the ribbon.
    for i in range(number_of_joints):
        cmds.select(ribbon_group, replace=True)
        u = (
            (i / number_of_joints) * ribbon_length
            if cyclic
            else (i / (number_of_joints - 1)) * ribbon_length
        )
        v = ribbon_width / 2.0
        position = get_surface_point(ribbon_object, u, v, swap_uv)
        joint_name = f"{ribbon_object}_point{i + 1}_DEF"
        created_joint = create_joint_at_point(
            ribbon_group, position, radius=0.5, joint_name=joint_name
        )

        if cyclic:
            uv_value = (i / (number_of_joints - 2)) * 2 if number_of_joints > 2 else 0.5
            uv_coord = f"{ribbon_object}.uv[{uv_value}][0.5]"
        else:
            uv_coord = (
                f"{ribbon_object}.uv[{v}][{u}]" if swap_uv else f"{ribbon_object}.uv[{u}][{v}]"
            )

        pin_joint_to_uv(ribbon_object, created_joint, uv_coord, local_space)


for obj in cmds.ls(selection=True):
    generate_ribbon(obj, cyclic=True)
