from typing import Literal

import maya.api.OpenMaya as om2
import maya.cmds as cmds
from maya.api.OpenMaya import (
    MColor,
    MColorArray,
    MDagPath,
    MFnMesh,
    MIntArray,
    MMatrix,
    MObject,
    MPoint,
    MPointArray,
    MSelectionList,
    MSpace,
    MVector,
)

from impulse.utils import color
from impulse.utils.math import remap
from impulse.utils.transform import get_shapes

X_MIRROR = MMatrix(((-1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
Y_MIRROR = MMatrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
Z_MIRROR = MMatrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))


def get_shape(node: str) -> str:
    if cmds.nodeType(node) == "mesh":
        shape = node
    else:
        shape = get_shapes(node)[0]
    return shape


def get_mirror_matrix(symmetry_axis: Literal["x", "y", "z"] | int = "x") -> MMatrix:
    match symmetry_axis:
        case "x":
            return X_MIRROR
        case "y":
            return Y_MIRROR
        case "z":
            return Z_MIRROR
        case _:
            raise ValueError(f"{symmetry_axis} is not a valid symmetry axis (x, y, z)")


def get_closest_vertex_id(mfn_mesh: MFnMesh, point: MPoint, space: MSpace) -> int:
    _, face_id = mfn_mesh.getClosestPoint(point, space)
    face_vertices: MIntArray = mfn_mesh.getPolygonVertices(face_id)

    return min(
        face_vertices, key=lambda vertex_id: mfn_mesh.getPoint(vertex_id, space).distanceTo(point)
    )


def color_from_symmetry_error(
    mesh: str,
    symmetry_axis: Literal["x", "y", "z"] | int = "x",
    space: MSpace = MSpace.kWorld,
    max_error: float = 0.01,
):
    shape = get_shape(mesh)
    msel: MSelectionList = om2.MSelectionList()
    msel.add(shape)
    shape_dag: MDagPath = msel.getDagPath(0)
    mfn_mesh: MFnMesh = om2.MFnMesh(shape_dag)
    mesh_points: MPointArray = mfn_mesh.getPoints(space)
    mirror_matrix = get_mirror_matrix(symmetry_axis)
    vertex_symmetry_error: list[float] = []
    vertex_indices: list[int] = []
    vertex_colors: list[MColor] = []
    for i, point in enumerate(mesh_points):
        mirrored_point: MPoint = point * mirror_matrix
        mirrored_vertex_id = get_closest_vertex_id(mfn_mesh, mirrored_point, space)
        mirrored_vertex_position: MPoint = mfn_mesh.getPoint(mirrored_vertex_id, space)
        error_vector: MVector = mirrored_vertex_position - mirrored_point
        error: float = error_vector.length()
        remapped_error = remap(input=error, input_range=(0, max_error), output_range=(0, 1))
        clamped_error = min(remapped_error, 1.0)
        vertex_symmetry_error.append(vertex_symmetry_error)
        heatmap_color_oklch = color.sample_spline_gradient(
            gradient=color.OKLCH_HEATMAP_GRADIENT, position=clamped_error
        )
        heatmap_color_oklab = color.lch_to_lab(heatmap_color_oklch)
        heatmap_color_linear_srgb = color.oklab_to_linear_srgb(heatmap_color_oklab)
        vertex_colors.append(MColor(heatmap_color_linear_srgb))
        vertex_indices.append(i)

    # make sure the target shape can show vertex colors
    cmds.setAttr(f"{shape}.displayColors", 1)
    cmds.setAttr(f"{shape}.displayColorChannel", "Diffuse", type="string")

    mfn_mesh.setVertexColors(vertex_colors, vertex_indices)


def color_by_gradient(shape: str, gradient: color.Gradient = color.OKLCH_HEATMAP_GRADIENT):
    msel: MSelectionList = om2.MSelectionList()
    msel.add(shape)
    shape_dag: MDagPath = msel.getDagPath(0)
    mfn_mesh: MFnMesh = om2.MFnMesh(shape_dag)
    mesh_points: MPointArray = mfn_mesh.getPoints(MSpace.kWorld)

    vertex_indices: list[int] = []
    vertex_colors: list[MColor] = []
    for i, point in enumerate(mesh_points):
        heatmap_color_oklch = color.sample_spline_gradient(gradient=gradient, position=point.x)
        heatmap_color_oklab = color.lch_to_lab(heatmap_color_oklch)
        heatmap_color_linear_srgb = color.oklab_to_linear_srgb(heatmap_color_oklab)
        vertex_colors.append(MColor(heatmap_color_linear_srgb))
        vertex_indices.append(i)

    # make sure the target shape can show vertex colors
    cmds.setAttr(f"{shape}.displayColors", 1)
    cmds.setAttr(f"{shape}.displayColorChannel", "Diffuse", type="string")

    mfn_mesh.setVertexColors(vertex_colors, vertex_indices)
    pass
