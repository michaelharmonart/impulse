from typing import Literal, Sequence

import maya.api.OpenMaya as om2
import maya.cmds as cmds
from maya.api.OpenMaya import (
    MColor,
    MDagPath,
    MFnMesh,
    MIntArray,
    MMatrix,
    MMeshIntersector,
    MObject,
    MPoint,
    MPointArray,
    MPointOnMesh,
    MSelectionList,
    MSpace,
    MVector,
)

from impulse.utils.color.convert import lch_to_lab, linear_srgb_to_rec2020, oklab_to_linear_srgb
from impulse.utils.color.gradient import (
    OKLCH_HEATMAP_GRADIENT,
    Gradient,
    get_gradient_knots,
    sample_spline_gradient,
)
from impulse.utils.math import remap
from impulse.utils.spline.math import get_weights_along_spline, point_on_spline_weights
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


def get_closest_vertex_id(
    mesh_intersector: MMeshIntersector,
    mfn_mesh: MFnMesh,
    mesh_points: Sequence[MPoint],
    point: MPoint,
) -> int:
    closest_point: MPointOnMesh = mesh_intersector.getClosestPoint(point)
    face_vertices: MIntArray = mfn_mesh.getPolygonVertices(closest_point.face)

    return min(
        face_vertices,
        key=lambda vertex_id: mesh_points[vertex_id].distanceTo(point),
    )


def blend_colors_by_weight(color_weights: Sequence[tuple[MColor, float]]) -> MColor:
    final_color: MColor = MColor((0.0, 0.0, 0.0))
    for color, weight in color_weights:
        final_color += color * weight
    return final_color


def fast_sample_heatmap(
    positions: Sequence[float], gradient: Gradient = OKLCH_HEATMAP_GRADIENT
) -> list[MColor]:
    gradient_stop_colors = [MColor(stop.color) for stop in gradient.stops]
    gradient_knots = get_gradient_knots(gradient)
    position_color_weights = get_weights_along_spline(
        cvs=gradient_stop_colors, parameters=positions, knots=gradient_knots, degree=gradient.degree
    )
    colors: list[MColor] = []
    for color_weights in position_color_weights:
        color = blend_colors_by_weight(color_weights)
        point_color_lch = lch_to_lab(tuple(color.getColor()))
        point_color_rgb = oklab_to_linear_srgb(color=point_color_lch)
        point_color = MColor(point_color_rgb)
        colors.append(point_color)
    return colors


def color_from_symmetry_error(
    mesh: str,
    symmetry_axis: Literal["x", "y", "z"] | int = "x",
    max_error: float = 0.01,
):
    shape = get_shape(mesh)
    msel: MSelectionList = om2.MSelectionList()
    msel.add(shape)
    shape_obj: MObject = msel.getDependNode(0)
    mfn_mesh: MFnMesh = om2.MFnMesh(shape_obj)
    mesh_intersector: MMeshIntersector = om2.MMeshIntersector().create(shape_obj)
    mesh_points: MPointArray = mfn_mesh.getPoints(MSpace.kObject)
    mirror_matrix = get_mirror_matrix(symmetry_axis)
    vertex_symmetry_error: list[float] = []
    vertex_indices: list[int] = []

    viz_errors: list[float] = []
    for i, point in enumerate(mesh_points):
        mirrored_point: MPoint = point * mirror_matrix
        mirrored_vertex_id = get_closest_vertex_id(
            mesh_intersector, mfn_mesh, mesh_points, mirrored_point
        )
        mirrored_vertex_position: MPoint = mfn_mesh.getPoint(mirrored_vertex_id, MSpace.kObject)
        error_vector: MVector = mirrored_vertex_position - mirrored_point
        error: float = error_vector.length()
        remapped_error = remap(input=error, input_range=(0, max_error), output_range=(0, 1))
        clamped_error = min(remapped_error, 1.0)
        vertex_symmetry_error.append(error)
        viz_errors.append(clamped_error)
        vertex_indices.append(i)

    vertex_colors: list[MColor] = fast_sample_heatmap(viz_errors)

    # make sure the target shape can show vertex colors
    cmds.setAttr(f"{shape}.displayColors", 1)
    cmds.setAttr(f"{shape}.displayColorChannel", "Diffuse", type="string")

    mfn_mesh.setVertexColors(vertex_colors, vertex_indices)


def color_by_gradient(shape: str, gradient: Gradient = OKLCH_HEATMAP_GRADIENT):
    msel: MSelectionList = om2.MSelectionList()
    msel.add(shape)
    shape_dag: MDagPath = msel.getDagPath(0)
    mfn_mesh: MFnMesh = om2.MFnMesh(shape_dag)
    mesh_points: MPointArray = mfn_mesh.getPoints(MSpace.kWorld)

    vertex_indices: list[int] = []
    vertex_colors: list[MColor] = []
    for i, point in enumerate(mesh_points):
        heatmap_color_oklch = sample_spline_gradient(gradient=gradient, position=point.x)
        heatmap_color_oklab = lch_to_lab(heatmap_color_oklch)
        heatmap_color_linear_srgb = oklab_to_linear_srgb(heatmap_color_oklab)
        vertex_colors.append(MColor(heatmap_color_linear_srgb))
        vertex_indices.append(i)

    # make sure the target shape can show vertex colors
    cmds.setAttr(f"{shape}.displayColors", 1)
    cmds.setAttr(f"{shape}.displayColorChannel", "Diffuse", type="string")

    mfn_mesh.setVertexColors(vertex_colors, vertex_indices)
    pass
