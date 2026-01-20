from typing import Literal, Sequence

import maya.api.OpenMaya as om2
import maya.cmds as cmds
import numpy as np
from maya.api.OpenMaya import (
    MColor,
    MColorArray,
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
from numpy.typing import NDArray

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


def oklch_to_linear_srgb(color: MColor) -> MColor:
    return MColor(linear_srgb_to_rec2020(oklab_to_linear_srgb(lch_to_lab(color.getColor()))))


def numpy_array_to_colors(array: NDArray[float]) -> MColorArray:
    color_array = MColorArray()
    color_array.setLength(len(array))
    for i, color_row in enumerate(array):
        color_array[i] = MColor(list(color_row))
    return color_array


def fast_sample_lch_gradient_as_linear_srgb(
    positions: Sequence[float],
    gradient: Gradient = OKLCH_HEATMAP_GRADIENT,
    sample_points: int = 128,
) -> MColorArray:
    result: MColorArray = MColorArray()
    result.setLength(len(positions))
    gradient_stop_colors = [MColor(stop.color) for stop in gradient.stops]
    gradient_stop_ids = list(range(len(gradient.stops)))
    gradient_knots = get_gradient_knots(gradient)

    result: list[MColor] = []
    if len(positions) <= sample_points:
        for index, position in enumerate(positions):
            weights = point_on_spline_weights(
                cvs=gradient_stop_colors,
                t=position,
                degree=gradient.degree,
                knots=gradient_knots,
                normalize=False,
            )
            result[index] = oklch_to_linear_srgb(blend_colors_by_weight(weights))
        return result

    # Precompute lookup table
    parameter_array = np.array(positions, dtype=float)
    min_t, max_t = min(positions), max(positions)
    t_range: float = max_t - min_t
    if t_range == 0:
        # All parameters are the same, just calculate the one weight
        weights = point_on_spline_weights(
            cvs=gradient_stop_colors,
            t=min_t,
            degree=gradient.degree,
            knots=gradient_knots,
            normalize=False,
        )
        for index, _ in enumerate(positions):
            result[index] = oklch_to_linear_srgb(blend_colors_by_weight(weights))
        return result

    # Get evenly spaced points from the minimum to maximum t value
    sample_params = np.linspace(min_t, max_t, sample_points, dtype=float)
    lut_colors = np.zeros((sample_points, 4), dtype=float)

    for sample_index, sample_parameter in enumerate(sample_params):
        weights = point_on_spline_weights(
            cvs=gradient_stop_colors,
            t=sample_parameter,
            degree=gradient.degree,
            knots=gradient_knots,
            normalize=False,
        )
        color = oklch_to_linear_srgb(blend_colors_by_weight(weights))
        # Take the weights and put them into the correct row in the array
        lut_colors[sample_index, :] = color.getColor()

    # Map each parameter to LUT index positions
    normalized_positions = (parameter_array - min_t) / t_range * (sample_points - 1)
    lower_indices = np.floor(normalized_positions).astype(int)
    upper_indices = np.clip(lower_indices + 1, 0, sample_points - 1)
    interpolation_alphas = (normalized_positions - lower_indices)[:, None]

    # Interpolate weights for all parameters in bulk
    interpolated_color_array = (1 - interpolation_alphas) * lut_colors[
        lower_indices, :
    ] + interpolation_alphas * lut_colors[upper_indices, :]

    # Reattach CV references to each interpolated weight row
    return numpy_array_to_colors(interpolated_color_array)


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
        mirrored_position: MPoint = MPoint(mesh_intersector.getClosestPoint(mirrored_point).point)
        error_vector: MVector = mirrored_position - mirrored_point
        error: float = error_vector.length()
        remapped_error = remap(input=error, input_range=(0, max_error), output_range=(0, 1))
        vertex_symmetry_error.append(error)
        viz_errors.append(remapped_error if remapped_error < 1 else 1)
        vertex_indices.append(i)

    vertex_colors: MColorArray = fast_sample_lch_gradient_as_linear_srgb(
        positions=viz_errors, gradient=OKLCH_HEATMAP_GRADIENT
    )

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
