import colorsys
from impulse.utils.color import linear_srgb_to_oklab, oklab_to_linear_srgb
from typing import Any
import maya.cmds as cmds
from maya.api import OpenMaya as om2
import os
from impulse.structs.transform import Vector3 as Vector3
from impulse.utils import spline as spline
from ngSkinTools2 import api as ng
from ngSkinTools2.api import plugin
from ngSkinTools2.api.influenceMapping import InfluenceMappingConfig
from ngSkinTools2.api.transfer import VertexTransferMode


def ensure_ng_initialized() -> None:
    if not plugin.is_plugin_loaded():
        plugin.load_plugin()


def init_layers(shape: str) -> ng.Layers:
    skin_cluster = ng.target_info.get_related_skin_cluster(shape)
    layers = ng.layers.init_layers(skin_cluster)
    base_layer: ng.Layer = layers.add("Base Weights")
    base_layer.set_weights
    return layers


def apply_ng_skin_weights(weights_file: str, geometry: str) -> None:
    """
    Applies an ngSkinTools JSON weights file to the specified geometry.
    Args:
        weights_file: The JSON weights file to read.
        geometry: The transform, shape, or skinCluster Node to apply to.
    """
    # shapes: list[str] = cmds.listRelatives(geometry, children=True, shapes=True) or []
    # if not shapes:
    #    raise RuntimeError(f"No shape nodes found on surface: {geometry}")
    # shape: str = shapes[0]
    ensure_ng_initialized()
    config: InfluenceMappingConfig = InfluenceMappingConfig()
    config.use_distance_matching = False
    config.use_name_matching = True

    if not os.path.isfile(path=weights_file):
        raise RuntimeError(f"{weights_file} doesn't exist, unable to load weights.")

    # Run the import
    ng.import_json(
        target=geometry,
        file=weights_file,
        vertex_transfer_mode=VertexTransferMode.vertexId,
        influences_mapping_config=config,
    )


def write_ng_skin_weights(filepath: str, geometry: str, force: bool = False) -> None:
    """
    Writes a ngSkinTools JSON file representing the weights of the given geometry.

    Args:
        filepath: The path and filename and extension to save under.
        geometry: The transform, shape, or skinCluster Node the weights are on.
        force: If True, will automatically overwrite any existing file at the filepath specified.

    """

    # If the file exists, only write it if force = True, or after asking for confirmation.
    if os.path.isfile(path=filepath):
        if force:
            pass
        else:
            confirm: str = cmds.confirmDialog(
                title="File Overwrite",
                message=f"{filepath} already exists and will be overwritten, are you sure you want to write the file?",
                button=["Yes", "No"],
                defaultButton="Yes",
                cancelButton="No",
                dismissString="No",
            )
            if confirm == "Yes":
                pass
            else:
                return

    ng.export_json(target=geometry, file=filepath)

    return


def skin_mesh(
    bind_joints: list[str], geometry: str, name: str | None = None, dual_quaternion: bool = False
) -> str:
    """
    Creates a skinCluster on the given geometry using the specified bind joints.

    Args:
        bind_joints (list[str]): A list of joint names to bind the geometry to.
        geometry (str): The name of the geometry to be skinned.
        name (str | None, optional): The name to assign to the skinCluster.
            If None, a name will be auto-generated based on the geometry name.
        dual_quaternion (bool, optional): Whether to use dual quaternion skinning.
            Defaults to False (classic linear skinning).

    Returns:
        str: The name of the created skinCluster node.
    """
    if not name:
        name: str = f"{geometry}_SC"
    skin_cluster = cmds.skinCluster(
        bind_joints, geometry, skinMethod=1 if dual_quaternion else 0, name=name
    )
    return skin_cluster


def get_mesh_spline_weights(
    mesh_shape: str, cv_transforms: list[str], degree: int = 2
) -> list[list[tuple[Any, float]]]:
    """
    Calculates spline-based weights for each vertex on a mesh relative to a temporary NURBS curve
    defined by a set of CV transforms.

    The function builds a curve from the given transforms, projects each mesh vertex onto the curve
    to compute the closest parameter value, then calculates De Boor-style basis weights using the
    curve's knot vector and degree.

    Args:
        mesh_shape (str): The name of the mesh shape node (not the transform).
        cv_transforms (list[str]): A list of transform names representing the CVs of the curve.
        degree (int, optional): Degree of the spline curve. Defaults to 2.

    Returns:
        list[list[tuple[Any, float]]]: A list of weights per vertex. Each entry is a list of tuples,
        where each tuple contains a CV transform and its corresponding influence weight on the vertex.
    """
    # Create a curve for checking the closest point
    cv_positions: list[list[float, float, float]] = []
    for transform in cv_transforms:
        position: list[float, float, float] = cmds.xform(
            transform, query=True, worldSpace=True, translation=True
        )
        position_tuple: tuple[float, float, float] = tuple(position)
        cv_positions.append(position)
    curve_transform: str = cmds.curve(
        name=f"{mesh_shape}_SplineWeightsTempCurve", point=cv_positions, degree=degree
    )

    # Get curve shape
    curve_shape: str = cmds.listRelatives(curve_transform, shapes=True)[0]

    # get the MDagPaths
    msel: om2.MSelectionList = om2.MSelectionList()
    msel.add(mesh_shape)
    msel.add(curve_shape)
    mesh_dag: om2.MDagPath = msel.getDagPath(0)
    curve_dag: om2.MDagPath = msel.getDagPath(1)

    # make the function set and get the points
    fn_mesh: om2.MFnMesh = om2.MFnMesh(mesh_dag)
    fn_curve: om2.MFnNurbsCurve = om2.MFnNurbsCurve(curve_dag)

    # get the points in world space
    mesh_points: om2.MPointArray = fn_mesh.getPoints(space=om2.MSpace.kWorld)

    point_weights: list[tuple[om2.MPoint, list[tuple[Any, float]]]] = []
    knots = spline.get_knots(curve_shape=curve_shape)
    vertex_colors: list[om2.MColor] = []
    vertex_indices: list[int] = []

    # iterate over the points and get the closest parameter
    parameters: list[float] = []
    for i, point in enumerate(mesh_points):
        parameter: float = fn_curve.closestPoint(point, space=om2.MSpace.kWorld)[1]
        parameters.append(parameter)
    spline_weights_per_vertex: list[list[tuple[Any, float]]] = spline.get_weights_along_spline(
        cvs=cv_transforms, parameters=parameters, degree=degree, knots=knots
    )

    return spline_weights_per_vertex


def visualize_split_weights(mesh: str, cv_transforms: list[str], degree: int = 2) -> None:
    """
    Visualizes spline-based weights as vertex colors on a mesh.

    The function assigns a unique color to each CV based on its hashed position. Then, for each vertex
    on the mesh, it computes the weighted color by blending CV colors using the spline-based weights.
    These vertex colors are set on the mesh and can be used to visually verify how influence weights
    fall off across the mesh.

    Args:
        mesh (str): The mesh transform node to visualize on.
        cv_transforms (list[str]): A list of transform names representing the CVs of the curve.
        degree (int, optional): Degree of the spline curve. Defaults to 2.

    Returns:
        None
    """

    # get the shape node
    mesh_shape: str = cmds.listRelatives(mesh, shapes=True)[0]
    cv_positions: list[list[float, float, float]] = []
    cv_colors: dict[str, om2.MColor] = {}
    for transform in cv_transforms:
        position: list[float, float, float] = cmds.xform(
            transform, query=True, worldSpace=True, translation=True
        )
        cv_positions.append(position)
        position_tuple: tuple[float, float, float] = tuple(position)

        lab_color: om2.MColor = om2.MColor(
            linear_srgb_to_oklab(
                colorsys.hsv_to_rgb(
                    (hash(position_tuple) % 128) / 128,
                    0.8,
                    0.9,
                )
            )
        )
        cv_colors[transform] = lab_color

    curve_transform: str = cmds.curve(
        name=f"{mesh}_SplineWeightsTempCurve", point=cv_positions, degree=degree
    )

    # get the shape nodes
    mesh_shape: str = cmds.listRelatives(mesh, shapes=True)[0]
    curve_shape: str = cmds.listRelatives(curve_transform, shapes=True)[0]

    # make sure the target shape can show vertex colors
    cmds.setAttr(f"{mesh_shape}.displayColors", 1)
    cmds.setAttr(f"{mesh_shape}.displayColorChannel", "Diffuse", type="string")

    # get the MDagPaths
    msel: om2.MSelectionList = om2.MSelectionList()
    msel.add(mesh_shape)
    msel.add(curve_shape)
    mesh_dag: om2.MDagPath = msel.getDagPath(0)
    curve_dag: om2.MDagPath = msel.getDagPath(1)

    # make the function set and get the points
    fn_mesh: om2.MFnMesh = om2.MFnMesh(mesh_dag)
    fn_curve: om2.MFnNurbsCurve = om2.MFnNurbsCurve(curve_dag)

    # get the points in world space
    mesh_points: om2.MPointArray = fn_mesh.getPoints(space=om2.MSpace.kWorld)

    point_weights: list[tuple[om2.MPoint, list[tuple[Any, float]]]] = []
    knots = spline.get_knots(curve_shape=curve_shape)
    vertex_colors: list[om2.MColor] = []
    vertex_indices: list[int] = []
    spline_weights_per_vertex = get_mesh_spline_weights(
        mesh_shape=mesh_shape, cv_transforms=cv_transforms, degree=degree
    )
    # iterate over the points and assign colors
    for i, point in enumerate(mesh_points):
        point_color: om2.MColor = om2.MColor([0, 0, 0])
        weights: list[tuple[Any, float]] = spline_weights_per_vertex[i]
        for transform, weight in weights:
            point_color += cv_colors[transform] * weight
        point_color_tuple: tuple[float, float, float, float] = tuple(point_color.getColor())
        point_color_rgb = oklab_to_linear_srgb(color=point_color_tuple)
        point_color = om2.MColor(point_color_rgb)
        vertex_colors.append(point_color)
        vertex_indices.append(i)
        # fn_mesh.setVertexColor(point_color, i)

    # Set all vertex colors at once
    fn_mesh.setVertexColors(vertex_colors, vertex_indices)
    cmds.delete(curve_transform)
    return
