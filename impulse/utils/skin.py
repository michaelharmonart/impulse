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
    if not name:
        name: str = f"{geometry}_SC"
    skin_cluster = cmds.skinCluster(
        bind_joints, geometry, skinMethod=1 if dual_quaternion else 0, name=name
    )
    return skin_cluster


def get_mesh_spline_weights(
    mesh: str, cv_transforms: list[str], degree: int = 2
) -> list[tuple[om2.MPoint, list[tuple[Any, float]]]]:

    # Create a curve for checking the closest point
    cv_positions: list[list[float, float, float]] = []
    cv_colors: list[om2.MColor] = []
    for transform in cv_transforms:
        position: list[float, float, float] = cmds.xform(
            transform, query=True, worldSpace=True, translation=True
        )
        position_tuple: tuple[float, float, float] = tuple(position)
        cv_positions.append(position)

        color = (
            om2.MColor(
                [
                    (hash(position_tuple) % 64) / 64,
                    (hash(position_tuple) % 128) / 128,
                    (hash(position_tuple) % 256) / 128,
                ]
            )
            * 0.5
        )
        cv_colors.append(color)

    curve_transform: str = cmds.curve(
        name=f"{mesh}_SplineWeightsTempCurve", point=cv_positions, degree=degree
    )

    # get the shape nodes
    mesh_shape: str = cmds.listRelatives(mesh, shapes=True)[0]
    curve_shape: str = cmds.listRelatives(curve_transform, shapes=True)[0]

    # make sure the target shape can show vertex colors
    cmds.setAttr(f"{mesh_shape}.displayColors", 1)

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

    # iterate over the points and get the deboor weights
    for i, point in enumerate(mesh_points):
        parameter: float = fn_curve.closestPoint(point, space=om2.MSpace.kWorld)[1]
        # color: om2.MColor = om2.MColor([parameter,parameter, parameter])

        degree: int = cmds.getAttr(f"{curve_shape}.degree")
        knots = spline.get_knots(curve_shape=curve_shape)
        weights: list[tuple[Any, float]] = spline.point_on_curve_weights(
            cvs=cv_colors, t=parameter, degree=degree, knots=knots, normalize=False
        )
        point_weights.append((point, weights))
        point_color: om2.MColor = om2.MColor([0, 0, 0])
        for color, weight in weights:
            point_color += color * weight
        fn_mesh.setVertexColor(point_color, i)

    cmds.delete(curve_transform)
    return point_weights
