import maya.cmds as cmds
import os
import json
from impulse.structs.transform import Vector3 as Vector3
from impulse.utils import spline as spline
from ngSkinTools2 import api as ng
from ngSkinTools2.api import plugin
from ngSkinTools2.api.influenceMapping import InfluenceMappingConfig
from ngSkinTools2.api.transfer import VertexTransferMode

def ensure_ng_initialized():
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
    #shapes: list[str] = cmds.listRelatives(geometry, children=True, shapes=True) or []
    #if not shapes:
    #    raise RuntimeError(f"No shape nodes found on surface: {geometry}")
    #shape: str = shapes[0]
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

def get_vertex_positions(shape: str) -> list[Vector3]:
    """
    Returns a list of Vector3 positions of all the vertices in the given shape.
    Args:
        shape: The mesh shape node to query for vertex positions.
    Returns:
        list: A list of positions (Vector3)
    """
    surface_type = cmds.objectType(shape)
    if surface_type != "mesh":
        raise RuntimeError(f"Unsupported surface type: {surface_type}")
    vertex_count: int = cmds.polyEvaluate(shape, vertex=True)
    vertex_list = cmds.xform(f"{shape}.vtx[*]", query=True, translation=True, worldSpace=True)
    vertex_positions = [
        Vector3(vertex_list[i], vertex_list[i + 1], vertex_list[i + 2]) for i in range(0, len(vertex_list), 3)
    ]

    return vertex_positions


def map_vertices_to_curve(vertex_positions: list[Vector3], curve_shape: str) -> list[float]:
    temp_info_node = cmds.createNode("nearestPointOnCurve", name="temp_nearestPointOnCurve")
    cmds.connectAttr(f"{curve_shape}.worldSpace", f"{temp_info_node}.inputCurve")
    vertex_parameters: list[float] = []
    for vertex_position in vertex_positions:
        cmds.setAttr(
            f"{temp_info_node}.inPosition", vertex_position.x, vertex_position.y, vertex_position.z, type="float3"
        )
        parameter: float = cmds.getAttr(f"{temp_info_node}.result.parameter")
        vertex_parameters.append(parameter)
    cmds.delete(temp_info_node)
    return vertex_parameters


def skin_mesh(bind_joints: list[str], geometry: str, name: str | None = None, dual_quaternion: bool = False) -> str:
    if not name:
        name: str = f"{geometry}_SC"
    skin_cluster = cmds.skinCluster(bind_joints, geometry, skinMethod=1 if dual_quaternion else 0, name=name)
    return skin_cluster
