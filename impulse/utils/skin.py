import maya.cmds as cmds
from impulse.structs.transform import Vector3 as Vector3
from impulse.utils import spline as spline
from ngSkinTools2 import api as ng
from ngSkinTools2.api.influenceMapping import InfluenceMappingConfig
from ngSkinTools2.api.transfer import VertexTransferMode

def init_layers(shape: str) -> ng.Layers:
    skin_cluster = ng.target_info.get_related_skin_cluster(shape)
    layers = ng.layers.init_layers(skin_cluster)
    base_layer: ng.Layer = layers.add("Base Weights")
    base_layer.set_weights
    return layers

def apply_ng_skin_weights(weights_file: str, geometry: str) -> None:
    shapes: list[str] = cmds.listRelatives(geometry, children=True, shapes=True) or []
    if not shapes:
        raise RuntimeError(f"No shape nodes found on surface: {geometry}")
    shape: str = shapes[0]


    config: InfluenceMappingConfig = InfluenceMappingConfig()
    config.use_distance_matching = False
    config.use_name_matching = True

    # run the import
    ng.import_json(
    shape,
    file=weights_file,
    vertex_transfer_mode=VertexTransferMode.vertexId,
    influences_mapping_config=config,
)


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
    vertex_list = cmds.xform(f"{shape}.vtx[*]", query = True, translation = True, worldSpace = True)
    vertex_positions = [Vector3(vertex_list[i], vertex_list[i+1], vertex_list[i+2]) for i in range(0, len(vertex_list), 3)]

    return vertex_positions

def map_vertices_to_curve(vertex_positions: list[Vector3], curve_shape: str) -> list[float]:
    temp_info_node = cmds.createNode("nearestPointOnCurve", name="temp_nearestPointOnCurve")
    cmds.connectAttr(f"{curve_shape}.worldSpace", f"{temp_info_node}.inputCurve")
    vertex_parameters: list[float] = []
    for vertex_position in vertex_positions:
        cmds.setAttr(f"{temp_info_node}.inPosition", vertex_position.x, vertex_position.y, vertex_position.z, type="float3")
        parameter : float = cmds.getAttr(f"{temp_info_node}.result.parameter")
        vertex_parameters.append(parameter)
    cmds.delete(temp_info_node)
    return vertex_parameters

def skin_mesh(bind_joints: list[str], geometry: str, name: str | None = None, dual_quaternion: bool = False) -> str:
    if not name:
        name: str = f"{geometry}_SC"
    skin_cluster = cmds.skinCluster(bind_joints, geometry, skinMethod = 1 if dual_quaternion else 0, name=name)
    return skin_cluster

