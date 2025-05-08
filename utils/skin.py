from ngSkinTools2 import api as ng
from ..structs.transform import Vector3 as Vector3
import maya.cmds as cmds

def init_layers(shape: str) -> ng.Layers:
    skin_cluster = ng.target_info.get_related_skin_cluster(shape)
    layers = ng.layers.init_layers(skin_cluster)
    base_layer: ng.Layer = layers.add("Base Weights")
    base_layer.set_weights
    return layers

def get_vertex_positions(shape: str) -> list[Vector3]:
    vertex_list = cmds.xform(f"{shape}.vtx[*]", query = True, translation = True, worldSpace = True)
    vertex_positions = [Vector3(vertex_list[i], vertex_list[i+1], vertex_list[i+2]) for i in range(0, len(vertex_list), 3)]
    return vertex_positions

