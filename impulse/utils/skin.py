import colorsys
import enum

from ngSkinTools2.api.layers import Layer, Layers
from maya.api.OpenMaya import (
    MDagPath,
    MDagPathArray,
    MDoubleArray,
    MFnSingleIndexedComponent,
    MIntArray,
    MObject,
    MPointArray,
    MSelectionList,
)
from impulse.utils.color import linear_srgb_to_oklab, oklab_to_linear_srgb
from typing import Any
import maya.cmds as cmds
from maya.api import OpenMaya as om2
from maya.api import OpenMayaAnim as oma
import os
from impulse.structs.transform import Vector3 as Vector3
from impulse.utils import spline as spline
from ngSkinTools2 import api as ng
from ngSkinTools2.api import Layers, plugin
from ngSkinTools2.api.influenceMapping import InfluenceMappingConfig
from ngSkinTools2.api.transfer import VertexTransferMode


def get_skin_cluster(mesh: str) -> str | None:
    history = cmds.listHistory(mesh, pdo=True) or []
    skin_clusters = cmds.ls(history, type="skinCluster")
    return skin_clusters[0] if skin_clusters else None


def ensure_ng_initialized() -> None:
    if not plugin.is_plugin_loaded():
        plugin.load_plugin()


def init_layers(shape: str) -> ng.Layers:
    skin_cluster = ng.target_info.get_related_skin_cluster(shape)
    layers = ng.layers.init_layers(skin_cluster)
    base_layer: ng.Layer = layers.add("Base Weights")
    base_layer.set_weights
    return layers


def get_or_create_ng_layer(skin_cluster: str, layer_name: str) -> Layer:
    """
    Gets or creates an ngSkinTools2 layer with the given name on the specified shape.

    Args:
        skin_cluster(str): The name of the skinCluster node.
        layer_name (str): The name of the layer to create or retrieve.

    Returns:
        ngSkinTools2.api.layers.Layer: The existing or newly created layer object.
    """

    layers: Layers = Layers(skin_cluster)

    # Check for existing layer
    for layer in layers.list():
        if layer.name == layer_name:
            return layer

    # Create and return new layer
    new_layer = layers.add(layer_name)
    return new_layer


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


def get_mesh_points(
    fn_mesh: om2.MFnMesh, vertex_indices: list[int] | None = None
) -> om2.MPointArray:
    if vertex_indices is None:
        mesh_points: om2.MPointArray = fn_mesh.getPoints(space=om2.MSpace.kWorld)
        vertex_indices = list(range(mesh_points.length()))
    else:
        mesh_points: om2.MPointArray = om2.MPointArray()
        all_points: om2.MPointArray = fn_mesh.getPoints(space=om2.MSpace.kWorld)
        for idx in vertex_indices:
            mesh_points.append(all_points[idx])
    return mesh_points


def get_mesh_spline_weights(
    mesh_shape: str,
    cv_transforms: list[str],
    degree: int = 2,
    vertex_indices: list[int] | None = None,
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
        vertex_indices: A list of vertex indices to output weights for.
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

    mesh_points: MPointArray = get_mesh_points(fn_mesh=fn_mesh, vertex_indices=vertex_indices)

    knots = spline.get_knots(curve_shape=curve_shape)

    # iterate over the points and get the closest parameter
    parameters: list[float] = []
    for i, point in enumerate(mesh_points):
        parameter: float = fn_curve.closestPoint(point, space=om2.MSpace.kWorld)[1]
        parameters.append(parameter)
    spline_weights_per_vertex: list[list[tuple[Any, float]]] = spline.get_weights_along_spline(
        cvs=cv_transforms, parameters=parameters, degree=degree, knots=knots
    )
    cmds.delete(curve_transform)
    return spline_weights_per_vertex


def get_weights_of_influence(skin_cluster: str, joint: str) -> dict[int, float]:
    sel: MSelectionList = om2.MSelectionList()
    sel.add(skin_cluster)
    sel.add(joint)
    skin_cluster_mob: MObject = sel.getDependNode(0)
    joint_dag: MDagPath = sel.getDagPath(1)
    mfn_skin_cluster: oma.MFnSkinCluster = oma.MFnSkinCluster(skin_cluster_mob)

    components: MSelectionList
    weights: list[float]
    components, weights = mfn_skin_cluster.getPointsAffectedByInfluence(joint_dag)

    index_weights: dict[int, float] = {}
    affected_indices: list[int] = []
    for i in range(components.length()):
        dag_path, component = components.getComponent(i)
        fn_comp: MFnSingleIndexedComponent = om2.MFnSingleIndexedComponent(component)
        indices: list[int] = fn_comp.getElements()
        affected_indices.extend(indices)
    for index, weight in zip(affected_indices, weights):
        index_weights[index] = weight

    return index_weights


def get_weights(shape: str, skin_cluster: str | None = None) -> dict[int, dict[str, float]]:
    """
    Retrieves skinCluster weights for all vertices of the given mesh shape.

    This function returns the non-zero skin weights per vertex, mapped to their
    associated influence (joint) names. It uses the Maya API to efficiently extract
    weights from the skinCluster deformer attached to the mesh.

    Args:
        shape (str): The name of the mesh shape node to query. Must have a skinCluster.
        skin_cluster: Optional specification of which skinCluster node.

    Returns:
        dict[int, dict[str, float]: A dictionary mapping each vertex index to a list of
        (joint_name, weight) dictionaries, including only non-zero weights.
    """
    if not skin_cluster:
        skin_cluster: str | None = get_skin_cluster(shape)
        if not skin_cluster:
            raise RuntimeError(f"No skinCluster on {shape}")

    sel: MSelectionList = om2.MSelectionList()
    sel.add(shape)
    sel.add(skin_cluster)
    shape_dag: MDagPath = sel.getDagPath(0)
    skin_cluster_mob: MObject = sel.getDependNode(1)
    mfn_skin_cluster: oma.MFnSkinCluster = oma.MFnSkinCluster(skin_cluster_mob)

    influence_paths = mfn_skin_cluster.influenceObjects()
    influence_map = {
        mfn_skin_cluster.indexForInfluenceObject(path): om2.MFnDependencyNode(path.node()).name()
        for path in influence_paths
    }

    # Create vertex component
    num_verts: int = om2.MFnMesh(shape_dag).numVertices
    fn_comp: MFnSingleIndexedComponent = om2.MFnSingleIndexedComponent()
    vtx_components = fn_comp.create(om2.MFn.kMeshVertComponent)
    fn_comp.addElements(list(range(num_verts)))

    flat_weights: list[float]
    influence_count: int
    flat_weights, influence_count = mfn_skin_cluster.getWeights(shape_dag, vtx_components)

    weights_dict: dict[int, list[tuple[str, float]]] = {}
    for vtx_id in range(num_verts):
        start_index: int = vtx_id * influence_count
        vtx_weights: dict[int, float] = {}
        for i in range(influence_count):
            weight_value = flat_weights[start_index + i]
            if weight_value > 1e-6:
                influence_name = influence_map.get(i)
                if influence_name:
                    vtx_weights[influence_name] = weight_value
        if vtx_weights:
            weights_dict[vtx_id] = vtx_weights
    return weights_dict


def set_weights(
    shape: str,
    new_weights: dict[int, dict[str, float]],
    skin_cluster: str | None = None,
    normalize=True,
) -> None:
    """
    Sets skinCluster weights for all vertices of the given mesh shape.

    Args:
        shape (str): The name of the mesh shape node to query. Must have a skinCluster.
        new_weights (dict): Dictionary of vertex weights: {vtx_index: {influence_name: weight}}.
        skin_cluster: Optional specification of which skinCluster node.
        normalize: When True, the given weights will additionally be normalized.
    """
    if not skin_cluster:
        skin_cluster: str | None = get_skin_cluster(shape)
        if not skin_cluster:
            raise RuntimeError(f"No skinCluster on {shape}")

    # Ensure all influences in new_weights exist on the skinCluster
    all_influences_in_data: set[str] = set(
        influence_name
        for vtx_weights in new_weights.values()
        for influence_name in vtx_weights.keys()
    )

    existing_influences = set(cmds.skinCluster(skin_cluster, query=True, influence=True) or [])
    
    # Add missing influences to the skinCluster
    influences_to_add: list[str] = sorted(all_influences_in_data - existing_influences)
    cmds.skinCluster(skin_cluster, edit=True, addInfluence=influences_to_add, weight=0.0)

    # Get a the actual MFnSkinCluster to apply weights with
    sel: MSelectionList = om2.MSelectionList()
    sel.add(shape)
    sel.add(skin_cluster)
    shape_dag: MDagPath = sel.getDagPath(0)
    skin_cluster_mob: MObject = sel.getDependNode(1)
    mfn_skin_cluster: oma.MFnSkinCluster = oma.MFnSkinCluster(skin_cluster_mob)

    # Get influence indices
    influence_paths: MDagPathArray = mfn_skin_cluster.influenceObjects()
    influence_indices: dict[str, int] = {
        om2.MFnDependencyNode(path.node()).name(): mfn_skin_cluster.indexForInfluenceObject(path)
        for path in influence_paths
    }

    ordered_influences: list[tuple[str, int]] = sorted(
        influence_indices.items(), key=lambda item: item[1]
    )
    ordered_influence_names = [name for name, index in ordered_influences]
    ordered_indices_only = [index for name, index in ordered_influences]

    influence_indices_array: MIntArray = om2.MIntArray()
    for index in ordered_indices_only:
        influence_indices_array.append(index)

    # Create vertex component
    num_verts: int = om2.MFnMesh(shape_dag).numVertices
    fn_comp: MFnSingleIndexedComponent = om2.MFnSingleIndexedComponent()
    vtx_components = fn_comp.create(om2.MFn.kMeshVertComponent)
    fn_comp.addElements(list(range(num_verts)))

    # Create a flat weight list and list of influence indices used
    weights_array: MDoubleArray = om2.MDoubleArray()

    for vtx_id in range(num_verts):
        vtx_weights = new_weights.get(vtx_id, {})
        for influence_name, influence_index in ordered_influences:
            weight = vtx_weights.get(influence_name, 0.0)
            weights_array.append(weight)

    if not mfn_skin_cluster.object().hasFn(om2.MFn.kSkinClusterFilter):
        raise RuntimeError(f"Selected node {skin_cluster} is not a skinCluster")

    # Set weights
    mfn_skin_cluster.setWeights(
        shape_dag,
        vtx_components,
        influence_indices_array,
        weights_array,
        normalize=normalize,
        returnOldWeights=False,
    )


def set_ng_layer_weights(
    shape: str,
    new_weights: dict[int, dict[str, float]],
    layer_name: str = "Generated Weights",
    skin_cluster: str | None = None,
    normalize: bool = True,
) -> None:
    """
    Applies split weights to a new ngSkinTools2 layer.

    Args:
        shape (str): Name of the mesh shape (must be bound to a skinCluster with ngSkinTools2).
        new_weights (dict): Vertex weights as {vtx_index: {influence_name: weight}}.
        layer_name (str): Name for the new layer.
        normalize (bool): Whether to normalize weights per vertex.
    """
    ensure_ng_initialized()
    if not skin_cluster:
        skin_cluster: str | None = get_skin_cluster(shape)
        if not skin_cluster:
            raise RuntimeError(f"No skinCluster on {shape}")

    normalize_value: int = cmds.getAttr(f"{skin_cluster}.normalizeWeights")
    cmds.setAttr(f"{skin_cluster}.normalizeWeights", 0)

    sel: MSelectionList = om2.MSelectionList()
    sel.add(shape)
    sel.add(skin_cluster)
    shape_dag: MDagPath = sel.getDagPath(0)
    skin_cluster_mob: MObject = sel.getDependNode(1)
    mfn_skin_cluster: oma.MFnSkinCluster = oma.MFnSkinCluster(skin_cluster_mob)

    # Get influence indices
    influence_paths: MDagPathArray = mfn_skin_cluster.influenceObjects()
    influence_indices: dict[str, int] = {
        om2.MFnDependencyNode(path.node()).name(): mfn_skin_cluster.indexForInfluenceObject(path)
        for path in influence_paths
    }
    if not ng.get_layers_enabled(skin_cluster):
        init_layers(shape)
    layers: Layers = Layers(skin_cluster)

    # Ensure all influences in new_weights exist on the skinCluster
    all_influences_in_data: set[str] = set(
        influence_name
        for vtx_weights in new_weights.values()
        for influence_name in vtx_weights.keys()
    )

    existing_influences = set(cmds.skinCluster(skin_cluster, query=True, influence=True) or [])

    # Add missing influences to the skinCluster
    for influence in sorted(all_influences_in_data - existing_influences):
        if not cmds.objExists(influence):
            raise RuntimeError(f"Influence '{influence}' does not exist in the scene.")
        cmds.skinCluster(skin_cluster, edit=True, addInfluence=influence, weight=0.0)

    num_verts: int = om2.MFnMesh(shape_dag).numVertices

    # Organize weights by influence rather than vertex
    weights_by_influence: dict[str, dict[int, float]] = {}
    for vertex in new_weights.keys():
        influence_weights: dict[str, float] = new_weights[vertex]
        for influence, weight in influence_weights.items():
            if influence in weights_by_influence:
                weights_by_influence[influence][vertex] = weight
            else:
                weights_by_influence[influence] = {vertex: weight}

    # Create and select new layer
    new_layer: Layer = get_or_create_ng_layer(skin_cluster=skin_cluster, layer_name=layer_name)

    # Build vertex weight arrays
    for influence, id in influence_indices.items():
        weights_list: list[float] = [
            weights_by_influence.get(influence, {}).get(i, 0) for i in range(num_verts)
        ]
        new_layer.set_weights(id, weights_list)

    cmds.setAttr(f"{skin_cluster}.normalizeWeights", normalize_value)


def split_weights(
    mesh: str,
    original_joints: list[str],
    split_joints: list[list[str]],
    degree: int = 2,
    add_ng_layer: bool = True,
) -> None:
    # get the shape node
    mesh_shape: str = cmds.listRelatives(mesh, shapes=True)[0]

    # get the skinCluster and weights
    skin_cluster: str | None = get_skin_cluster(mesh)
    original_weights: dict[int, dict[str, float]] = get_weights(
        shape=mesh_shape, skin_cluster=skin_cluster
    )

    # Copy the original weights for modification.
    new_weights: dict[int, dict[str, float]] = {
        vtx: weights.copy() for vtx, weights in original_weights.items()
    }

    # Organize weights by influence rather than vertex
    weights_by_influence: dict[str, dict[int, float]] = {}
    for vertex in original_weights.keys():
        influence_weights: dict[str, float] = original_weights[vertex]
        for influence, weight in influence_weights.items():
            if influence in weights_by_influence:
                weights_by_influence[influence][vertex] = weight
            else:
                weights_by_influence[influence] = {vertex: weight}

    for original_joint, split_joints_list in zip(original_joints, split_joints):
        vertex_weights: dict[int, float] = weights_by_influence.get(original_joint, {})

        # Get a list of all vertices that are influenced by the given influence/joint
        influenced_vertex_weights: list[tuple[int, float]] = []
        influenced_vertices: list[int] = []
        for vertex in vertex_weights.keys():
            weight: float = vertex_weights[vertex]
            if weight > 0:
                influenced_vertex_weights.append((vertex, weight))
                influenced_vertices.append(vertex)

        spline_weights: list[list[tuple[Any, float]]] = get_mesh_spline_weights(
            mesh_shape=mesh_shape,
            cv_transforms=split_joints_list,
            degree=degree,
            vertex_indices=influenced_vertices,
        )

        for i, (vertex, original_weight) in enumerate(influenced_vertex_weights):
            if vertex not in new_weights:
                # Start with a copy of the original weights
                new_weights[vertex] = original_weights[vertex].copy()

            # Remove original joint weight
            new_weights[vertex][original_joint] = 0.0

            # Add redistributed weights to split joints
            for influence, spline_weight in spline_weights[i]:
                if influence not in new_weights[vertex]:
                    new_weights[vertex][influence] = 0.0
                new_weights[vertex][influence] += spline_weight * original_weight

    if add_ng_layer:
        set_ng_layer_weights(
            shape=mesh_shape,
            new_weights=new_weights,
            skin_cluster=skin_cluster,
            normalize=True,
            layer_name="Split Weights",
        )
    else:
        set_weights(
            shape=mesh_shape, new_weights=new_weights, skin_cluster=skin_cluster, normalize=True
        )


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
    knots: list[float] = spline.get_knots(curve_shape=curve_shape)
    vertex_colors: list[om2.MColor] = []
    vertex_indices: list[int] = []
    spline_weights_per_vertex: list[list[tuple[Any, float]]] = get_mesh_spline_weights(
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
