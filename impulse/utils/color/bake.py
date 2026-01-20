import maya.api.OpenMaya as om2
import maya.cmds as cmds
from maya.api.OpenMaya import MColor, MFnMesh, MSelectionList

from impulse.utils.color.convert import linear_srgb_to_rec2020, srgb_to_linear_color


def get_texture_from_shader(shader: str) -> str | None:
    # Check if the 'color' plug exists and is connected
    if cmds.objExists(f"{shader}.color"):
        color_inputs = cmds.listConnections(
            f"{shader}.color", source=True, destination=False, type="file"
        )
        if color_inputs:
            return color_inputs[0]

    # Check if the 'baseColor' plug exists and is connected (e.g. Arnold or StandardSurface shaders)
    if cmds.objExists(f"{shader}.baseColor"):
        base_color_inputs = cmds.listConnections(
            f"{shader}.baseColor", source=True, destination=False, type="file"
        )
        if base_color_inputs:
            return base_color_inputs[0]
    return None


def sample_from_file_node(
    file_node: str, uv_list: list[tuple[float, float]]
) -> list[tuple[float, float, float]]:
    """
    Samples a Maya file texture node's color at the given uv positions

    Args:
        file_node (str): Name of the file node (e.g. "file1").
        uv_list (list): List of (u, v) tuples in [0,1].

    Returns:
        list of (r, g, b) float tuples.
    """

    u_list: list[float] = []
    v_list: list[float] = []

    for u, v in uv_list:
        u_list.append(u)
        v_list.append(v)

    flat_color_list: list[float] = cmds.colorAtPoint(
        file_node, coordU=u_list, coordV=v_list, output="RGB"
    )
    rgb_tuples: list[tuple[float, float, float]] = [
        (flat_color_list[i], flat_color_list[i + 1], flat_color_list[i + 2])
        for i in range(0, len(flat_color_list), 3)
    ]
    return rgb_tuples


def face_color_from_texture(mesh: str, anti_alias: bool = False) -> None:
    """
    Samples texture color at each face of the given mesh and assigns the result as per-face vertex color.

    The function traces the mesh's connected shader, extracts the associated file texture,
    and samples the color at the average UV position of each face. The resulting color
    is converted from sRGB to linear and stored as a per-face color on the mesh.

    Args:
        mesh (str): The name of the mesh transform or shape node to process.
        anti_alias (bool): If True, samples color from all UVs of the face and averages them
            for anti-aliased result. If False, samples a single color at the average UV.
    """
    shapes = cmds.listRelatives(mesh, shapes=True) or []
    if not shapes:
        raise RuntimeError(f"No shape node found for {mesh}")
    shape: str = shapes[0]

    # Prepare mesh function set
    sel: MSelectionList = om2.MSelectionList()
    sel.add(shape)
    dag = sel.getDagPath(0)
    fn_mesh: MFnMesh = om2.MFnMesh(dag)

    # Confirm that UVs are available
    uv_set_name: str = fn_mesh.currentUVSetName()
    uv_counts, uv_ids = fn_mesh.getAssignedUVs(uv_set_name)

    # Check if any UVs are assigned at all
    if not uv_ids or not any(uv_counts):
        raise RuntimeError(f"No UVs assigned on mesh: {mesh} in UV set: {uv_set_name}")

    # make sure the target shape can show vertex colors
    cmds.setAttr(f"{shape}.displayColors", 1)
    cmds.setAttr(f"{shape}.displayColorChannel", "Diffuse", type="string")

    # Get shading group(s)
    shading_groups = cmds.listConnections(shape, type="shadingEngine") or []
    if not shading_groups:
        raise RuntimeError(f"No shading group connected to {shape}")

    # Get surface shader
    shader_attr = cmds.connectionInfo(
        f"{shading_groups[0]}.surfaceShader", sourceFromDestination=True
    )
    if not shader_attr:
        raise RuntimeError(f"No surface shader connected {shape}")
    shader_node = shader_attr.split(".")[0]

    # Get texture
    texture_node = get_texture_from_shader(shader_node)
    if not texture_node:
        raise RuntimeError(f"No texture connected to shader {shader_node}")

    face_count: int = fn_mesh.numPolygons
    face_colors: list[MColor] = []
    face_indices: list[int] = []
    uv_sample_coords: list[tuple[float, float]] = []
    face_uv_indices: dict[int, list[int]] = {}
    for face_index in range(face_count):
        # Get UVs and vertices
        face_vertices = fn_mesh.getPolygonVertices(face_index)
        u: float = 0
        v: float = 0
        uv_list: list[tuple[float, float]] = []
        num_face_verts: int = 0
        for index, vert_index in enumerate(face_vertices):
            vert_u, vert_v = fn_mesh.getPolygonUV(face_index, index)
            u += vert_u
            v += vert_v
            uv_list.append((vert_u, vert_v))
            num_face_verts += 1
        u_average = u / num_face_verts
        v_average = v / num_face_verts
        uv_average: tuple[float, float] = (u_average, v_average)

        if anti_alias:
            start_index = len(uv_sample_coords)
            uv_sample_coords.extend(uv_list)
            end_index = len(uv_sample_coords)
            face_uv_indices[face_index] = list(range(start_index, end_index))
        else:
            uv_sample_coords.append(uv_average)
            face_uv_indices[face_index] = [len(uv_sample_coords) - 1]

    sampled_colors: list[tuple[float, float, float]] = sample_from_file_node(
        file_node=texture_node, uv_list=uv_sample_coords
    )

    for face_index, uv_indices in face_uv_indices.items():
        colors = [sampled_colors[i] for i in uv_indices]
        # Average RGB
        avg_color = tuple(sum(channel) / len(colors) for channel in zip(*colors))

        linear_color = linear_srgb_to_rec2020(srgb_to_linear_color(avg_color))

        color = MColor(linear_color)
        face_colors.append(color)
        face_indices.append(face_index)

    fn_mesh.setFaceColors(face_colors, face_indices)
