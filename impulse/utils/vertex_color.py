from maya.api.OpenMaya import MColor, MColorArray, MFnMesh, MSelectionList
import maya.cmds as cmds
import maya.api.OpenMaya as om2

def linear_to_srgb_color(linear_color: om2.MColor) -> om2.MColor:
    """
    Convert a linear MColor to sRGB space.

    Args:
        linear_color (om2.MColor): Linear color with RGBA channels in [0,1].

    Returns:
        om2.MColor: sRGB converted color.
    """
    def convert_channel(c: float) -> float:
        if c <= 0.0031308:
            return 12.92 * c
        else:
            return 1.055 * (pow(base=c,exp=(1.0 / 2.4))) - 0.055

    r = convert_channel(linear_color.r)
    g = convert_channel(linear_color.g)
    b = convert_channel(linear_color.b)
    a = linear_color.a  # alpha usually linear, keep unchanged

    # Clamp results between 0 and 1 to avoid out of gamut
    return om2.MColor([
        max(0.0, min(1.0, r)),
        max(0.0, min(1.0, g)),
        max(0.0, min(1.0, b)),
        a
    ])


def srgb_to_linear_color(srgb_color: om2.MColor) -> om2.MColor:
    """
    Convert an sRGB MColor to linear color space.

    Args:
        srgb_color (om2.MColor): sRGB color with RGBA channels in [0,1].

    Returns:
        om2.MColor: Linear color.
    """
    def convert_channel(c: float) -> float:
        if c <= 0.04045:
            return c / 12.92
        else:
            return ((c + 0.055) / 1.055) ** 2.4

    r = convert_channel(srgb_color.r)
    g = convert_channel(srgb_color.g)
    b = convert_channel(srgb_color.b)
    a = srgb_color.a  # alpha usually linear, keep unchanged

    # Clamp between 0 and 1
    return om2.MColor([
        max(0.0, min(1.0, r)),
        max(0.0, min(1.0, g)),
        max(0.0, min(1.0, b)),
        a
    ])


def get_texture_from_shader(shader: str) -> str | None:
    # Check if the 'color' plug exists and is connected
    if cmds.objExists(f"{shader}.color"):
        color_inputs = cmds.listConnections(f"{shader}.color", source=True, destination=False, type="file")
        if color_inputs:
            return color_inputs[0]

    # Check if the 'baseColor' plug exists and is connected (e.g. Arnold or StandardSurface shaders)
    if cmds.objExists(f"{shader}.baseColor"):
        base_color_inputs = cmds.listConnections(f"{shader}.baseColor", source=True, destination=False, type="file")
        if base_color_inputs:
            return base_color_inputs[0]
    return None


def face_color_from_texture(mesh: str, anti_alias: bool = False) -> None:
    shapes = cmds.listRelatives(mesh, shapes=True) or []
    if not shapes:
        raise RuntimeError(f"No shape node found for {mesh}")
    shape: str = shapes[0]

    # make sure the target shape can show vertex colors
    cmds.setAttr(f"{shape}.displayColors", 1)
    cmds.setAttr(f"{shape}.displayColorChannel", "Diffuse", type="string") 

    # Get shading group(s)
    shading_groups = cmds.listConnections(shape, type="shadingEngine") or []
    if not shading_groups:
        raise RuntimeError(f"No shading group connected to {shape}")

    # Get surface shader
    shader_attr = cmds.connectionInfo(f"{shading_groups[0]}.surfaceShader", sourceFromDestination=True)
    if not shader_attr:
        raise RuntimeError(f"No surface shader connected {shape}")
    shader_node = shader_attr.split(".")[0]

    # Get texture
    texture_node = get_texture_from_shader(shader_node)
    if not texture_node:
        raise RuntimeError(f"No texture connected to shader {shader_node}")

    # Prepare mesh function set
    sel: MSelectionList = om2.MSelectionList()
    sel.add(shape)
    dag = sel.getDagPath(0)
    fn_mesh: MFnMesh = om2.MFnMesh(dag)

    
    face_count: int = fn_mesh.numPolygons
    # Sample texture at each face
    face_colors: list[MColor] = []
    face_indices: list[int] = []
    for face_index in range(face_count):
        
        # Get UVs and vertices
        face_vertices = fn_mesh.getPolygonVertices(face_index)
        u: float = 0
        v: float = 0
        u_list: list[float] = []
        v_list: list[float] = []
        num_face_verts: int = 0
        for index, vert_index in enumerate(face_vertices):
            vert_u, vert_v = fn_mesh.getPolygonUV(face_index, index)
            u += vert_u
            v += vert_v
            u_list.append(vert_u)
            v_list.append(vert_v)
            num_face_verts += 1
        u_average = u / num_face_verts
        v_average = v / num_face_verts
        # Sample color from shader using UV
        if anti_alias:
            result = cmds.colorAtPoint(texture_node, output='RGB', u=tuple(u_list), v=tuple(v_list))
            r, g, b = result
        else:
            result = cmds.colorAtPoint(texture_node, output='RGB', u=u_average, v=v_average)

        if result:
            r, g, b = result
            color: MColor = om2.MColor([r, g, b, 1.0])
            color = srgb_to_linear_color(color)
            face_colors.append(color)
            face_indices.append(face_index)
    fn_mesh.setFaceColors(face_colors, face_indices)