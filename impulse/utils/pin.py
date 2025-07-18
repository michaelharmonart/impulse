import maya.cmds as cmds


def make_uv_pin(
    object_to_pin: str,
    surface: str,
    u: float = 0,
    v: float = 0,
    local_space: bool = False,
    normalize: bool = False,
    normal_axis: str | None = "y",
    tangent_axis: str | None = "x",
    reset_transforms: bool = True,
) -> str:
    """
    Create a UVPin node that pins an object to a given surface at specified UV coordinates.

    Args:
        object_to_pin: The name of the object to be pinned.
        surface: The name of the surface (mesh or NURBS) to pin to.
        u: The U coordinate.
        v: The V coordinate.
        local_space: When true, sets UVPin node to local relativeSpaceMode. 
        When false, the pinned object has inheritsTransform disabled to prevent double transforms.
        normalize: Enable Isoparm normalization (NURBS UV will be remapped between 0-1).
        normal_axis: Normal axis of the generated uvPin, can be x y z -x -y -z.
        tangent_axis: Tangent axis of the generated uvPin, can be x y z -x -y -z.
        reset_transforms: When True, reset the pinned object's transforms.
    Returns:
        The name of the created UVPin node.
    """
    # Retrieve shape nodes from the surface.
    shapes = cmds.listRelatives(surface, children=True, shapes=True) or []
    if not shapes:
        cmds.error(f"No shape nodes found on surface: {surface}")

    # Choose the primary shape (non-intermediate if available) and check for an existing intermediate shape.
    primary_shape = next(
        (s for s in shapes if not cmds.getAttr(f"{s}.intermediateObject")), shapes[0]
    )
    shape_origin = next((s for s in shapes if cmds.getAttr(f"{s}.intermediateObject")), None)

    # Determine attribute names based on surface type.
    surface_type = cmds.objectType(primary_shape)
    if surface_type == "mesh":
        attr_input = ".inMesh"
        attr_world = ".worldMesh[0]"
        attr_output = ".outMesh"
    elif surface_type == "nurbsSurface":
        attr_input = ".create"
        attr_world = ".worldSpace[0]"
        attr_output = ".local"
    else:
        cmds.error(f"Unsupported surface type: {surface_type}")

    # If no intermediate shape exists, create one.
    if shape_origin is None:
        duplicated = cmds.duplicate(primary_shape)[0]
        shape_origin_list = cmds.listRelatives(duplicated, children=True, shapes=True)
        if not shape_origin_list:
            cmds.error("Could not create intermediate shape.")
        shape_origin = shape_origin_list[0]
        cmds.parent(shape_origin, surface, shape=True, relative=True)
        cmds.delete(duplicated)
        new_name = f"{primary_shape}Orig"
        shape_origin = cmds.rename(shape_origin, new_name)
        # If there is an incoming connection, reconnect it.
        in_conn = cmds.listConnections(
            f"{primary_shape}{attr_input}", plugs=True, connections=True, destination=True
        )
        if in_conn:
            cmds.connectAttr(in_conn[1], f"{shape_origin}{attr_input}")
        cmds.connectAttr(f"{shape_origin}{attr_world}", f"{primary_shape}{attr_input}", force=True)
        cmds.setAttr(f"{shape_origin}.intermediateObject", 1)

    # Create the UVPin node and connect it.
    uv_pin = cmds.createNode("uvPin", name=f"{object_to_pin}_uvPin")
    cmds.connectAttr(f"{primary_shape}{attr_world}", f"{uv_pin}.deformedGeometry")
    cmds.connectAttr(f"{shape_origin}{attr_output}", f"{uv_pin}.originalGeometry")

    if reset_transforms:
        cmds.xform(object_to_pin, translation=[0, 0, 0], rotation=[0, 0, 0])

    if normal_axis is None:
        normal_axis = "y"
    if tangent_axis is None:
        tangent_axis = "x"

    if normal_axis == "x":
        cmds.setAttr(f"{uv_pin}.normalAxis", 0)
    elif normal_axis == "y":
        cmds.setAttr(f"{uv_pin}.normalAxis", 1)
    elif normal_axis == "z":
        cmds.setAttr(f"{uv_pin}.normalAxis", 2)
    elif normal_axis == "-x":
        cmds.setAttr(f"{uv_pin}.normalAxis", 3)
    elif normal_axis == "-y":
        cmds.setAttr(f"{uv_pin}.normalAxis", 4)
    elif normal_axis == "-z":
        cmds.setAttr(f"{uv_pin}.normalAxis", 5)
    else:
        raise RuntimeError(f"{normal_axis} isn't a valid axis, it should be x y z -x -y -z")

    if tangent_axis == "x":
        cmds.setAttr(f"{uv_pin}.tangentAxis", 0)
    elif tangent_axis == "y":
        cmds.setAttr(f"{uv_pin}.tangentAxis", 1)
    elif tangent_axis == "z":
        cmds.setAttr(f"{uv_pin}.tangentAxis", 2)
    elif tangent_axis == "-x":
        cmds.setAttr(f"{uv_pin}.tangentAxis", 3)
    elif tangent_axis == "-y":
        cmds.setAttr(f"{uv_pin}.tangentAxis", 4)
    elif tangent_axis == "-z":
        cmds.setAttr(f"{uv_pin}.tangentAxis", 5)
    else:
        raise RuntimeError(f"{tangent_axis} isn't a valid axis, it should be x y z -x -y -z")

    cmds.setAttr(f"{uv_pin}.normalizedIsoParms", 0)
    cmds.setAttr(f"{uv_pin}.coordinate[0]", u, v, type="float2")
    cmds.connectAttr(f"{uv_pin}.outputMatrix[0]", f"{object_to_pin}.offsetParentMatrix")

    if normalize:
        cmds.setAttr(f"{uv_pin}.normalizedIsoParms", 1)

    if local_space:
        cmds.setAttr(f"{uv_pin}.relativeSpaceMode", 1)
    else:
        cmds.setAttr(f"{object_to_pin}.inheritsTransform", 0)
    return uv_pin


def consolidate_uvpins() -> None:
    uv_pin_nodes = cmds.ls(type="uvPin")
    uv_pin_dict = {}
    for uv_pin_node in uv_pin_nodes:
        try:
            input_geo: tuple = (
                cmds.listConnections(f"{uv_pin_node}.originalGeometry", source=True, plugs=True)[0],
                cmds.listConnections(f"{uv_pin_node}.deformedGeometry", source=True, plugs=True)[0],
            )
        except:
            continue
        connections = cmds.listConnections(f"{uv_pin_node}", connections=True, plugs=True)
        attributes = cmds.listAttr(f"{uv_pin_node}", multi=True)
        attribute_values = []
        for attribute in attributes:
            if attribute in [
                "uvSetName",
                "normalOverride",
                "railCurve",
                "normalAxis",
                "tangentAxis",
                "normalizedIsoParms",
                "relativeSpaceMode",
                "relativeSpaceMatrix",
            ]:
                attribute_values.append((attribute, cmds.getAttr(f"{uv_pin_node}.{attribute}")))
            if "coordinateU" in attribute:
                attribute_values.append((attribute, cmds.getAttr(f"{uv_pin_node}.{attribute}")))
            if "coordinateV" in attribute:
                attribute_values.append((attribute, cmds.getAttr(f"{uv_pin_node}.{attribute}")))
        if input_geo in uv_pin_dict:
            uv_pin_dict[input_geo].append((connections, attribute_values))
        else:
            uv_pin_dict[input_geo] = [(connections, attribute_values)]
    for input_geo, connections_attributes in uv_pin_dict.items():
        uv_pin = cmds.createNode(
            "uvPin", name=f"{input_geo[1]}_uvPin".replace("Shape.worldSpace", "_master")
        )
        cmds.connectAttr(input_geo[0], f"{uv_pin}.originalGeometry")
        cmds.connectAttr(input_geo[1], f"{uv_pin}.deformedGeometry")
        pin_num: int = 0
        for attribute_value in connections_attributes[0][1]:
            if attribute_value[0] == "uvSetName":
                cmds.setAttr(f"{uv_pin}.{attribute_value[0]}", attribute_value[1], type="string")
            if attribute_value[0] in [
                "relativeSpaceMode",
                "normalAxis",
                "tangentAxis",
                "normalOverride",
                "normalizedIsoParms",
            ]:
                cmds.setAttr(f"{uv_pin}.{attribute_value[0]}", attribute_value[1])
            if attribute_value[0] == "relativeSpaceMatrix":
                cmds.setAttr(f"{uv_pin}.{attribute_value[0]}", attribute_value[1], type="matrix")

        for connection_attribute in connections_attributes:
            connections = connection_attribute[0]
            connection_pairs = list(zip(connections[0::2], connections[1::2]))
            u_map = {}
            v_map = {}
            for connection in connection_pairs:
                if "coordinateU" in connection[0]:
                    cmds.disconnectAttr(connection[1], connection[0])
                    cmds.connectAttr(connection[1], f"{uv_pin}.coordinate[{pin_num}].coordinateU")
                    u_map[pin_num] = True
                if "coordinateV" in connection[0]:
                    cmds.disconnectAttr(connection[1], connection[0])
                    cmds.connectAttr(connection[1], f"{uv_pin}.coordinate[{pin_num}].coordinateV")
                    v_map[pin_num] = True
                if "outputMatrix" in connection[0]:
                    cmds.disconnectAttr(connection[0], connection[1])
                    cmds.connectAttr(f"{uv_pin}.outputMatrix[{pin_num}]", connection[1])
            for attribute in connection_attribute[1]:
                if "coordinateU" in attribute[0]:
                    if pin_num not in u_map:
                        cmds.setAttr(f"{uv_pin}.coordinate[{pin_num}].coordinateU", attribute[1])
                if "coordinateV" in attribute[0]:
                    if pin_num not in v_map:
                        cmds.setAttr(f"{uv_pin}.coordinate[{pin_num}].coordinateV", attribute[1])
            pin_num += 1
    for uv_pin_node in uv_pin_nodes:
        cmds.delete(uv_pin_node)


def make_matrix_pin(
    object_to_pin: str,
    surface: str,
    u: float = 0,
    v: float = 0,
    local_space: bool = False,
    normalize: bool = False,
    normal_axis: str | None = "y",
    tangent_axis: str | None = "x",
    bitangent_axis: str | None = "-z",
    volume_preservation: bool = False,
    reset_transforms: bool = True,
    margin: float = 0.0001,
) -> str:
    """
    Create a network of nodes that pins an object to a given surface at specified UV coordinates with stretch and shear.

    Args:
        object_to_pin: The name of the object to be pinned.
        surface: The name of the surface (NURBS) to pin to.
        u: The U coordinate.
        v: The V coordinate.
        local_space: When false, the pinned object has inheritsTransform disabled to prevent double transforms.
        normalize: Enable Isoparm normalization (NURBS UV will be remapped between 0-1).
        normal_axis: Normal axis of the generated uvPin, can be x y z -x -y -z.
        tangent_axis: Tangent axis of the generated uvPin, can be x y z -x -y -z.
        volume_preservation: When true, the volume of the pinned object will be maintained as it stretches and shears.
        reset_transforms: When True, reset the pinned object's transforms.
        margin: Margin in UV space to clamp from the edges of the surface to avoid spurious calculation.
    Returns:
        The name of the created network node that holds the parameterU and parameterV attributes.
    """
    # Retrieve shape nodes from the surface.
    shapes = cmds.listRelatives(surface, children=True, shapes=True) or []
    if not shapes:
        cmds.error(f"No shape nodes found on surface: {surface}")

    # Choose the primary shape (non-intermediate if available) and check for an existing intermediate shape.
    primary_shape = next(
        (s for s in shapes if not cmds.getAttr(f"{s}.intermediateObject")), shapes[0]
    )
    shape_origin = next((s for s in shapes if cmds.getAttr(f"{s}.intermediateObject")), None)

    # Determine attribute names based on surface type.
    surface_type = cmds.objectType(primary_shape)
    if surface_type == "nurbsSurface":
        attr_input = ".create"
        attr_world = ".worldSpace[0]"
        attr_local = ".local"
        attr_surface = ".inputSurface"
    else:
        cmds.error(f"Unsupported surface type: {surface_type}")

    # If no intermediate shape exists, create one.
    if shape_origin is None:
        duplicated = cmds.duplicate(primary_shape)[0]
        shape_origin_list = cmds.listRelatives(duplicated, children=True, shapes=True)
        if not shape_origin_list:
            cmds.error("Could not create intermediate shape.")
        shape_origin = shape_origin_list[0]
        cmds.parent(shape_origin, surface, shape=True, relative=True)
        cmds.delete(duplicated)
        new_name = f"{primary_shape}Orig"
        shape_origin = cmds.rename(shape_origin, new_name)
        # If there is an incoming connection, reconnect it.
        in_conn = cmds.listConnections(
            f"{primary_shape}{attr_input}", plugs=True, connections=True, destination=True
        )
        if in_conn:
            cmds.connectAttr(in_conn[1], f"{shape_origin}{attr_input}")
        cmds.connectAttr(f"{shape_origin}{attr_world}", f"{primary_shape}{attr_input}", force=True)
        cmds.setAttr(f"{shape_origin}.intermediateObject", 1)

    # Create network node to hold matrix pin attributes
    matrix_pin = cmds.createNode("network", name=f"{object_to_pin}_matrixPin")
    cmds.addAttr(f"{matrix_pin}", longName="parameterU", attributeType="float")
    cmds.addAttr(f"{matrix_pin}", longName="parameterV", attributeType="float")
    cmds.setAttr(f"{matrix_pin}.parameterU", u)
    cmds.setAttr(f"{matrix_pin}.parameterV", v)

    # Create the pointOnSurfaceInfo node and connect it.
    surface_info = cmds.createNode("pointOnSurfaceInfo", name=f"{object_to_pin}_surfInfo")
    orig_surface_info = cmds.createNode("pointOnSurfaceInfo", name=f"{object_to_pin}_origSurfInfo")

    if local_space:
        cmds.connectAttr(f"{primary_shape}{attr_local}", f"{surface_info}.inputSurface")
    else:
        cmds.connectAttr(f"{primary_shape}{attr_world}", f"{surface_info}.inputSurface")

    cmds.connectAttr(f"{shape_origin}{attr_local}", f"{orig_surface_info}.inputSurface")

    u_clamp = cmds.createNode("clampRange", name=f"{matrix_pin}_uClamp")
    cmds.connectAttr(f"{matrix_pin}.parameterU", f"{u_clamp}.input")
    v_clamp = cmds.createNode("clampRange", name=f"{matrix_pin}_vClamp")
    cmds.connectAttr(f"{matrix_pin}.parameterV", f"{v_clamp}.input")

    if normalize:
        cmds.setAttr(f"{orig_surface_info}.turnOnPercentage", 1)
        cmds.setAttr(f"{surface_info}.turnOnPercentage", 1)

        cmds.setAttr(f"{u_clamp}.minimum", 0 + margin)
        cmds.setAttr(f"{u_clamp}.maximum", 1 - margin)
        cmds.setAttr(f"{v_clamp}.minimum", 0 + margin)
        cmds.setAttr(f"{v_clamp}.maximum", 1 - margin)
    else:
        u_min = cmds.createNode("addDoubleLinear", name=f"{u_clamp}_min")
        cmds.connectAttr(f"{shape_origin}.minValueU", f"{u_min}.input1")
        cmds.setAttr(f"{u_min}.input2", margin)
        u_max = cmds.createNode("addDoubleLinear", name=f"{u_clamp}_max")
        cmds.connectAttr(f"{shape_origin}.maxValueU", f"{u_max}.input1")
        cmds.setAttr(f"{u_max}.input2", -margin)

        v_min = cmds.createNode("addDoubleLinear", name=f"{v_clamp}_min")
        cmds.connectAttr(f"{shape_origin}.minValueV", f"{v_min}.input1")
        cmds.setAttr(f"{v_min}.input2", margin)
        v_max = cmds.createNode("addDoubleLinear", name=f"{v_clamp}_max")
        cmds.connectAttr(f"{shape_origin}.maxValueV", f"{v_max}.input1")
        cmds.setAttr(f"{v_max}.input2", -margin)

        cmds.connectAttr(f"{u_min}.output", f"{u_clamp}.minimum")
        cmds.connectAttr(f"{u_max}.output", f"{u_clamp}.maximum")
        cmds.connectAttr(f"{v_min}.output", f"{v_clamp}.minimum")
        cmds.connectAttr(f"{v_max}.output", f"{v_clamp}.maximum")

    cmds.connectAttr(f"{u_clamp}.output", f"{surface_info}.parameterU")
    cmds.connectAttr(f"{v_clamp}.output", f"{surface_info}.parameterV")
    cmds.connectAttr(f"{u_clamp}.output", f"{orig_surface_info}.parameterU")
    cmds.connectAttr(f"{v_clamp}.output", f"{orig_surface_info}.parameterV")

    # Create nodes to normalize tangent magnitudes (relative to rest pose)
    u_length = cmds.createNode("length", name=f"{object_to_pin}_uLength")
    cmds.connectAttr(f"{orig_surface_info}.tangentU", f"{u_length}.input")
    u_norm = cmds.createNode("multiplyDivide", name=f"{object_to_pin}_NormalizedTanU")

    cmds.setAttr(f"{u_norm}.operation", 2)

    cmds.connectAttr(f"{surface_info}.tangentU", f"{u_norm}.input1")
    cmds.connectAttr(f"{u_length}.output", f"{u_norm}.input2X")
    cmds.connectAttr(f"{u_length}.output", f"{u_norm}.input2Y")
    cmds.connectAttr(f"{u_length}.output", f"{u_norm}.input2Z")
    u_norm_attr = f"{u_norm}.output"

    v_length = cmds.createNode("length", name=f"{object_to_pin}_vLength")
    cmds.connectAttr(f"{orig_surface_info}.tangentV", f"{v_length}.input")
    v_norm = cmds.createNode("multiplyDivide", name=f"{object_to_pin}_NormalizedTanV")

    cmds.setAttr(f"{v_norm}.operation", 2)

    cmds.connectAttr(f"{surface_info}.tangentV", f"{v_norm}.input1")
    cmds.connectAttr(f"{v_length}.output", f"{v_norm}.input2X")
    cmds.connectAttr(f"{v_length}.output", f"{v_norm}.input2Y")
    cmds.connectAttr(f"{v_length}.output", f"{v_norm}.input2Z")
    v_norm_attr = f"{v_norm}.output"

    n_norm_attr = f"{surface_info}.normalizedNormal"
    if volume_preservation:
        orig_cross = cmds.createNode("crossProduct", name=f"{object_to_pin}_OrigNormalCross")
        cmds.connectAttr(f"{orig_surface_info}.tangentU", f"{orig_cross}.input1")
        cmds.connectAttr(f"{orig_surface_info}.tangentV", f"{orig_cross}.input2")

        orig_cross_length = cmds.createNode("length", name=f"{object_to_pin}_OrigCrossLength")
        cmds.connectAttr(f"{orig_cross}.output", f"{orig_cross_length}.input")

        cross = cmds.createNode("crossProduct", name=f"{object_to_pin}_normalCross")
        cmds.connectAttr(f"{surface_info}.tangentU", f"{cross}.input1")
        cmds.connectAttr(f"{surface_info}.tangentV", f"{cross}.input2")

        cross_length = cmds.createNode("length", name=f"{object_to_pin}_CrossLength")
        cmds.connectAttr(f"{cross}.output", f"{cross_length}.input")

        cross_volume = cmds.createNode("divide", name=f"{object_to_pin}_Volume")
        cmds.setAttr(f"{cross_volume}.input1", 1)
        cmds.connectAttr(f"{cross_length}.output", f"{cross_volume}.input2")

        norm_volume = cmds.createNode("multDoubleLinear", name=f"{object_to_pin}_NormalizedVolume")
        cmds.connectAttr(f"{cross_volume}.output", f"{norm_volume}.input1")
        cmds.connectAttr(f"{orig_cross_length}.output", f"{norm_volume}.input2")

        n_norm = cmds.createNode("multiplyDivide", name=f"{object_to_pin}_normalizedNorm")
        cmds.setAttr(f"{n_norm}.operation", 1)
        cmds.connectAttr(f"{surface_info}.normalizedNormal", f"{n_norm}.input1")
        cmds.connectAttr(f"{norm_volume}.output", f"{n_norm}.input2X")
        cmds.connectAttr(f"{norm_volume}.output", f"{n_norm}.input2Y")
        cmds.connectAttr(f"{norm_volume}.output", f"{n_norm}.input2Z")

        n_norm_attr = f"{n_norm}.output"

    pinMatrix = cmds.createNode("fourByFourMatrix", name=f"{object_to_pin}_pinMatrix")

    cmds.connectAttr(f"{surface_info}.positionX", f"{pinMatrix}.in30")
    cmds.connectAttr(f"{surface_info}.positionY", f"{pinMatrix}.in31")
    cmds.connectAttr(f"{surface_info}.positionZ", f"{pinMatrix}.in32")

    axis_attr_map: dict[str, tuple[tuple[str, str, str], bool]] = {
        "x": (("in00", "in01", "in02"), False),
        "y": (("in10", "in11", "in12"), False),
        "z": (("in20", "in21", "in22"), False),
        "-x": (("in00", "in01", "in02"), True),
        "-y": (("in10", "in11", "in12"), True),
        "-z": (("in20", "in21", "in22"), True),
    }

    if normal_axis is None:
        normal_axis = "y"
    if tangent_axis is None:
        tangent_axis = "x"
    if bitangent_axis is None:
        bitangent_axis = "-z"

    if axis_attr_map[normal_axis][1]:
        norm_reverse = cmds.createNode("multiplyDivide", name=f"{object_to_pin}_nReverse")
        cmds.connectAttr(n_norm_attr, f"{norm_reverse}.input1")
        cmds.setAttr(f"{norm_reverse}.input2", -1, -1, -1, type="float3")
        n_norm_attr = f"{norm_reverse}.output"
    cmds.connectAttr(f"{n_norm_attr}X", f"{pinMatrix}.{axis_attr_map[normal_axis][0][0]}")
    cmds.connectAttr(f"{n_norm_attr}Y", f"{pinMatrix}.{axis_attr_map[normal_axis][0][1]}")
    cmds.connectAttr(f"{n_norm_attr}Z", f"{pinMatrix}.{axis_attr_map[normal_axis][0][2]}")

    if axis_attr_map[tangent_axis][1]:
        u_reverse = cmds.createNode("multiplyDivide", name=f"{object_to_pin}_uReverse")
        cmds.connectAttr(u_norm_attr, f"{u_reverse}.input1")
        cmds.setAttr(f"{u_reverse}.input2", -1, -1, -1, type="float3")
        u_norm_attr = f"{u_reverse}.output"
    cmds.connectAttr(f"{u_norm_attr}X", f"{pinMatrix}.{axis_attr_map[tangent_axis][0][0]}")
    cmds.connectAttr(f"{u_norm_attr}Y", f"{pinMatrix}.{axis_attr_map[tangent_axis][0][1]}")
    cmds.connectAttr(f"{u_norm_attr}Z", f"{pinMatrix}.{axis_attr_map[tangent_axis][0][2]}")

    if axis_attr_map[bitangent_axis][1]:
        v_reverse = cmds.createNode("multiplyDivide", name=f"{object_to_pin}_vReverse")
        cmds.connectAttr(v_norm_attr, f"{v_reverse}.input1")
        cmds.setAttr(f"{v_reverse}.input2", -1, -1, -1, type="float3")
        v_norm_attr = f"{v_reverse}.output"
    cmds.connectAttr(f"{v_norm_attr}X", f"{pinMatrix}.{axis_attr_map[bitangent_axis][0][0]}")
    cmds.connectAttr(f"{v_norm_attr}Y", f"{pinMatrix}.{axis_attr_map[bitangent_axis][0][1]}")
    cmds.connectAttr(f"{v_norm_attr}Z", f"{pinMatrix}.{axis_attr_map[bitangent_axis][0][2]}")

    if reset_transforms:
        cmds.xform(object_to_pin, translation=[0, 0, 0], rotation=[0, 0, 0])
    cmds.connectAttr(f"{pinMatrix}.output", f"{object_to_pin}.offsetParentMatrix")

    if not local_space:
        cmds.setAttr(f"{object_to_pin}.inheritsTransform", 0)
    return matrix_pin
