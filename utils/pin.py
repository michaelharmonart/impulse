import maya.cmds as cmds

def make_uv_pin (
        object_to_pin: str, 
        surface: str, 
        u: float = 0, 
        v: float = 0,
        local_space: bool = False,
        normalize: bool = False,
        normal_axis: str = None,
        tangent_axis: str = None,
        reset_transforms: bool = True,
) -> str:
    """
    Create a UVPin node that pins an object to a given surface at specified UV coordinates.

    Args:
        object_to_pin: The name of the object to be pinned.
        surface: The name of the surface (mesh or NURBS) to pin to.
        u: The U coordinate.
        v: The V coordinate.
        local_space: When true, sets UVPin node to local relativeSpaceMode. When false, the pinned object has inheritsTransform disabled to prevent double transforms.
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
    primary_shape = next((s for s in shapes if not cmds.getAttr(f"{s}.intermediateObject")), shapes[0])
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
        in_conn = cmds.listConnections(f"{primary_shape}{attr_input}", plugs=True, connections=True, destination=True)
        if in_conn:
            cmds.connectAttr(in_conn[1], f"{shape_origin}{attr_input}")
        cmds.connectAttr(f"{shape_origin}{attr_world}", f"{primary_shape}{attr_input}", force=True)
        cmds.setAttr(f"{shape_origin}.intermediateObject", 1)
    
    # Create the UVPin node and connect it.
    uv_pin = cmds.createNode("uvPin", name=f"{object_to_pin}_uvPin")
    cmds.connectAttr(f"{primary_shape}{attr_world}", f"{uv_pin}.deformedGeometry")
    cmds.connectAttr(f"{shape_origin}{attr_output}", f"{uv_pin}.originalGeometry")
    cmds.xform(object_to_pin, translation=[0, 0, 0], rotation=[0, 0, 0])
    
    if normal_axis:
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
    else:
        cmds.setAttr(f"{uv_pin}.normalAxis", 1)

    if tangent_axis:
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
    else:
        cmds.setAttr(f"{uv_pin}.tangentAxis", 0)
    

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
            input_geo: tuple = (cmds.listConnections(f"{uv_pin_node}.originalGeometry", source=True, plugs=True)[0], cmds.listConnections(f"{uv_pin_node}.deformedGeometry", source=True, plugs=True)[0])
        except: 
            continue
        connections = cmds.listConnections(f"{uv_pin_node}", connections=True, plugs=True)
        attributes = cmds.listAttr(f"{uv_pin_node}", multi=True)
        attribute_values = []
        for attribute in attributes:
            if attribute in ["uvSetName", "normalOverride", "railCurve", "normalAxis", "tangentAxis", "normalizedIsoParms", "relativeSpaceMode", "relativeSpaceMatrix"]:
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
        uv_pin = cmds.createNode("uvPin", name=f"{input_geo[1]}_uvPin".replace("Shape.worldSpace", "_master"))
        cmds.connectAttr(input_geo[0], f"{uv_pin}.originalGeometry")
        cmds.connectAttr(input_geo[1], f"{uv_pin}.deformedGeometry")
        pin_num: int = 0 
        for attribute_value in connections_attributes[0][1]:
            if attribute_value[0] == "uvSetName":
                cmds.setAttr(f"{uv_pin}.{attribute_value[0]}", attribute_value[1], type="string")
            if attribute_value[0] in ["relativeSpaceMode", "normalAxis", "tangentAxis", "normalOverride", "normalizedIsoParms"]:
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