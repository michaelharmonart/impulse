import maya.cmds as cmds
from . import control_gen as control
#This uses the UVPin command which has no flags, and depends on the settings you have set in the option box in maya. 

def make_UVPin (
        object_to_pin: str,
        surface: str,
        u: float,
        v: float,
    ) -> str:
    shapeOrig = None
    shapes = cmds.listRelatives(surface, c = True, s = True)
    
    if len(shapes)>1:
        for s in shapes:
            io = cmds.getAttr(f"{s}.intermediateObject")
            if io == 1:
                shapeOrig = s
            else:
                shape = s  

    surface_type = cmds.objectType(shapes[0])

    if surface_type == "mesh":
        cAttr = ".inMesh"
        cAttr2 = ".worldMesh[0]"
        cAttr3 = ".outMesh"
    elif surface_type == "nurbsSurface":
        cAttr = ".create"
        cAttr2 = ".worldSpace[0]"
        cAttr3 = ".local"

    #create shape origin if there isn't one
    if shapeOrig is None:
        shape = shapes[0]
        dup = cmds.duplicate(shape)
        shapeOrig = cmds.listRelatives(dup, c = True, s = True)
        cmds.parent(shapeOrig, surface, s= True, r = True)
        cmds.delete(dup)
        shapeOrig = cmds.rename(shapeOrig, f"{shape}Orig")
        #check if inMesh attr has connection
        inConn = cmds.listConnections(f"{shape}{cAttr}", plugs=True, connections=True, destination=True)
        if inConn is not None:
            cmds.connectAttr(inConn[1], f"{shapeOrig}{cAttr}")
        cmds.connectAttr(f"{shapeOrig}{cAttr2}", f"{shape}{cAttr}", f = True)
        cmds.setAttr(f"{shapeOrig}.intermediateObject", 1)
    
    pin = cmds.createNode("uvPin", name = f"{object_to_pin}_uvPin")
    cmds.connectAttr(f"{shape}{cAttr2}", f"{pin}.deformedGeometry")
    cmds.connectAttr(f"{shapeOrig}{cAttr3}", f"{pin}.originalGeometry")
    cmds.xform(object_to_pin, translation=[0,0,0], rotation=[0,0,0])
    cmds.setAttr(f"{pin}.normalAxis", 1)
    cmds.setAttr(f"{pin}.tangentAxis", 0)
    cmds.setAttr(f"{pin}.normalizedIsoParms", 0)
    cmds.setAttr(f"{pin}.coordinate[0]", u,v, type='float2')
    cmds.connectAttr("{0}.outputMatrix[0]", f"{object_to_pin}.offsetParentMatrix") 
    #cmds.select(surface+".uv"+"["+str(u)+"]"+"["+str(v)+"]", replace=True)
    #cmds.select(object_to_pin, add=True)
    #cmds.UVPin()
    return pin

def generate_ribbon (
        nurbs_surface_name: str, 
        number_of_joints: int = 16,
        cyclic: bool = False,
        swap_uv: bool = False,
        local_space: bool = False, 
        control_joints: bool = True, 
        number_of_controls: int = None,
        half_controls: bool = True,
        hide_joints: bool = True,
        hide_surfaces: bool = False,
    ):
    ribbon_length: float = 2

    #Get the shape node and confirm it's a NURBS surface
    surface_shape: str = cmds.listRelatives(nurbs_surface_name, shapes=True)[0]
    if cmds.nodeType(surface_shape) == "nurbsSurface":
        if swap_uv:
            ribbon_length = cmds.getAttr(str(surface_shape)+".minMaxRangeV")[0][1]
            ribbon_width = cmds.getAttr(str(surface_shape)+".minMaxRangeU")[0][1]
        else:
            ribbon_length = cmds.getAttr(str(surface_shape)+".minMaxRangeU")[0][1]
            ribbon_width = cmds.getAttr(str(surface_shape)+".minMaxRangeV")[0][1]
        if not number_of_controls:
            number_of_controls = int(ribbon_length)
            if half_controls:
                number_of_controls = int(number_of_controls/2)

       
        ribbon_object: str = cmds.duplicate(nurbs_surface_name, name=nurbs_surface_name+"_ribbon")[0]
        ribbon_group = cmds.group(ribbon_object, name=ribbon_object+"_GRP")
        
        ctl_group = cmds.group(empty=True, parent=ribbon_group, name=ribbon_object+"_CTL")
        
        if hide_surfaces:
            cmds.hide(nurbs_surface_name)
            cmds.hide(ribbon_object)
        if not local_space:
            cmds.setAttr(ribbon_object+".inheritsTransform", 0)
       
        if control_joints:
            
            if cyclic:
                for i in range(number_of_controls):
                    control_spacing: float = (ribbon_length/(number_of_controls))
                    u: float = (control_spacing*i)
                    v: float = ribbon_width/2
                    uv: list[float, float] = [u, v]
                    if swap_uv:
                        u = uv[1]
                        v = uv[0]
                    position = cmds.pointOnSurface(ribbon_object, position=True, parameterU=u, parameterV=v)
                    cmds.select(ribbon_group, replace=True)
                    joint_name = cmds.joint(position=position, radius=1, name=str(ribbon_object)+"_ControlJoint"+str(i+1)+"_JNT")
                    ctl_name: str = control.generate_control(position, size= 0.4, parent=ribbon_group)
                    ctl_name = cmds.rename(str(ribbon_object)+"_ControlJoint"+str(i+1)+"_CTL")
                    cmds.parent(ctl_name, ctl_group)
                    locator_name: str = cmds.joint(position=position)
                    cmds.parentConstraint(ctl_name, joint_name, weight=1)
                    cmds.scaleConstraint(ctl_name, joint_name, weight=1)
                    make_UVPin(object_to_pin=locator_name, surface=ribbon_object, u=u, v=v)
                    if not local_space:
                        cmds.setAttr(locator_name+".inheritsTransform", 0)
                    cmds.matchTransform(ctl_name, locator_name)
                    cmds.delete(locator_name)
            else:
                for i in range(number_of_controls + 1):
                    control_spacing: float = (ribbon_length/(number_of_controls))
                    u: float = (control_spacing*i)
                    v: float = ribbon_width/2
                    uv: list[float, float] = [u, v]
                    if swap_uv:
                        u = uv[1]
                        v = uv[0]
                    position = cmds.pointOnSurface(ribbon_object, position=True, parameterU=u, parameterV=v)
                    cmds.select(ribbon_group, replace=True)
                    joint_name = cmds.joint(position=position, radius=1, name=str(ribbon_object)+"_ControlJoint"+str(i+1)+"_JNT")
                    ctl_name: str = control.generate_control(position, size= 0.4, parent=ribbon_group)
                    ctl_name = cmds.rename(str(ribbon_object)+"_ControlJoint"+str(i+1)+"_CTL")
                    cmds.parent(ctl_name, ctl_group)
                    cmds.parentConstraint(ctl_name, joint_name, weight=1)
                    cmds.scaleConstraint(ctl_name, joint_name, weight=1)
                    make_UVPin(object_to_pin=locator_name, surface=ribbon_object, u=u, v=v)
                    if not local_space:
                        cmds.setAttr(locator_name+".inheritsTransform", 0)
                    cmds.matchTransform(ctl_name, locator_name)
                    cmds.delete(locator_name)

        if cyclic:
            for i in range(number_of_joints):
                cmds.select(ribbon_group, replace=True)
                u: float = ((i/(number_of_joints))*ribbon_length)
                v: float = ribbon_width/2
                uv: list[float, float] = [u, v]
                if swap_uv:
                    u = uv[1]
                    v = uv[0]
                position = cmds.pointOnSurface(ribbon_object, position=True, parameterU=u, parameterV=v)
                joint_name = cmds.joint(position=position, radius=0.5, name=str(ribbon_object)+"_point"+str(i+1)+"_DEF")
                if hide_joints:
                    cmds.hide(joint_name)
                ctl_name: str = control.generate_control(position, size= 0.2, parent=ribbon_group)
                ctl_name = cmds.rename(str(ribbon_object)+"_point"+str(i+1)+"_CTL")
                make_UVPin(object_to_pin=ctl_name, surface=ribbon_object, u=u, v=v)
                cmds.makeIdentity(ctl_name, apply=False)
                cmds.parent(ctl_name, ctl_group)
                cmds.parentConstraint(ctl_name, joint_name, weight=1)
                cmds.scaleConstraint(ctl_name, joint_name, weight=1)
                if not local_space:
                    cmds.setAttr(ctl_name+".inheritsTransform", 0)
        else:
            for i in range(number_of_joints):
                cmds.select(ribbon_group, replace=True)
                u: float = ((i/(number_of_joints-1))*ribbon_length)
                v: float = ribbon_width/2
                if swap_uv:
                    u = uv[1]
                    v = uv[0]
                position = cmds.pointOnSurface(ribbon_object, position=True, parameterU=u, parameterV=v)
                joint_name = cmds.joint(position=position, radius=0.5, name=str(ribbon_object)+"_point"+str(i+1)+"_DEF")
                if hide_joints:
                    cmds.hide(joint_name)
                ctl_name: str = control.generate_control(position, size= 0.2, parent=ribbon_group)
                ctl_name = cmds.rename(str(ribbon_object)+"_point"+str(i+1)+"_CTL")
                make_UVPin(object_to_pin=ctl_name, surface=ribbon_object, u=u, v=v)
                cmds.makeIdentity(ctl_name, apply=False)
                cmds.parent(ctl_name, ctl_group)
                cmds.parentConstraint(ctl_name, joint_name, weight=1)
                cmds.scaleConstraint(ctl_name, joint_name, weight=1)
                if not local_space:
                    cmds.setAttr(joint_name+".inheritsTransform", 0)
               
def ribbon_from_selected():
    selected_objects: str = []
    selected_objects = cmds.ls(selection=True)
    for object in selected_objects:
        generate_ribbon(object, cyclic=True)

def ribbon_interpolate (
    primary_ribbon_group: str,
    secondary_ribbon_group: str,
    mesh_object: str,
    number_of_loops: int = 4
    ):

    mesh_shape: str = cmds.listRelatives(mesh_object, shapes=True)[0]
    primary_ribbon_children: list[str] = cmds.listRelatives(primary_ribbon_group, children=True, type="transform")
    primary_ribbon_joints: list[str] = []
    for child in primary_ribbon_children:
        if child.endswith("DEF"):
            primary_ribbon_joints.append(child)
    secondary_ribbon_children: list[str] = cmds.listRelatives(secondary_ribbon_group, children=True, type="transform")
    secondary_ribbon_joints: list[str] = []
    for child in secondary_ribbon_children:
        if child.endswith("DEF"):
            secondary_ribbon_joints.append(child)
    if len(primary_ribbon_joints) == len(secondary_ribbon_joints):
        group: str = cmds.group(empty=True, name=primary_ribbon_group.replace("GRP","_Interpolate_GRP"))
        cmds.select(group)
        number_of_joints: int = len(primary_ribbon_joints)
        row_group_list: list[str] = []
        for i in range(number_of_loops):
            row_lerp: float = (1/(number_of_loops+1))*(i+1)
            row_group: str = cmds.group(name=primary_ribbon_group.replace("GRP","Row"+str(i+1)+"_GRP"), empty=True, parent=group)
            row_group_list.append(row_group)
            cmds.addAttr(row_group,longName="rowBlend", attributeType="float")
            cmds.setAttr(row_group+".rowBlend", row_lerp)

        for i in range(number_of_joints):
                primary_joint_location_node: str = cmds.createNode("pointMatrixMult", name=primary_ribbon_joints[i].replace("DEF","Position"))
                cmds.connectAttr(primary_ribbon_joints[i]+".parentMatrix", primary_joint_location_node+".inMatrix")
                cmds.connectAttr(primary_ribbon_joints[i]+".translate", primary_joint_location_node+".inPoint")
                primary_closest_point_node: str = cmds.createNode("closestPointOnMesh" ,name=primary_ribbon_joints[i].replace("DEF","ClosestPoint"))
                cmds.connectAttr(mesh_shape+".outMesh", primary_closest_point_node+".inMesh")
                cmds.connectAttr(primary_joint_location_node+".output", primary_closest_point_node+".inPosition")
                cmds.connectAttr(mesh_object+".worldMatrix", primary_closest_point_node+".inputMatrix")
                #locater: str = cmds.spaceLocator(name=primary_ribbon_joints[i].replace("DEF","LOC"))[0]
                #cmds.connectAttr(primary_closest_point_node+".result.position", locater+".translate")

                secondary_joint_location_node: str = cmds.createNode("pointMatrixMult", name=secondary_ribbon_joints[i].replace("DEF","Position"))
                cmds.connectAttr(secondary_ribbon_joints[i]+".parentMatrix", secondary_joint_location_node+".inMatrix")
                cmds.connectAttr(secondary_ribbon_joints[i]+".translate", secondary_joint_location_node+".inPoint")
                secondary_closest_point_node: str = cmds.createNode("closestPointOnMesh" ,name=secondary_ribbon_joints[i].replace("DEF","ClosestPoint"))
                cmds.connectAttr(mesh_shape+".outMesh", secondary_closest_point_node+".inMesh")
                cmds.connectAttr(secondary_joint_location_node+".output", secondary_closest_point_node+".inPosition")
                cmds.connectAttr(mesh_object+".worldMatrix", secondary_closest_point_node+".inputMatrix")
                #locater: str = cmds.spaceLocator(name=secondary_ribbon_joints[i].replace("DEF","LOC"))[0]
                #cmds.connectAttr(secondary_closest_point_node+".result.position", locater+".translate")
        
                for i in range(number_of_loops):
                    cmds.select(row_group_list[i])
                    row_lerp: float = (1/(number_of_loops+1))*(i)
                    joint_name = cmds.joint(radius=1, name= primary_ribbon_joints[i].replace("DEF","")+"_Row"+str(i+1)+"_CTL")
                    cmds.setAttr(joint_name+".inheritsTransform", 0)
                    uv_pin_node: str = make_UVPin(object_to_pin=joint_name, surface=mesh_object, u=0.5, v=0.5)

                    #cmds.parentConstraint(ctl_name, joint_name, weight=1)

                    blend_node_u: str = cmds.createNode("blendTwoAttr", name=uv_pin_node+"_Blend_U")
                    cmds.connectAttr(primary_closest_point_node+".result.parameterU", blend_node_u+".input[0]")
                    cmds.connectAttr(secondary_closest_point_node+".result.parameterU", blend_node_u+".input[1]")
                    cmds.connectAttr(row_group_list[i]+".rowBlend", blend_node_u+".attributesBlender")
                    blend_node_v: str = cmds.createNode("blendTwoAttr", name=uv_pin_node+"_Blend_V")
                    cmds.connectAttr(primary_closest_point_node+".result.parameterV", blend_node_v+".input[0]")
                    cmds.connectAttr(secondary_closest_point_node+".result.parameterV", blend_node_v+".input[1]")
                    cmds.connectAttr(row_group_list[i]+".rowBlend", blend_node_v+".attributesBlender")
                    
                    cmds.connectAttr(blend_node_u+".output", uv_pin_node+".coordinate[0].coordinateU")
                    cmds.connectAttr(blend_node_v+".output", uv_pin_node+".coordinate[0].coordinateV")
    else:
        raise Exception("Make sure you only have two ribbons selected and the number of joints is the same in both")
