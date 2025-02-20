import maya.cmds as cmds
import maya.mel as mel
from . import control_gen as control
#This uses the UVPin command which has no flags, and depends on the settings you have set in the option box in maya. 

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
                    uv: List[float, float] = [u, v]
                    if swap_uv:
                        u = uv[1]
                        v = uv[0]
                    position = cmds.pointOnSurface(ribbon_object, position=True, parameterU=u, parameterV=v)
                    cmds.select(ribbon_group, replace=True)
                    joint_name = cmds.joint(position=position, radius=1, name=str(ribbon_object)+"_ControlJoint"+str(i+1)+"_JNT")
                    ctl_name: str = control.generate_control(position, size= 0.4, parent=ribbon_group, control_shape=control.ControlShape.SQUARE)
                    ctl_name = cmds.rename(str(ribbon_object)+"_ControlJoint"+str(i+1)+"_CTL")
                    cmds.parent(ctl_name, ctl_group)
                    locator_name: str = cmds.joint(position=position)
                    cmds.parentConstraint(ctl_name, joint_name, weight=1)
                    cmds.scaleConstraint(ctl_name, joint_name, weight=1)
                    cmds.select(ribbon_object+".uv"+"["+str(u)+"]"+"["+str(v)+"]", replace=True)
                    cmds.select(locator_name, add=True)
                    cmds.UVPin()
                    if not local_space:
                        cmds.setAttr(locator_name+".inheritsTransform", 0)
                    cmds.matchTransform(ctl_name, locator_name)
                    cmds.delete(locator_name)
            else:
                for i in range(number_of_controls + 1):
                    control_spacing: float = (ribbon_length/(number_of_controls))
                    u: float = (control_spacing*i)
                    v: float = ribbon_width/2
                    uv: List[float, float] = [u, v]
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
                    cmds.select(ribbon_object+".uv"+"["+str(u)+"]"+"["+str(v)+"]", replace=True)
                    cmds.select(locator_name, add=True)
                    cmds.UVPin()
                    if not local_space:
                        cmds.setAttr(locator_name+".inheritsTransform", 0)
                    cmds.matchTransform(ctl_name, locator_name)
                    cmds.delete(locator_name)

        if cyclic:
            for i in range(number_of_joints):
                cmds.select(ribbon_group, replace=True)
                u: float = ((i/(number_of_joints))*ribbon_length)
                v: float = ribbon_width/2
                uv: List[float, float] = [u, v]
                if swap_uv:
                    u = uv[1]
                    v = uv[0]
                position = cmds.pointOnSurface(ribbon_object, position=True, parameterU=u, parameterV=v)
                joint_name = cmds.joint(position=position, radius=0.5, name=str(ribbon_object)+"_point"+str(i+1)+"_DEF")
                if hide_joints:
                    cmds.hide(joint_name)
                ctl_name: str = control.generate_control(position, size= 0.2, parent=ribbon_group)
                ctl_name = cmds.rename(str(ribbon_object)+"_point"+str(i+1)+"_CTL")
                cmds.select(ribbon_object+".uv"+"["+str(u)+"]"+"["+str(v)+"]", replace=True)
                cmds.select(ctl_name, add=True)
                cmds.UVPin()
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
                cmds.select(ribbon_object+".uv"+"["+str(u)+"]"+"["+str(v)+"]", replace=True)
                cmds.select(ctl_name, add=True)
                cmds.UVPin()
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