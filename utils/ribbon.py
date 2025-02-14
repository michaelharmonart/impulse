import maya.cmds as cmds
#This uses the UVPin command which has no flags, and depends on the settings you have set in the option box in maya. 

def generate_ribbon (
        nurbs_surface_name: str, 
        number_of_joints: int = 8,
        cyclic: bool = False,
        local_space: bool = False, 
        control_joints: bool = False, 
        number_of_controls: int = None
    ):
    ribbon_length: float = 2
    surface_shape = cmds.listRelatives(nurbs_surface_name, shapes=True)[0]
    if cmds.nodeType(surface_shape) == "nurbsSurface":
        ribbon_length = cmds.getAttr(str(surface_shape)+".minMaxRangeU")[0][1]
        if not number_of_controls:
            number_of_controls = int(ribbon_length)

        ribbon_object: str = cmds.duplicate(nurbs_surface_name, name=nurbs_surface_name+"_ribbon")[0]
        ribbon_group = cmds.group(ribbon_object, name=ribbon_object+"_GRP")

        if control_joints:
            
            if cyclic:
                for i in range(number_of_controls):
                    control_spacing: float = (ribbon_length/(number_of_controls))
                    position = cmds.pointOnSurface(ribbon_object, position=True, parameterU=(control_spacing*i), parameterV=0.5)
                    cmds.select(ribbon_group, replace=True)
                    joint_name = cmds.joint(position=position, radius=1, name=str(ribbon_object)+str(i+1)+"_ControlJoint")
            else:
                for i in range(number_of_controls + 1):
                    control_spacing: float = (ribbon_length/(number_of_controls))
                    position = cmds.pointOnSurface(ribbon_object, position=True, parameterU=(control_spacing*i), parameterV=0.5)
                    cmds.select(ribbon_group, replace=True)
                    joint_name = cmds.joint(position=position, radius=1, name=str(ribbon_object)+str(i+1)+"_ControlJoint")

        if cyclic:
            for i in range(number_of_joints):
                cmds.select(ribbon_group, replace=True)
                position = cmds.pointOnSurface(ribbon_object, position=True, parameterU=((i/(number_of_joints))*ribbon_length), parameterV=0.5)
                joint_name = cmds.joint(position=position, radius=0.5, name=str(ribbon_object)+"_point"+str(i+1)+"_DEF")
                cmds.select(ribbon_object+".uv"+"["+str((i/(number_of_joints-2))*2)+"]"+"[0.5]", replace=True)
                cmds.select(joint_name, add=True)
                cmds.UVPin()
                if not local_space:
                    cmds.setAttr(joint_name+".inheritsTransform", 0)
        else:
            for i in range(number_of_joints):
                cmds.select(ribbon_group, replace=True)
                position = cmds.pointOnSurface(ribbon_object, position=True, parameterU=((i/(number_of_joints-1))*ribbon_length), parameterV=0.5)
                joint_name = cmds.joint(position=position, radius=0.5, name=str(ribbon_object)+"_point"+str(i+1)+"_DEF")
                cmds.select(ribbon_object+".uv"+"["+str((i/(number_of_joints-1))*2)+"]"+"[0.5]", replace=True)
                cmds.select(joint_name, add=True)
                cmds.UVPin()
                if not local_space:
                    cmds.setAttr(joint_name+".inheritsTransform", 0)
               

selected_objects: str = []
selected_objects = cmds.ls(selection=True)
for object in selected_objects:
    generate_ribbon(object, control_joints=True, cyclic=True, local_space=False)