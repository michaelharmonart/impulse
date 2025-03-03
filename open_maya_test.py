import maya.api.OpenMaya as om


current_selection: om.MSelectionList = om.MGlobal.getActiveSelectionList()
for i in range(current_selection.length()):
    object_path: om.MDagPath = current_selection.getComponent(i)[0]
    shape_path: om.MDagPath = object_path.extendToShape()
    shape_object: om.MObject = shape_path.node()
    shape_type: om.MFn = shape_object.apiType()
    if shape_type == om.MFn.kNurbsSurface:
        print(shape_path)
