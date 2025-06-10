import maya.cmds as cmds


def match_transform(transform: str, target_transform: str) -> None:
    """
    Match a transform to another in world space.

    Args:
        transform: Object to be moved to the specified transform.
        target_transform: Name of the transform to match to.
    """
    source_matrix = cmds.xform(target_transform, query=True, worldSpace=True, matrix=True)
    cmds.xform(transform, worldSpace=True, matrix=source_matrix)


def matrix_constraint(
    source_transform: str, constrain_transform: str, keep_offset: bool = True, local_space: bool = True, use_joint_orient: bool = False,
) -> None:
    """
    Constrain a transform to another

    Args:
        source_transform: joint to match.
        constrain_joint: joint to constrain.
        keep_offset: keep the offset of the constrained transform to the source at time of constraint generation.
        local_space: if False the constrained transform will have inheritsTransform turned off.
        use_joint_orient: when true the joint orient is taken into account, otherwise it is set to zero.
    """

    # Create node to multiply matrices, as well as a counter to make sure to input into the right slot.
    mult_index: int = 0
    mult_matrix: str = cmds.createNode("multMatrix", name=f"{constrain_transform}_ConstraintMultMatrix")

    # If we want to keep the offset, we put the position of the constrained transform into the source transform's space and record it.
    if keep_offset:
        # Get the offset matrix
        offset_matrix_node: str = cmds.createNode("multMatrix", name=f"{constrain_transform}_OffsetMatrix")
        cmds.connectAttr(f"{constrain_transform}.worldMatrix[0]", f"{offset_matrix_node}.matrixIn[0]")
        cmds.connectAttr(f"{source_transform}.worldInverseMatrix[0]", f"{offset_matrix_node}.matrixIn[1]")
        offset_matrix = cmds.getAttr(f"{offset_matrix_node}.matrixSum")
        
        # Check the matrix against an identity matrix. If it's the same within a margin of error the transforms aren't offset.
        # Meaning we can skip that extra matrix multiplication.
        if any(abs(value - identity) > 0.001 for value, identity in zip(offset_matrix, [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])):
            # Put the offset into the matrix multiplier
            cmds.setAttr(f"{mult_matrix}.matrixIn[{mult_index}]", offset_matrix, type="matrix")
            mult_index += 1
        else:
            keep_offset = False

        cmds.delete(offset_matrix_node)

    # Next we multiply by the world matrix of the source transform
    cmds.connectAttr(f"{source_transform}.worldMatrix[0]", f"{mult_matrix}.matrixIn[{mult_index}]")
    mult_index += 1

    # If we have a parent transform we then put it into that space by multiplying by it's worldInverseMatrix
    if local_space:
        cmds.connectAttr(f"{constrain_transform}.parentInverseMatrix[0]", f"{mult_matrix}.matrixIn[{mult_index}]")
        mult_index += 1
    else:
        cmds.setAttr(f"{constrain_transform}.inheritsTransform", 0)

    # Create the decomposed matrix and connect it's input
    decompose_matrix: str = cmds.createNode("decomposeMatrix", name=f"{constrain_transform}_ConstrainMatrixDecompose")
    cmds.connectAttr(f"{mult_matrix}.matrixSum", f"{decompose_matrix}.inputMatrix")

    rotate_attr: str = f"{decompose_matrix}.outputRotate"
    # If it's a joint we have to do a whole bunch of other nonsense to account for joint orient (I was up till 2am because of this)
    if cmds.nodeType(constrain_transform) == "joint":
        if use_joint_orient:
            # Check if the joint orient isn't about 0
            joint_orient: tuple[float, float, float] = cmds.getAttr(f"{constrain_transform}.jointOrient")[0]
            if any(abs(i) > 0.01 for i in joint_orient):
                # Get our joint orient and turn it into a matrix
                orient_node: str = cmds.createNode("composeMatrix", name=f"{constrain_transform}_OrientMatrix")
                cmds.connectAttr(f"{constrain_transform}.jointOrient", f"{orient_node}.inputRotate")
                orient_matrix = cmds.getAttr(f"{orient_node}.outputMatrix")

                # We need to compose a different matrix to drive just the rotation due to the joint orient
                orient_offset_node: str = cmds.createNode("inverseMatrix", name=f"{constrain_transform}_OrientOffsetMatrix")
                orient_mult_matrix: str = cmds.createNode(
                    "multMatrix", name=f"{constrain_transform}_ConstraintOrientMatrix"
                )
                orient_mult_index: int = 0

                # If we have an offset it'll be our first matrix in the multiplier (same as above)
                if keep_offset:
                    cmds.setAttr(f"{orient_mult_matrix}.matrixIn[{orient_mult_index}]", offset_matrix, type="matrix")
                    orient_mult_index += 1

                # Next we multiply by the world matrix of the source transform
                cmds.connectAttr(
                    f"{source_transform}.worldMatrix[0]", f"{orient_mult_matrix}.matrixIn[{orient_mult_index}]"
                )
                orient_mult_index += 1

                # Depending on if we need to take a parent into account we'll need a few extra nodes
                # (otherwise just pre-calculate a matrix and plop it in)
                # Bless Jared Love for figuring this out https://www.youtube.com/watch?v=_LNhZB8jQyo
                # Essentially we need to take the inverse of the orient * the world matrix of the parent and multiply by that
                if local_space:
                    # Create a node to multiply the joint orient by the world matrix of the parent
                    orient_parent_mult_matrix: str = cmds.createNode(
                        "multMatrix", name=f"{constrain_transform}_ConstraintOrientMultMatrix"
                    )
                    cmds.setAttr(f"{orient_parent_mult_matrix}.matrixIn[0]", orient_matrix, type="matrix")
                    cmds.connectAttr(f"{constrain_transform}.parentMatrix[0]", f"{orient_parent_mult_matrix}.matrixIn[1]")

                    # Create an inverse node and connect it to the result of the last step
                    cmds.connectAttr(f"{orient_parent_mult_matrix}.matrixSum", f"{orient_offset_node}.inputMatrix")

                    # Finally add this to a slot on the matrix multiplier node
                    cmds.connectAttr(
                        f"{orient_offset_node}.outputMatrix", f"{orient_mult_matrix}.matrixIn[{orient_mult_index}]"
                    )
                    orient_mult_index += 1
                else:
                    # If we don't care about a parent, just make a temp inverse node and store the inverse of the joint orient
                    cmds.connectAttr(f"{orient_node}.outputMatrix", f"{orient_offset_node}.inputMatrix")
                    inverse_orient_matrix = cmds.getAttr(f"{orient_offset_node}.outputMatrix")

                    # And then set it in a slot on the matrix multiplier
                    cmds.setAttr(
                        f"{orient_mult_matrix}.matrixIn[{orient_mult_index}]", inverse_orient_matrix, type="matrix"
                    )
                    orient_mult_index += 1
                    # Cleanup temp node
                    cmds.delete(orient_offset_node)

                #  Hook up the matrix multiplier to our decomposeMatrix and feed it into the rotate attribute of the joint
                orient_decompose_matrix: str = cmds.createNode(
                    "decomposeMatrix", name=f"{constrain_transform}_ConstrainOrientDecompose"
                )
                cmds.connectAttr(f"{orient_mult_matrix}.matrixSum", f"{orient_decompose_matrix}.inputMatrix")
                rotate_attr = f"{orient_decompose_matrix}.outputRotate"
        else:
            cmds.setAttr(f"{constrain_transform}.jointOrient", 0,0,0, type="float3")

    cmds.connectAttr(rotate_attr, f"{constrain_transform}.rotate")
    cmds.connectAttr(f"{decompose_matrix}.outputTranslate", f"{constrain_transform}.translate")
    cmds.connectAttr(f"{decompose_matrix}.outputScale", f"{constrain_transform}.scale")
    cmds.connectAttr(f"{decompose_matrix}.outputShear", f"{constrain_transform}.shear")


def constrain_transforms(source_transforms: list[str], constrain_transforms: list[str]) -> None:
    """
    Constrain a set of transforms to another

    Args:
        source_transforms: joints to match.
        constrain_transforms: joints to constrain.
    """
    if len(source_transforms) != len(constrain_transforms):
        raise RuntimeError("Number of joints in source joints and constraint joints don't match!")
    for index, transform in enumerate(constrain_transforms):
        matrix_constraint(source_transform=source_transforms[index], constrain_transform=transform)
