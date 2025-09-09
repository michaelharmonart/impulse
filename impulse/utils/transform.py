from enum import Enum

import maya.cmds as cmds
from maya.api.OpenMaya import (
    MAngle,
    MDagPath,
    MEulerRotation,
    MMatrix,
    MSelectionList,
    MSpace,
    MTransformationMatrix,
)

from impulse.utils.naming import flip_side


class RotationOrder(Enum):
    """Enum for Maya rotation orders."""

    XYZ = 0
    YZX = 1
    ZXY = 2
    XZY = 3
    YXZ = 4
    ZYX = 5


def is_identity_matrix(matrix: list[float], epsilon: float = 0.001) -> bool:
    return all(
        abs(value - identity) < epsilon
        for value, identity in zip(matrix, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    )


def get_shapes(transform: str) -> list[str]:
    # list the shapes of node
    shape_list: list[str] = cmds.listRelatives(
        transform, shapes=True, noIntermediate=True, children=True
    )

    if shape_list:
        return shape_list
    else:
        raise RuntimeError(f"{transform} has no child shape nodes")


def mmatrix_to_list(matrix: MMatrix) -> list[float]:
    return [matrix.getElement(row, col) for row in range(4) for col in range(4)]


def get_world_matrix(transform: str) -> MMatrix:
    """
    Returns the full world matrix of a transform, including rotateAxis, jointOrient, etc.
    Equivalent to Maya's internal world matrix.
    """
    selection = MSelectionList()
    selection.add(transform)
    dag_path: MDagPath = selection.getDagPath(0)
    return dag_path.inclusiveMatrix()


def set_world_matrix(transform: str, matrix: MMatrix, fallback=False) -> None:
    """
    Set the world matrix of a transform by decomposing it into local components.

    Args:
        transform: Maya transform node name.
        matrix: Target world space matrix.
        fallback: If True, use cmds.xform instead of manual decomposition.
    """
    if fallback:
        cmds.xform(transform, worldSpace=True, matrix=mmatrix_to_list(matrix))
    else:
        # Get parent and calculate local matrix
        parents: list[str] = cmds.listRelatives(transform, parent=True)
        parent: str | None = parents[0] if parents else None
        if parent:
            # Has parent - calculate local matrix relative to parent
            parent_world_matrix = get_world_matrix(parent)
            inverse_matrix: MMatrix = parent_world_matrix.inverse()
            local_matrix: MMatrix = matrix * inverse_matrix
        else:
            # No parent - world matrix IS the local matrix
            local_matrix: MMatrix = matrix

        # Apply local matrix using transformation matrix
        transform_matrix: MTransformationMatrix = MTransformationMatrix(local_matrix)
        # Set translation
        translation = transform_matrix.translation(MSpace.kTransform)
        cmds.setAttr(f"{transform}.translate", translation.x, translation.y, translation.z)
        node_type = cmds.nodeType(transform)

        rotate_order = cmds.getAttr(f"{transform}.rotateOrder")
        transform_matrix.reorderRotation(rotate_order + 1)
        rotation = transform_matrix.rotation()
        cmds.setAttr(
            f"{transform}.rotate",
            MAngle(rotation.x).asDegrees(),
            MAngle(rotation.y).asDegrees(),
            MAngle(rotation.z).asDegrees(),
        )

        if node_type == "joint":
            # Zero the rotate channel
            cmds.setAttr(f"{transform}.jointOrient", 0, 0, 0)

        # Set scale
        scale = transform_matrix.scale(MSpace.kTransform)
        cmds.setAttr(f"{transform}.scale", scale[0], scale[1], scale[2])

        # Set shear
        shear = transform_matrix.shear(MSpace.kTransform)
        cmds.setAttr(f"{transform}.shear", shear[0], shear[1], shear[2])


def match_transform(transform: str, target_transform: str) -> None:
    """
    Match a transform to another in world space.

    Args:
        transform: Object to be moved to the specified transform.
        target_transform: Name of the transform to match to.
    """
    source_matrix: MMatrix = get_world_matrix(transform=target_transform)
    set_world_matrix(transform=transform, matrix=source_matrix)


def match_location(transform: str, target_transform: str) -> None:
    """
    Match a transforms location to another in world space.

    Args:
        transform: Object to be moved to the specified transform.
        target_transform: Name of the transform to match to.
    """
    # Get the world-space translation of the target object.
    target_pos = cmds.xform(target_transform, query=True, worldSpace=True, translation=True)

    # Set the world-space translation of the source object to the target's position.
    cmds.xform(transform, worldSpace=True, translation=target_pos)


def zero_rotate_axis(transform: str) -> None:
    node_type = cmds.nodeType(transform)
    if node_type == "joint":
        cmds.joint(transform, edit=True, zeroScaleOrient=True)
    else:
        temp_transform = cmds.group(empty=True, name=f"{transform}_temp")
        match_transform(temp_transform, transform)
        cmds.setAttr(f"{transform}.rotateAxis", 0, 0, 0, type="float3")
        match_transform(transform, temp_transform)
        cmds.delete(temp_transform)


def clean_parent(transform: str, parent: str, joint_orient: bool = True) -> None:
    """
    Parent a node while preserving its world transform without creating
    Maya's intermediate "compensation" transforms.

    - For transforms: world matrix is preserved.
    - For joints (if joint_orient=True): rotation is baked into jointOrient
      and rotate is zeroed, keeping the joint clean for IK/FK.

    Args:
        transform: Node to reparent.
        parent: New parent node.
        joint_orient: If True, bake rotation into jointOrient for joints.
    """
    object_world_matrix: MMatrix = get_world_matrix(transform)
    node_type = cmds.nodeType(transform)
    cmds.parent(transform, parent, relative=True)

    if node_type == "joint" and joint_orient:
        cmds.setAttr(f"{transform}.jointOrient", 0, 0, 0)
        set_world_matrix(transform, object_world_matrix)
        # Get current rotation info
        rotate_order = cmds.getAttr(f"{transform}.rotateOrder")
        rotation = cmds.getAttr(f"{transform}.rotate")[0]
        # Convert rotation XYZ rotate order for replacing the joint orient
        euler: MEulerRotation = MEulerRotation(
            MAngle(rotation[0], MAngle.kDegrees).asRadians(),
            MAngle(rotation[1], MAngle.kDegrees).asRadians(),
            MAngle(rotation[2], MAngle.kDegrees).asRadians(),
            rotate_order,
        )
        euler.reorderIt(MEulerRotation.kXYZ)
        # Apply to jointOrient (convert back to degrees)
        cmds.setAttr(
            f"{transform}.jointOrient",
            MAngle(euler.x).asDegrees(),
            MAngle(euler.y).asDegrees(),
            MAngle(euler.z).asDegrees(),
        )
        # Zero the rotate channel
        cmds.setAttr(f"{transform}.rotate", 0, 0, 0)
    else:
        set_world_matrix(transform, object_world_matrix)


def matrix_constraint(
    source_transform: str,
    constrain_transform: str,
    keep_offset: bool = True,
    local_space: bool = True,
    use_joint_orient: bool = False,
    translate: bool = True,
) -> None:
    """
    Constrain a transform to another

    Args:
        source_transform: joint to match.
        constrain_joint: joint to constrain.
        keep_offset: keep the offset of the constrained transform to the source at time of constraint generation.
        local_space: if False the constrained transform will have inheritsTransform turned off.
        use_joint_orient: when true the joint orient is taken into account, otherwise it is set to zero.
        translate: whether to constrain translation.
    """
    constraint_name: str = constrain_transform.split("|")[-1]

    # Create node to multiply matrices, as well as a counter to make sure to input into the right slot.
    mult_index: int = 0
    mult_matrix: str = cmds.createNode("multMatrix", name=f"{constraint_name}_ConstraintMultMatrix")

    # If we want to keep the offset, we put the position of the constrained transform into
    # the source transform's space and record it.
    if keep_offset:
        # Get the offset matrix
        offset_matrix_node: str = cmds.createNode(
            "multMatrix", name=f"{constraint_name}_OffsetMatrix"
        )
        cmds.connectAttr(
            f"{constrain_transform}.worldMatrix[0]", f"{offset_matrix_node}.matrixIn[0]"
        )
        cmds.connectAttr(
            f"{source_transform}.worldInverseMatrix[0]", f"{offset_matrix_node}.matrixIn[1]"
        )
        offset_matrix = cmds.getAttr(f"{offset_matrix_node}.matrixSum")

        # Check the matrix against an identity matrix. If it's the same within a margin of error,
        # the transforms aren't offset, meaning we can skip that extra matrix multiplication.
        if not is_identity_matrix(matrix=offset_matrix):
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
        cmds.connectAttr(
            f"{constrain_transform}.parentInverseMatrix[0]", f"{mult_matrix}.matrixIn[{mult_index}]"
        )
        mult_index += 1
    else:
        cmds.setAttr(f"{constrain_transform}.inheritsTransform", 0)

    # Create the decomposed matrix and connect it's inputs
    decompose_matrix: str = cmds.createNode(
        "decomposeMatrix", name=f"{constraint_name}_ConstrainMatrixDecompose"
    )
    cmds.connectAttr(f"{mult_matrix}.matrixSum", f"{decompose_matrix}.inputMatrix")
    cmds.connectAttr(f"{constrain_transform}.rotateOrder", f"{decompose_matrix}.inputRotateOrder")

    rotate_attr: str = f"{decompose_matrix}.outputRotate"
    # If it's a joint we have to do a whole bunch of other nonsense to account for joint orient (I was up till 2am because of this)
    if cmds.nodeType(constrain_transform) == "joint":
        if use_joint_orient:
            # Check if the joint orient isn't about 0
            joint_orient: tuple[float, float, float] = cmds.getAttr(
                f"{constrain_transform}.jointOrient"
            )[0]
            if any(abs(i) > 0.01 for i in joint_orient):
                # Get our joint orient and turn it into a matrix
                orient_node: str = cmds.createNode(
                    "composeMatrix", name=f"{constraint_name}_OrientMatrix"
                )
                cmds.connectAttr(f"{constrain_transform}.jointOrient", f"{orient_node}.inputRotate")
                orient_matrix = cmds.getAttr(f"{orient_node}.outputMatrix")

                # We need to compose a different matrix to drive just the rotation due to the joint orient
                orient_offset_node: str = cmds.createNode(
                    "inverseMatrix", name=f"{constraint_name}_OrientOffsetMatrix"
                )
                orient_mult_matrix: str = cmds.createNode(
                    "multMatrix", name=f"{constraint_name}_ConstraintOrientMatrix"
                )
                orient_mult_index: int = 0

                # If we have an offset it'll be our first matrix in the multiplier (same as above)
                if keep_offset:
                    cmds.setAttr(
                        f"{orient_mult_matrix}.matrixIn[{orient_mult_index}]",
                        offset_matrix,
                        type="matrix",
                    )
                    orient_mult_index += 1

                # Next we multiply by the world matrix of the source transform
                cmds.connectAttr(
                    f"{source_transform}.worldMatrix[0]",
                    f"{orient_mult_matrix}.matrixIn[{orient_mult_index}]",
                )
                orient_mult_index += 1

                # Depending on if we need to take a parent into account we'll need a few extra nodes
                # (otherwise just pre-calculate a matrix and plop it in)
                # Bless Jared Love for figuring this out https://www.youtube.com/watch?v=_LNhZB8jQyo
                # Essentially we need to take the inverse of the orient * the world matrix of the parent and multiply by that
                if local_space:
                    # Create a node to multiply the joint orient by the world matrix of the parent
                    orient_parent_mult_matrix: str = cmds.createNode(
                        "multMatrix", name=f"{constraint_name}_ConstraintOrientMultMatrix"
                    )
                    cmds.setAttr(
                        f"{orient_parent_mult_matrix}.matrixIn[0]", orient_matrix, type="matrix"
                    )
                    cmds.connectAttr(
                        f"{constrain_transform}.parentMatrix[0]",
                        f"{orient_parent_mult_matrix}.matrixIn[1]",
                    )

                    # Create an inverse node and connect it to the result of the last step
                    cmds.connectAttr(
                        f"{orient_parent_mult_matrix}.matrixSum",
                        f"{orient_offset_node}.inputMatrix",
                    )

                    # Finally add this to a slot on the matrix multiplier node
                    cmds.connectAttr(
                        f"{orient_offset_node}.outputMatrix",
                        f"{orient_mult_matrix}.matrixIn[{orient_mult_index}]",
                    )
                    orient_mult_index += 1
                else:
                    # If we don't care about a parent, just make a temp inverse node and store the inverse of the joint orient
                    cmds.connectAttr(
                        f"{orient_node}.outputMatrix", f"{orient_offset_node}.inputMatrix"
                    )
                    inverse_orient_matrix = cmds.getAttr(f"{orient_offset_node}.outputMatrix")

                    # And then set it in a slot on the matrix multiplier
                    cmds.setAttr(
                        f"{orient_mult_matrix}.matrixIn[{orient_mult_index}]",
                        inverse_orient_matrix,
                        type="matrix",
                    )
                    orient_mult_index += 1
                    # Cleanup temp node
                    cmds.delete(orient_offset_node)

                #  Hook up the matrix multiplier to our decomposeMatrix and feed it into the rotate attribute of the joint
                orient_decompose_matrix: str = cmds.createNode(
                    "decomposeMatrix", name=f"{constraint_name}_ConstrainOrientDecompose"
                )
                cmds.connectAttr(
                    f"{orient_mult_matrix}.matrixSum", f"{orient_decompose_matrix}.inputMatrix"
                )
                cmds.connectAttr(
                    f"{constrain_transform}.rotateOrder",
                    f"{orient_decompose_matrix}.inputRotateOrder",
                )
                rotate_attr = f"{orient_decompose_matrix}.outputRotate"
        else:
            cmds.setAttr(f"{constrain_transform}.jointOrient", 0, 0, 0, type="float3")
    cmds.setAttr(f"{constrain_transform}.rotateAxis", 0, 0, 0, type="float3")
    cmds.connectAttr(rotate_attr, f"{constrain_transform}.rotate")
    if translate:
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


def orient_to_world(transform: str) -> None:
    """
    Orient a transform to the world.

    Args:
        transform: transform or joint to orient to the world.
    """
    # Remember parent so we can reparent later
    parents: list[str] = cmds.listRelatives(transform, parent=True)
    parent: str | None = parents[0] if parents else None

    # Un-parent to world to avoid inherited transforms
    if parent:
        cmds.parent(transform, world=True, absolute=True)

    # Zero out rotate and jointOrient by using identity rotation
    cmds.makeIdentity(transform, translate=False, rotate=True, scale=False, apply=True)
    if cmds.nodeType(transform) == "joint":
        cmds.makeIdentity(
            transform, translate=False, rotate=True, jointOrient=True, scale=False, apply=True
        )

    # Now joint is aligned to world, reparent it back
    if parent:
        cmds.parent(transform, parent, absolute=True)


def mirror_transform(transform: str, from_side: str = "L", to_side: str = "R") -> str:
    """
    Duplicate and mirror the specified transform (and optionally children) along the X axis.

    Args:
        transform: transform or joint to orient to the world.
        replace_from: Substring to replace in the original name.
        replace_to: Replacement substring for the mirrored name.

    Returns:
        str: name of the new mirrored transform.
    """

    mirror_group = cmds.group(empty=True)

    # Duplicate full hierarchy
    mirrored_roots: list[str] = cmds.duplicate(transform, renameChildren=True)
    mirrored_root: str = mirrored_roots[0]

    # Remember parent so we can reparent later
    parents: list[str] = cmds.listRelatives(mirrored_root, parent=True)
    parent: str | None = parents[0] if parents else None

    # Reparent, apply negative scale in X,and restore original parent.
    cmds.parent(mirrored_root, mirror_group, absolute=True)
    cmds.scale(-1, 1, 1, mirror_group, absolute=True)
    if parent:
        cmds.parent(mirrored_root, parent, absolute=True)
    else:
        cmds.parent(mirrored_root, world=True, absolute=True)
    cmds.delete(mirror_group)

    # Get original and mirrored transforms in top-down order
    original = [transform] + (cmds.listRelatives(transform, allDescendents=True) or [])
    mirrored = [mirrored_root] + (cmds.listRelatives(mirrored_root, allDescendents=True) or [])

    # Rename all transforms
    renamed_transforms = []
    for orig, mirror in zip(original, mirrored):
        new_name = flip_side(orig, from_side, to_side)
        renamed = cmds.rename(mirror, new_name)
        renamed_transforms.append(renamed)

    return renamed_transforms[0]
