import maya.cmds as cmds
from maya.api.OpenMaya import (
    MDoubleArray,
    MFnNurbsCurve,
    MFnNurbsCurveData,
    MObject,
    MPoint,
    MPointArray,
    MSpace,
)

from impulse.maya_api import node
from impulse.structs.transform import Vector3
from impulse.utils.spline.math import (
    generate_knots,
    point_on_spline_weights,
    tangent_on_spline_weights,
)

class MatrixSpline:
    def __init__(
        self,
        cv_transforms: list[str],
        degree: int = 3,
        knots: list[float] | None = None,
        periodic: bool = False,
        name: str | None = None,
    ) -> None:
        """
        A matrix-based B-spline representation driven by transform nodes (CVs).

        Encapsulates a B-spline where each control vertex (CV) is represented by a transform
        in the scene. Instead of interpolating only point positions, the spline blends full
        4x4 matrices derived from each CV’s transform, with scale encoded in the matrix’s
        empty elements.

        Args:
            cv_transforms (list[str]): Transform node names used as control vertices.
            degree (int, optional): Spline degree. Defaults to 3.
            knots (list[float] | None, optional): Knot vector. If None, a suitable vector
                is generated from the CV count, degree, and periodic setting.
            periodic (bool, optional): Whether the spline is periodic (closed). Defaults to False.
            name (str | None, optional): Base name for created scene nodes. Defaults to "MatrixSpline".
        """
        self.pinned_transforms: list[str] = []
        self.pinned_drivers: list[str] = []
        self.curve: str | None = None
        self.periodic: bool = periodic
        self.degree: int = degree
        self.cv_transforms: list[str] = cv_transforms
        number_of_cvs: int = len(cv_transforms) + (periodic * degree)
        if knots:
            self.knots: list[float] = knots
        else:
            self.knots: list[float] = generate_knots(
                count=number_of_cvs, degree=degree, periodic=periodic
            )
        if name:
            self.name: str = name
        else:
            self.name: str = "MatrixSpline"

        cv_matrices: list[str] = []
        cv_position_attrs: list[tuple[str, str, str]] = []
        for index, cv_transform in enumerate(cv_transforms):
            # Remove scale and shear from matrix since they will interfere with the
            # linear interpolation of the basis vectors (causing flipping)
            pick_matrix = cmds.createNode("pickMatrix", name=f"{cv_transform}_PickMatrix")
            cmds.connectAttr(f"{cv_transform}.matrix", f"{pick_matrix}.inputMatrix")
            cmds.setAttr(f"{pick_matrix}.useShear", 0)
            cmds.setAttr(f"{pick_matrix}.useScale", 0)
            # Add nodes to connect individual values from the matrix,
            # I don't know why maya makes us do this instead of just connecting directly
            deconstruct_matrix_attribute = f"{pick_matrix}.outputMatrix"
            row1 = node.RowFromMatrixNode(name=f"{cv_transform}_row1")
            cmds.connectAttr(deconstruct_matrix_attribute, row1.matrix)
            cmds.setAttr(row1.input, 0)
            row2 = node.RowFromMatrixNode(name=f"{cv_transform}_row2")
            cmds.connectAttr(deconstruct_matrix_attribute, row2.matrix)
            cmds.setAttr(row2.input, 1)
            row3 = node.RowFromMatrixNode(name=f"{cv_transform}_row3")
            cmds.setAttr(row3.input, 2)
            cmds.connectAttr(deconstruct_matrix_attribute, row3.matrix)
            row4 = node.RowFromMatrixNode(name=f"{cv_transform}_row4")
            cmds.connectAttr(deconstruct_matrix_attribute, row4.matrix)
            cmds.setAttr(row4.input, 3)

            # Rebuild the matrix but encode the scale into the empty values in the matrix
            # (this needs to be extracted after the weighted matrix sum)
            cv_matrix = cmds.createNode("fourByFourMatrix", name=f"{cv_transform}_CvMatrix")
            cmds.connectAttr(f"{row1.output}X", f"{cv_matrix}.in00")
            cmds.connectAttr(f"{row1.output}Y", f"{cv_matrix}.in01")
            cmds.connectAttr(f"{row1.output}Z", f"{cv_matrix}.in02")
            cmds.connectAttr(f"{cv_transform}.scaleX", f"{cv_matrix}.in03")

            cmds.connectAttr(f"{row2.output}X", f"{cv_matrix}.in10")
            cmds.connectAttr(f"{row2.output}Y", f"{cv_matrix}.in11")
            cmds.connectAttr(f"{row2.output}Z", f"{cv_matrix}.in12")
            cmds.connectAttr(f"{cv_transform}.scaleY", f"{cv_matrix}.in13")

            cmds.connectAttr(f"{row3.output}X", f"{cv_matrix}.in20")
            cmds.connectAttr(f"{row3.output}Y", f"{cv_matrix}.in21")
            cmds.connectAttr(f"{row3.output}Z", f"{cv_matrix}.in22")
            cmds.connectAttr(f"{cv_transform}.scaleZ", f"{cv_matrix}.in23")

            cmds.connectAttr(f"{row4.output}X", f"{cv_matrix}.in30")
            cmds.connectAttr(f"{row4.output}Y", f"{cv_matrix}.in31")
            cmds.connectAttr(f"{row4.output}Z", f"{cv_matrix}.in32")
            cmds.connectAttr(f"{row4.output}W", f"{cv_matrix}.in33")

            cv_matrices.append(f"{cv_matrix}.output")
            cv_position_attrs.append((f"{row4.output}X", f"{row4.output}Y", f"{row4.output}Z"))

        # If the curve is periodic there are we need to re-add CVs that move together.
        if periodic:
            for i in range(degree):
                cv_matrices.append(cv_matrices[i])

        self.cv_matrices: list[str] = cv_matrices
        self.cv_position_attrs: list[tuple[str, str, str]] = cv_position_attrs


def bound_curve_from_matrix_spline(
    matrix_spline: MatrixSpline, curve_parent: str | None = None
) -> str:
    """
    Creates a NURBS curve driven by a MatrixSpline’s control transforms.

    This function builds a NURBS curve whose control points are bound directly to
    the CV transforms of a given MatrixSpline. This is useful for having calculating a 
    live attribute for the MatrixSpline arc length for example.

    Args:
        matrix_spline (MatrixSpline): The MatrixSpline instance providing CVs, knots,
            and degree information.
        curve_parent (str | None, optional): Optional parent transform to parent the
            created curve under. If provided, the curve is parented relatively.

    Returns:
        str: The name of the created curve transform node.
    """
    maya_knots: list[float] = matrix_spline.knots[1:-1]
    if matrix_spline.periodic:
        extended_cvs: list[str] = (
            matrix_spline.cv_transforms + matrix_spline.cv_transforms[: matrix_spline.degree]
        )
    else:
        extended_cvs: list[str] = matrix_spline.cv_transforms
    curve_transform: str = cmds.curve(
        name=f"{matrix_spline.name}_Curve",
        point=[
            cmds.xform(cv, query=True, worldSpace=True, translation=True) for cv in extended_cvs
        ],
        periodic=matrix_spline.periodic,
        knot=maya_knots,
        degree=matrix_spline.degree,
    )
    if curve_parent is not None:
        cmds.parent(curve_transform, curve_parent, relative=True)

    for index, cv_position_attrs in enumerate(matrix_spline.cv_position_attrs):
        cmds.connectAttr(cv_position_attrs[0], f"{curve_transform}.controlPoints[{index}].xValue")
        cmds.connectAttr(cv_position_attrs[1], f"{curve_transform}.controlPoints[{index}].yValue")
        cmds.connectAttr(cv_position_attrs[2], f"{curve_transform}.controlPoints[{index}].zValue")
    return curve_transform

def closest_point_on_matrix_spline(
    matrix_spline: MatrixSpline, position: list[float, float, float]
) -> float:
    """
    Finds the closest parameter value on a spline (defined by a MatrixSpline) to a given 3D position.

    Args:
        matrix_spline: Spline definition object.
        position: The world-space point to project onto the spline.

    Returns:
        float: The curve parameter value (in knot space) at the closest point to the input position.
    """
    knots: list[float] = matrix_spline.knots
    degree: int = matrix_spline.degree
    periodic: bool = matrix_spline.periodic
    cv_transforms: list[str] = matrix_spline.cv_transforms
    cv_positions: MPointArray = []
    for transform in cv_transforms:
        cv_position: tuple[float, float, float] = cmds.xform(
            transform, query=True, worldSpace=True, translation=True
        )
        position_tuple: tuple[float, float, float] = tuple(cv_position)
        cv_positions.append(MPoint(*position_tuple))
    maya_knots: list[float] = knots[1:-1]

    fn_data: MFnNurbsCurveData = MFnNurbsCurveData()
    data_obj: MObject = fn_data.create()
    fn_curve: MFnNurbsCurve = MFnNurbsCurve()
    curve_obj: MFnNurbsCurve = fn_curve.create(
        cv_positions,
        MDoubleArray(maya_knots),
        degree,
        MFnNurbsCurve.kOpen if not periodic else MFnNurbsCurve.kPeriodic,
        False,  # create2D
        False,  # not rational
        data_obj,
    )

    parameter: float = fn_curve.closestPoint(
        MPoint(position[0], position[1], position[2]), space=MSpace.kObject
    )[1]

    return parameter


def pin_to_matrix_spline(
    matrix_spline: MatrixSpline,
    pinned_transform: str,
    parameter: float,
    normalize_parameter: bool = True,
    stretch: bool = True,
    primary_axis: tuple[int, int, int] | None = (0, 1, 0),
    secondary_axis: tuple[int, int, int] | None = (0, 0, 1),
    twist: bool = True,
    align_tangent: bool = True,
) -> None:
    """
    Pins a transform to a matrix spline at a given parameter along the curve.

    Args:
        matrix_spline: The matrix spline data object.
        pinned_transform: Transform to pin to the spline.
        parameter: Position along the spline (0–1).
        stretch: Whether to apply automatic scaling along the spline tangent.
        primary_axis (tuple[int, int, int], optional): Local axis of the pinned
            transform that should aim down the spline tangent. Must be one of
            the cardinal axes (±X, ±Y, ±Z). Defaults to (0, 1, 0) (the +Y axis).
        secondary_axis (tuple[int, int, int], optional): Local axis of the pinned
            transform that should be aligned to a secondary reference direction
            from the spline. Used to resolve orientation. Must be one of the
            cardinal axes (±X, ±Y, ±Z) and orthogonal to ``primary_axis``.
            Defaults to (0, 0, 1) (the +Z axis).
        twist (bool): When True the twist is calculated by averaging the secondary axis vector
            as the up vector for the aim matrix. If False no vector is set and the orientation is the swing
            part of a swing twist decomposition.
        align_tangent: When True the pinned segments will align their primary axis along the spline. 
    Returns:
        None
    """
    if not primary_axis:
        primary_axis = (0, 1, 0)
    if not secondary_axis:
        secondary_axis = (0, 0, 1)

    CARDINALS = {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}
    if tuple(primary_axis) not in CARDINALS or tuple(secondary_axis) not in CARDINALS:
        raise ValueError(
            "primary_axis and secondary_axis must be one of the cardinal axes (±X, ±Y, ±Z)."
        )

    cv_matrices: list[str] = matrix_spline.cv_matrices
    degree: int = matrix_spline.degree
    knots: list[float] = matrix_spline.knots
    segment_name: str = pinned_transform

    # Create node that blends the matrices based on the calculated DeBoor weights.
    blended_matrix = cmds.createNode("wtAddMatrix", name=f"{segment_name}_BaseMatrix")
    point_weights = point_on_spline_weights(
        cvs=cv_matrices, t=parameter, degree=degree, knots=knots, normalize=normalize_parameter
    )
    for index, point_weight in enumerate(point_weights):
        cmds.setAttr(f"{blended_matrix}.wtMatrix[{index}].weightIn", point_weight[1])
        cmds.connectAttr(f"{point_weight[0]}", f"{blended_matrix}.wtMatrix[{index}].matrixIn")

    # Create nodes to access the values of the blended matrix node.
    deconstruct_matrix_attribute = f"{blended_matrix}.matrixSum"
    blended_matrix_row1 = node.RowFromMatrixNode(name=f"{blended_matrix}_row1")
    cmds.setAttr(blended_matrix_row1.input, 0)
    cmds.connectAttr(deconstruct_matrix_attribute, blended_matrix_row1.matrix)
    blended_matrix_row2 = node.RowFromMatrixNode(name=f"{blended_matrix}_row2")
    cmds.connectAttr(deconstruct_matrix_attribute, blended_matrix_row2.matrix)
    cmds.setAttr(blended_matrix_row2.input, 1)
    blended_matrix_row3 = node.RowFromMatrixNode(name=f"{blended_matrix}_row3")
    cmds.connectAttr(deconstruct_matrix_attribute, blended_matrix_row3.matrix)
    cmds.setAttr(blended_matrix_row3.input, 2)
    blended_matrix_row4 = node.RowFromMatrixNode(name=f"{blended_matrix}_row4")
    cmds.connectAttr(deconstruct_matrix_attribute, blended_matrix_row4.matrix)
    cmds.setAttr(blended_matrix_row4.input, 3)

    axis_to_row: dict[tuple[int, int, int], node.RowFromMatrixNode] = {
        (1, 0, 0): blended_matrix_row1,
        (0, 1, 0): blended_matrix_row2,
        (0, 0, 1): blended_matrix_row3,
        (-1, 0, 0): blended_matrix_row1,  # flipped
        (0, -1, 0): blended_matrix_row2,
        (0, 0, -1): blended_matrix_row3,
    }

    if align_tangent:
        blended_tangent_matrix = cmds.createNode("wtAddMatrix", name=f"{segment_name}_TangentMatrix")
        tangent_weights = tangent_on_spline_weights(
            cvs=cv_matrices, t=parameter, degree=degree, knots=knots, normalize=normalize_parameter
        )
        for index, tangent_weight in enumerate(tangent_weights):
            cmds.setAttr(
                f"{blended_tangent_matrix}.wtMatrix[{index}].weightIn",
                tangent_weight[1],
            )
            cmds.connectAttr(
                f"{tangent_weight[0]}",
                f"{blended_tangent_matrix}.wtMatrix[{index}].matrixIn",
            )
        tangent_vector_node: node.MultiplyPointByMatrixNode = node.MultiplyPointByMatrixNode(
            name=f"{blended_tangent_matrix}_TangentVector"
        )
        cmds.connectAttr(f"{blended_tangent_matrix}.matrixSum", tangent_vector_node.input_matrix)

        # Create aim matrix node.
        aim_matrix = cmds.createNode("aimMatrix", name=f"{segment_name}_AimMatrix")
        cmds.setAttr(f"{aim_matrix}.primaryMode", 2)
        cmds.setAttr(f"{aim_matrix}.primaryInputAxis", *primary_axis)
        cmds.connectAttr(tangent_vector_node.output, f"{aim_matrix}.primary.primaryTargetVector")

        secondary_row: node.RowFromMatrixNode | None = axis_to_row.get(tuple(secondary_axis))
        if secondary_row and twist:
            cmds.setAttr(f"{aim_matrix}.secondaryMode", 2)
            cmds.setAttr(f"{aim_matrix}.secondaryInputAxis", *secondary_axis)
            cmds.connectAttr(
                f"{secondary_row.output}X", f"{aim_matrix}.secondary.secondaryTargetVectorX"
            )
            cmds.connectAttr(
                f"{secondary_row.output}Y", f"{aim_matrix}.secondary.secondaryTargetVectorY"
            )
            cmds.connectAttr(
                f"{secondary_row.output}Z", f"{aim_matrix}.secondary.secondaryTargetVectorZ"
            )
        else:
            cmds.setAttr(f"{aim_matrix}.secondaryMode", 0)
        rigid_matrix = aim_matrix
        rigid_matrix_output= f"{aim_matrix}.outputMatrix"
    else:
        pick_matrix = cmds.createNode("pickMatrix", name=f"{segment_name}_Ortho")
        cmds.setAttr(f"{pick_matrix}.useTranslate", 1)
        cmds.setAttr(f"{pick_matrix}.useRotate", 1)
        cmds.setAttr(f"{pick_matrix}.useScale", 0)
        cmds.setAttr(f"{pick_matrix}.useShear", 0)
        cmds.connectAttr(deconstruct_matrix_attribute, f"{pick_matrix}.inputMatrix")
        rigid_matrix = pick_matrix
        rigid_matrix_output= f"{pick_matrix}.outputMatrix"

    # Create nodes to access the values of the rigid matrix (aim matrix or pick matrix) node.
    rigid_matrix_row1 = node.RowFromMatrixNode(name=f"{rigid_matrix}_row1")
    cmds.connectAttr(rigid_matrix_output, rigid_matrix_row1.matrix)
    cmds.setAttr(rigid_matrix_row1.input, 0)
    rigid_matrix_row2 = node.RowFromMatrixNode(name=f"{rigid_matrix}_row2")
    cmds.connectAttr(rigid_matrix_output, rigid_matrix_row2.matrix)
    cmds.setAttr(rigid_matrix_row2.input, 1)
    rigid_matrix_row3 = node.RowFromMatrixNode(name=f"{rigid_matrix}_row3")
    cmds.connectAttr(rigid_matrix_output, rigid_matrix_row3.matrix)
    cmds.setAttr(rigid_matrix_row3.input, 2)

    if align_tangent and stretch:
        # Get tangent vector magnitude
        tangent_vector_length = node.LengthNode(name=f"{segment_name}_tangentVectorLength")
        cmds.connectAttr(tangent_vector_node.output, tangent_vector_length.input)
        tangent_vector_length_scaled: node.MultiplyNode = node.MultiplyNode(
            name=f"{segment_name}_tangentVectorLengthScaled"
        )
        cmds.connectAttr(tangent_vector_length.output, tangent_vector_length_scaled.input[0])

        tangent_sample = cmds.getAttr(tangent_vector_node.output)[0]
        tangent_length = Vector3(tangent_sample[0], tangent_sample[1], tangent_sample[2]).length()
        if tangent_length == 0:
            raise RuntimeError(
                f"{pinned_transform} had a tangent magnitude of 0 and wasn't able to be pinned with stretching enabled."
            )
        cmds.setAttr(tangent_vector_length_scaled.input[1], 1 / tangent_length)
        tangent_scale_attr: str = tangent_vector_length_scaled.output
    else:
        tangent_scale_attr = None

    def is_same_axis(axis1: tuple[int, int, int], axis2: tuple[int, int, int]) -> bool:
        # Compare absolute values to handle flips: (0,1,0) == (0,-1,0)
        return tuple(abs(v) for v in axis1) == tuple(abs(v) for v in axis2)

    def scale_vector(
        vector_attr: str, scale_attr: str, node_name: str, axis: tuple[int, int, int]
    ) -> str:
        scale_node = cmds.createNode("multiplyDivide", name=node_name)
        cmds.connectAttr(f"{vector_attr}X", f"{scale_node}.input1X")
        cmds.connectAttr(f"{vector_attr}Y", f"{scale_node}.input1Y")
        cmds.connectAttr(f"{vector_attr}Z", f"{scale_node}.input1Z")
        if stretch and tangent_scale_attr is not None and is_same_axis(axis, primary_axis):
            scalar_to_connect: str = tangent_scale_attr
        else:
            scalar_to_connect: str = scale_attr
        cmds.connectAttr(scalar_to_connect, f"{scale_node}.input2X")
        cmds.connectAttr(scalar_to_connect, f"{scale_node}.input2Y")
        cmds.connectAttr(scalar_to_connect, f"{scale_node}.input2Z")
        return scale_node

    # Create Nodes to re-apply scale
    X_AXIS = (1, 0, 0)
    Y_AXIS = (0, 1, 0)
    Z_AXIS = (0, 0, 1)

    x_scaled: str = scale_vector(
        node_name=f"{segment_name}_xScale",
        vector_attr=rigid_matrix_row1.output,
        scale_attr=f"{blended_matrix_row1.output}W",
        axis=X_AXIS,
    )
    y_scaled: str = scale_vector(
        node_name=f"{segment_name}_yScale",
        vector_attr=rigid_matrix_row2.output,
        scale_attr=f"{blended_matrix_row2.output}W",
        axis=Y_AXIS,
    )
    z_scaled: str = scale_vector(
        node_name=f"{segment_name}_zScale",
        vector_attr=rigid_matrix_row3.output,
        scale_attr=f"{blended_matrix_row3.output}W",
        axis=Z_AXIS,
    )

    # Rebuild the matrix
    output_matrix = cmds.createNode("fourByFourMatrix", name=f"{segment_name}_OutputMatrix")
    cmds.connectAttr(f"{x_scaled}.outputX", f"{output_matrix}.in00")
    cmds.connectAttr(f"{x_scaled}.outputY", f"{output_matrix}.in01")
    cmds.connectAttr(f"{x_scaled}.outputZ", f"{output_matrix}.in02")

    cmds.connectAttr(f"{y_scaled}.outputX", f"{output_matrix}.in10")
    cmds.connectAttr(f"{y_scaled}.outputY", f"{output_matrix}.in11")
    cmds.connectAttr(f"{y_scaled}.outputZ", f"{output_matrix}.in12")

    cmds.connectAttr(f"{z_scaled}.outputX", f"{output_matrix}.in20")
    cmds.connectAttr(f"{z_scaled}.outputY", f"{output_matrix}.in21")
    cmds.connectAttr(f"{z_scaled}.outputZ", f"{output_matrix}.in22")

    cmds.connectAttr(f"{blended_matrix_row4.output}X", f"{output_matrix}.in30")
    cmds.connectAttr(f"{blended_matrix_row4.output}Y", f"{output_matrix}.in31")
    cmds.connectAttr(f"{blended_matrix_row4.output}Z", f"{output_matrix}.in32")

    cmds.connectAttr(f"{output_matrix}.output", f"{pinned_transform}.offsetParentMatrix")
    matrix_spline.pinned_transforms.append(pinned_transform)
