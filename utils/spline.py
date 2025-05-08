"""
Functions for working with splines.

"""

from inspect import Parameter
import maya.cmds as cmds
from ..structs.transform import Vector3 as Vector3


def generateKnots(count: int, degree: int = 3) -> list[float]:
    # Algorithm and code originally from Cole O'Brien
    # https://coleobrien.medium.com/matrix-splines-in-maya-ec17f3b3741
    # https://gist.github.com/obriencole11/354e6db8a55738cb479523f15f1fd367
    """
    Gets a default knot vector for a given number of cvs and degrees.
    Args:
        count(int): The number of cvs.
        degree(int): The curve degree.
    Returns:
        list: A list of knot values. (aka knot vector)
    """
    knots = [0 for i in range(degree)] + [
        i for i in range(count - degree + 1)
    ]  # put degree number of 0s at the beginning
    knots += [count - degree for i in range(degree)]  # put degree number of the last knot value at the end
    return [float(knot) for knot in knots]


def getKnots(curve_shape: str) -> list[float]:
    # Refer to https://openusd.org/dev/api/class_usd_geom_nurbs_curves.html
    curve_info = cmds.createNode("curveInfo", name=f"temp_curveInfo")
    cmds.connectAttr(f"{curve_shape}.worldSpace", f"{curve_info}.inputCurve")
    knots: list[float] = cmds.getAttr(f"{curve_info}.knots[*]")
    cmds.delete(curve_info)

    knots.insert(0, 0)
    knots.append(0)
    if cmds.getAttr(f"{curve_shape}.form") == 2:
        knots[0] = knots[1] - (knots[-1] - knots[-3])
        knots[-1] = knots[-2] - (knots[2] - knots[1])
    else:
        knots[0] = knots[1]
        knots[-1] = knots[-2]
    return knots


def getCvs(curve_shape: str) -> list[Vector3]:
    curve_info = cmds.createNode("curveInfo", name=f"temp_curveInfo")
    cmds.connectAttr(f"{curve_shape}.worldSpace", f"{curve_info}.inputCurve")
    cv_list: list[float] = cmds.getAttr(f"{curve_info}.controlPoints[*]")
    print(cv_list)
    cmds.delete(curve_info)
    position_list = [Vector3(position[0], position[1], position[2]) for position in cv_list]
    return position_list

def deBoorSetup(cvs: list, t: float, degree: int = 3, knots: list[float] = None, normalize: bool = True) -> tuple[list[float], int, float, bool]:
    """
    Extracts information needed for DeBoors Algorithm
    Args:
        cvs(list): A list of cvs, these are used for the return value.
        t(float): A parameter value.
        degree(int): The curve dimensions.
        knots(list): A list of knot values.
        normalize(bool): When true, the curve is parameter is normalized from 0-1
    Returns:
        tuple: Tuple containing list of knot values, span number, parameter(t), and a boolean for wether the curve is periodic.
    """

    order = degree + 1  # Our functions often use order instead of degree
    if len(cvs) <= degree:
        raise ValueError(f"Curves of degree {degree} require at least {degree + 1} CVs.")

    knots = knots or generateKnots(len(cvs), degree)  # Defaults to even knot distribution
    if len(knots) != len(cvs) + order:
        raise ValueError(
            "Not enough knots provided. Curves with %s cvs must have a knot vector of length %s. "
            "Received a knot vector of length %s: %s. "
            "Total knot count must equal len(cvs) + degree + 1." % (len(cvs), len(cvs) + order, len(knots), knots)
        )

    # Determine if curve is periodic
    if (
        knots[degree] != knots[degree - 1]  # no left clamping
        and knots[-degree - 1] != knots[-degree]  # no right clamping
        and knots[0] == knots[1] - (knots[-1] - knots[-3])
        and knots[-1] == knots[-2] - (knots[2] - knots[1])  # fits requirement for periodic knots
    ):
        periodic = True
    else:
        periodic = False

    # Optional normalization of t
    domain_start = knots[degree]
    domain_end = knots[-degree - 1]
    domain_range = domain_end - domain_start

    if normalize:
        t = (t * domain_range) + domain_start
    else:
        t = t + domain_start
    if periodic:
        
        t = ((t - domain_start) % domain_range) + domain_start # Wrap t into valid domain

    # Find knot span (segment)
    segment = None
    for i in range(len(knots) - 1):
        if knots[i] <= t < knots[i + 1]:
            segment = i
            break
    if segment is None:
        # If t == last knot, use the last valid span
        segment = len(knots) - order - 1
    return (knots, segment, t, periodic)

def deBoorWeights(cvs: list, t: float, span: int, degree: int = 3, knots: list[float] = None) -> dict[any, float]:
    """
    Extracts information needed for DeBoors Algorithm
    Args:
        cvs(list): A list of cvs, these are used for the return value.
        t(float): A parameter value.
        span(int): Span index (can be retrieved with deBoorSetup)
        degree(int): The curve dimensions.
        knots(list): A list of knot values.
    Returns:
        dict: Dictionary with cv: weight mappings
    """
    # Run a modified version of de Boors algorithm
    cvWeights = [{cv: 1.0} for cv in cvs]  # initialize weights with a value of 1 for every cv
    for r in range(1, degree + 1):  # Loop once per degree
        for j in range(degree, r - 1, -1):  # Loop backwards from degree to r
            right = j + 1 + span - r
            left = j + span - degree
            alpha = (t - knots[left]) / (
                knots[right] - knots[left]
            )  # Alpha is how much influence comes from the left vs right cv

            weights = {}
            for cv, weight in cvWeights[j].items():
                weights[cv] = weight * alpha

            for cv, weight in cvWeights[j - 1].items():
                if cv in weights:
                    weights[cv] += weight * (1 - alpha)
                else:
                    weights[cv] = weight * (1 - alpha)

            cvWeights[j] = weights

    cvWeights = cvWeights[degree]
    return cvWeights

def pointOnCurveWeights(cvs: list, t: float, degree: int = 3, knots: list[float] = None, normalize: bool = True):
    # Algorithm and code originally from Cole O'Brien
    # https://coleobrien.medium.com/matrix-splines-in-maya-ec17f3b3741
    # https://gist.github.com/obriencole11/354e6db8a55738cb479523f15f1fd367
    """
    Creates a mapping of cvs to curve weight values on a spline curve.
    While all cvs are required, only the cvs with non-zero weights will be returned.
    This function is based on de Boor's algorithm for evaluating splines and has been modified to consolidate weights.
    Args:
        cvs(list): A list of cvs, these are used for the return value.
        t(float): A parameter value.
        degree(int): The curve dimensions.
        knots(list): A list of knot values.
        normalize(bool): When true, the curve is parameter is normalized from 0-1
    Returns:
        list: A list of control point, weight pairs.
    """

    curve_setup = deBoorSetup(cvs=cvs, t=t, degree=degree, knots=knots, normalize=normalize)
    knots = curve_setup[0]
    segment = curve_setup[1]
    t = curve_setup[2]
    periodic = curve_setup[3]

    # Convert cvs into hash-able indices
    _cvs = cvs
    cvs: list[int] = [i for i in range(len(cvs))]

    # Filter out cvs we won't be using
    cvs = [cvs[j + segment - degree] for j in range(0, degree + 1)]

    # Run a modified version of de Boors algorithm
    cvWeights = deBoorWeights(cvs=cvs, t=t, span=segment, degree=degree, knots=knots)
    return [[_cvs[index], weight] for index, weight in cvWeights.items()]


def tangentOnCurveWeights(cvs: list, t: float, degree: int = 3, knots: list[float] = None, normalize: bool = True): 
    # Algorithm and code originally from Cole O'Brien
    # https://coleobrien.medium.com/matrix-splines-in-maya-ec17f3b3741
    # https://gist.github.com/obriencole11/354e6db8a55738cb479523f15f1fd367
    """
    Creates a mapping of cvs to curve tangent weight values.
    While all cvs are required, only the cvs with non-zero weights will be returned.
    Args:
        cvs(list): A list of cvs, these are used for the return value.
        t(float): A parameter value.
        degree(int): The curve dimensions.
        knots(list): A list of knot values.
        normalize(bool): When true, the curve is parameter is normalized from 0-1
    Returns:
        list: A list of control point, weight pairs.
    """

    curve_setup = deBoorSetup(cvs=cvs, t=t, degree=degree, knots=knots, normalize=normalize)
    knots = curve_setup[0]
    segment = curve_setup[1]
    t = curve_setup[2]
    periodic = curve_setup[3]

    # Convert cvs into hash-able indices
    _cvs = cvs
    cvs = [i for i in range(len(cvs))]

    # In order to find the tangent we need to find points on a lower degree curve
    degree = degree - 1
    weights = deBoorWeights(cvs=cvs, t=t, span=segment, degree=degree, knots=knots)

    # Take the lower order weights and match them to our actual cvs
    cvWeights = []
    for j in range(0, degree + 1):
        weight = weights[j]
        cv0 = j + segment - degree
        cv1 = j + segment - degree - 1
        alpha = weight * (degree + 1) / (knots[j + segment + 1] - knots[j + segment - degree])
        cvWeights.append([cvs[cv0], alpha])
        cvWeights.append([cvs[cv1], -alpha])

    return [[_cvs[index], weight] for index, weight in cvWeights]


def pointOnSurfaceWeights(cvs, u, v, uKnots=None, vKnots=None, degree=3):
    # Algorithm and code originally from Cole O'Brien
    # https://coleobrien.medium.com/matrix-splines-in-maya-ec17f3b3741
    # https://gist.github.com/obriencole11/354e6db8a55738cb479523f15f1fd367
    """
    Creates a mapping of cvs to surface point weight values.
    Args:
        cvs(list): A list of cv rows, these are used for the return value.
        u(float): The u parameter value on the curve.
        v(float): The v parameter value on the curve.
        uKnots(list, optional): A list of knot integers along u.
        vKnots(list, optional): A list of knot integers along v.
        degree(int, optional): The degree of the curve. Minimum is 2.
    Returns:
        list: A list of control point, weight pairs.
    """
    matrixWeightRows = [pointOnCurveWeights(row, u, degree, uKnots) for row in cvs]
    matrixWeightColumns = pointOnCurveWeights([i for i in range(len(matrixWeightRows))], v, degree, vKnots)
    surfaceMatrixWeights = []
    for index, weight in matrixWeightColumns:
        matrixWeights = matrixWeightRows[index]
        surfaceMatrixWeights.extend([[m, (w * weight)] for m, w in matrixWeights])

    return surfaceMatrixWeights


def tangentUOnSurfaceWeights(cvs, u, v, uKnots=None, vKnots=None, degree=3):
    # Algorithm and code originally from Cole O'Brien
    # https://coleobrien.medium.com/matrix-splines-in-maya-ec17f3b3741
    # https://gist.github.com/obriencole11/354e6db8a55738cb479523f15f1fd367
    """
    Creates a mapping of cvs to surface tangent weight values along the u axis.
    Args:
        cvs(list): A list of cv rows, these are used for the return value.
        u(float): The u parameter value on the curve.
        v(float): The v parameter value on the curve.
        uKnots(list, optional): A list of knot integers along u.
        vKnots(list, optional): A list of knot integers along v.
        degree(int, optional): The degree of the curve. Minimum is 2.
    Returns:
        list: A list of control point, weight pairs.
    """

    matrixWeightRows = [pointOnCurveWeights(row, u, degree, uKnots) for row in cvs]
    matrixWeightColumns = tangentOnCurveWeights([i for i in range(len(matrixWeightRows))], v, degree, vKnots)
    surfaceMatrixWeights = []
    for index, weight in matrixWeightColumns:
        matrixWeights = matrixWeightRows[index]
        surfaceMatrixWeights.extend([[m, (w * weight)] for m, w in matrixWeights])

    return surfaceMatrixWeights


def tangentVOnSurfaceWeights(cvs, u, v, uKnots=None, vKnots=None, degree=3):
    # Algorithm and code originally from Cole O'Brien
    # https://coleobrien.medium.com/matrix-splines-in-maya-ec17f3b3741
    # https://gist.github.com/obriencole11/354e6db8a55738cb479523f15f1fd367
    """
    Creates a mapping of cvs to surface tangent weight values along the v axis.
    Args:
        cvs(list): A list of cv rows, these are used for the return value.
        u(float): The u parameter value on the curve.
        v(float): The v parameter value on the curve.
        uKnots(list, optional): A list of knot integers along u.
        vKnots(list, optional): A list of knot integers along v.
        degree(int, optional): The degree of the curve. Minimum is 2.
    Returns:
        list: A list of control point, weight pairs.
    """
    # Re-order the cvs
    rowCount = len(cvs)
    columnCount = len(cvs[0])
    reorderedCvs = [[cvs[row][col] for row in range(rowCount)] for col in range(columnCount)]
    return tangentUOnSurfaceWeights(reorderedCvs, v, u, uKnots=vKnots, vKnots=uKnots, degree=degree)


def getPointOnSpline(cv_positions: list[Vector3], t: float, degree: int = 3, knots: list[float] = None) -> Vector3:
    position: Vector3 = Vector3()
    for control_point, weight in pointOnCurveWeights(cvs=cv_positions, t=t, degree=degree, knots=knots):
        position += control_point * weight
    return position


def getTangentOnSpline(cv_positions: list[Vector3], t: float, degree: int = 3, knots: list[float] = None) -> Vector3:
    tangent: Vector3 = Vector3()
    for control_point, weight in tangentOnCurveWeights(cvs=cv_positions, t=t, degree=degree, knots=knots):
        tangent += control_point * weight
    return tangent


def getPointsOnSpline(
    cv_positions: list[Vector3],
    number_of_points: int,
    degree: int = 3,
    knots: list[float] = None,
    sample_points: int = 128,
) -> list[float]:
    """
    Takes curve CV positions and returns the parameter of evenly spaced points along the curve.
    Args:
        cv_positions: list of vectors containing XYZ of the CV positions.
        segments: Number of point positions along the curve.
        degree: Degree of the spline CVs
        sample_points: The number of points to sample along the curve to find even arc-length segments. More points will be more accurate/evenly spaced.
    Returns:
        list: List of the parameter values of the picked points along the curve.
    """
    samples: list[Vector3] = []
    for i in range(sample_points):
        parameter: float = i * (1 / (sample_points - 1))
        sample_pos: Vector3 = getPointOnSpline(cv_positions=cv_positions, t=parameter, degree=degree, knots=knots)
        samples.append(sample_pos)

    arc_lengths: list[float] = []
    c_length: float = 0
    prev_sample: Vector3 = None
    for index, sample in enumerate(samples):
        if not prev_sample:
            prev_sample = sample
        distance: float = (sample - prev_sample).length()
        c_length += distance
        arc_lengths.append(c_length)
        prev_sample = sample

    total_length: float = arc_lengths[len(arc_lengths) - 1]
    point_parameters: list[float] = []
    for i in range(number_of_points):
        u: float = i / (number_of_points - 1)
        mapped_t: float = u
        target_length: float = u * total_length

        # Binary search to find the first point equal or greater than the target length
        low: int = 0
        high: int = len(arc_lengths) - 1
        index: int = 0
        while low < high:
            index = low + (high - low) // 2
            if arc_lengths[index] < target_length:
                low = index + 1
            else:
                high = index

        # Step back by one
        if arc_lengths[index] > target_length:
            index -= 1
        length_before: float = arc_lengths[index]

        # If the sample is exactly our target point return it, if it's the last, return 1, otherwise interpolate between the closest samples
        if length_before == target_length:
            mapped_t = index / len(arc_lengths)
        elif i == number_of_points - 1:
            mapped_t = 1
        else:
            sample_distance = arc_lengths[index + 1] - arc_lengths[index]
            sample_fraction = (
                target_length - length_before
            ) / sample_distance  # How far we are along the current segment
            mapped_t = (index + sample_fraction) / len(arc_lengths)

        point_parameters.append(mapped_t)

    return point_parameters


def curveToMatrixSpline(curve: str, segments: int) -> tuple[list[str], list[str]]:
    """
    Takes a curve shape and returns the attributes of the offset_parent_matrix for each segment.
    Args:
        curve: The curve transform.
        segments: Number of matrices to pin to the curve.
    Returns:
        tuple: Tuple of the curve CV matrix attributes, and of the output matrix attributes.
    """
    # Retrieve the surface shape and ensure it is a NURBS surface.
    curve_shapes = cmds.listRelatives(curve, shapes=True) or []
    if not curve_shapes:
        cmds.error(f"No shape node found for {curve_shapes}")
    curve_shape = curve_shapes[0]

    if cmds.nodeType(curve_shape) != "nurbsCurve":
        cmds.error(f"Node {curve_shape} is not a nurbsCurve.")

    periodic: bool = cmds.getAttr(f"{curve}.form") == 2
    spans: int = cmds.getAttr(f"{curve}.spans")
    degree: int = cmds.getAttr(f"{curve}.degree")
    num_cvs: int = spans + degree
    cv_list = [f"{curve}.cv[{i}]" for i in range(num_cvs)]
    position_vectors: list[Vector3] = getCvs(curve_shape=curve_shape)
    positions = [(position.x, position.y, position.z) for position in position_vectors]
    knots: list[float] = getKnots(curve_shape)
    print(knots)
    cv_matrices: list[str] = []

    for i in range(num_cvs):
        # Create Transform for CV and move it to the position of the CV on the curve
        cv_transform = cmds.polySphere(name=f"{curve}_CV{i}")[0]
        cmds.setAttr(
            f"{cv_transform}.translate",
            positions[i][0],
            positions[i][1],
            positions[i][2],
        )

        # Remove scale and shear from matrix since they will interfere with the linear interpolation of the basis vectors (causing flipping)
        pick_matrix = cmds.createNode("pickMatrix", name=f"{cv_transform}_PickMatrix")
        cmds.connectAttr(f"{cv_transform}.matrix", f"{pick_matrix}.inputMatrix")
        cmds.setAttr(f"{pick_matrix}.useShear", 0)
        cmds.setAttr(f"{pick_matrix}.useScale", 0)

        # Add nodes to connect individual values from the matrix, I don't know why maya makes us do this instead of just connecting directly
        deconstruct_matrix_attribute = f"{pick_matrix}.outputMatrix"
        row1 = cmds.createNode("rowFromMatrix", name=f"{cv_transform}_row1")
        cmds.connectAttr(deconstruct_matrix_attribute, f"{row1}.matrix")
        cmds.setAttr(f"{row1}.input", 0)
        row2 = cmds.createNode("rowFromMatrix", name=f"{cv_transform}_row2")
        cmds.connectAttr(deconstruct_matrix_attribute, f"{row2}.matrix")
        cmds.setAttr(f"{row2}.input", 1)
        row3 = cmds.createNode("rowFromMatrix", name=f"{cv_transform}_row3")
        cmds.setAttr(f"{row3}.input", 2)
        cmds.connectAttr(deconstruct_matrix_attribute, f"{row3}.matrix")
        row4 = cmds.createNode("rowFromMatrix", name=f"{cv_transform}_row4")
        cmds.connectAttr(deconstruct_matrix_attribute, f"{row4}.matrix")
        cmds.setAttr(f"{row4}.input", 3)

        # Rebuild the matrix but encode the scale into the empty values in the matrix (this needs to be extracted after the weighted matrix sum)
        cv_matrix = cmds.createNode("fourByFourMatrix", name=f"{cv_transform}_CvMatrix")
        cmds.connectAttr(f"{row1}.outputX", f"{cv_matrix}.in00")
        cmds.connectAttr(f"{row1}.outputY", f"{cv_matrix}.in01")
        cmds.connectAttr(f"{row1}.outputZ", f"{cv_matrix}.in02")
        cmds.connectAttr(f"{cv_transform}.scaleX", f"{cv_matrix}.in03")

        cmds.connectAttr(f"{row2}.outputX", f"{cv_matrix}.in10")
        cmds.connectAttr(f"{row2}.outputY", f"{cv_matrix}.in11")
        cmds.connectAttr(f"{row2}.outputZ", f"{cv_matrix}.in12")
        cmds.connectAttr(f"{cv_transform}.scaleY", f"{cv_matrix}.in13")

        cmds.connectAttr(f"{row3}.outputX", f"{cv_matrix}.in20")
        cmds.connectAttr(f"{row3}.outputY", f"{cv_matrix}.in21")
        cmds.connectAttr(f"{row3}.outputZ", f"{cv_matrix}.in22")
        cmds.connectAttr(f"{cv_transform}.scaleZ", f"{cv_matrix}.in23")

        cmds.connectAttr(f"{row4}.outputX", f"{cv_matrix}.in30")
        cmds.connectAttr(f"{row4}.outputY", f"{cv_matrix}.in31")
        cmds.connectAttr(f"{row4}.outputZ", f"{cv_matrix}.in32")
        cmds.connectAttr(f"{row4}.outputW", f"{cv_matrix}.in33")

        cv_matrices.append(f"{cv_matrix}.output")

    segment_parameters = getPointsOnSpline(cv_positions=position_vectors, number_of_points=segments, degree=degree)
    for i in range(segments):
        segment_name = f"{curve}_matrixSpline_Segment{i + 1}"

        parameter = segment_parameters[i]

        # Create node that blends the matrices based on the calculated DeBoor weights.
        blended_matrix = cmds.createNode("wtAddMatrix", name=f"{segment_name}_BaseMatrix")
        point_weights = pointOnCurveWeights(cvs=cv_matrices, t=parameter, degree=degree, knots=knots)
        for index, point_weight in enumerate(point_weights):
            cmds.setAttr(f"{blended_matrix}.wtMatrix[{index}].weightIn", point_weight[1])
            cmds.connectAttr(f"{point_weight[0]}", f"{blended_matrix}.wtMatrix[{index}].matrixIn")

        blended_tangent_matrix = cmds.createNode("wtAddMatrix", name=f"{segment_name}_TangentMatrix")
        tangent_weights = tangentOnCurveWeights(cvs=cv_matrices, t=parameter, degree=degree, knots=knots)
        for index, tangent_weight in enumerate(tangent_weights):
            cmds.setAttr(
                f"{blended_tangent_matrix}.wtMatrix[{index}].weightIn",
                tangent_weight[1],
            )
            cmds.connectAttr(
                f"{tangent_weight[0]}",
                f"{blended_tangent_matrix}.wtMatrix[{index}].matrixIn",
            )

        tangent_vector = cmds.createNode("pointMatrixMult", name=f"{blended_tangent_matrix}_TangentVector")
        cmds.connectAttr(f"{blended_tangent_matrix}.matrixSum", f"{tangent_vector}.inMatrix")

        # Create nodes to access the values of the blended matrix node.
        deconstruct_matrix_attribute = f"{blended_matrix}.matrixSum"
        blended_matrix_row1 = cmds.createNode("rowFromMatrix", name=f"{blended_matrix}_row1")
        cmds.setAttr(f"{blended_matrix_row1}.input", 0)
        cmds.connectAttr(deconstruct_matrix_attribute, f"{blended_matrix_row1}.matrix")
        blended_matrix_row2 = cmds.createNode("rowFromMatrix", name=f"{blended_matrix}_row2")
        cmds.connectAttr(deconstruct_matrix_attribute, f"{blended_matrix_row2}.matrix")
        cmds.setAttr(f"{blended_matrix_row2}.input", 1)
        blended_matrix_row3 = cmds.createNode("rowFromMatrix", name=f"{blended_matrix}_row3")
        cmds.connectAttr(deconstruct_matrix_attribute, f"{blended_matrix_row3}.matrix")
        cmds.setAttr(f"{blended_matrix_row3}.input", 2)
        blended_matrix_row4 = cmds.createNode("rowFromMatrix", name=f"{blended_matrix}_row4")
        cmds.connectAttr(deconstruct_matrix_attribute, f"{blended_matrix_row4}.matrix")
        cmds.setAttr(f"{blended_matrix_row4}.input", 3)

        # Create aim matrix node.
        aim_matrix = cmds.createNode("aimMatrix", name=f"{segment_name}_AimMatrix")
        cmds.setAttr(f"{aim_matrix}.primaryMode", 2)
        cmds.setAttr(f"{aim_matrix}.primaryInputAxis", 0, 1, 0)
        cmds.setAttr(f"{aim_matrix}.secondaryMode", 2)
        cmds.setAttr(f"{aim_matrix}.secondaryInputAxis", 0, 0, 1)
        cmds.connectAttr(f"{tangent_vector}.output", f"{aim_matrix}.primary.primaryTargetVector")
        cmds.connectAttr(
            f"{blended_matrix_row3}.outputX",
            f"{aim_matrix}.secondary.secondaryTargetVectorX",
        )
        cmds.connectAttr(
            f"{blended_matrix_row3}.outputY",
            f"{aim_matrix}.secondary.secondaryTargetVectorY",
        )
        cmds.connectAttr(
            f"{blended_matrix_row3}.outputZ",
            f"{aim_matrix}.secondary.secondaryTargetVectorZ",
        )

        # Create nodes to access the values of the aim matrix node.
        deconstruct_matrix_attribute = f"{aim_matrix}.outputMatrix"
        aim_matrix_row1 = cmds.createNode("rowFromMatrix", name=f"{aim_matrix}_row1")
        cmds.connectAttr(deconstruct_matrix_attribute, f"{aim_matrix_row1}.matrix")
        cmds.setAttr(f"{aim_matrix_row1}.input", 0)
        aim_matrix_row2 = cmds.createNode("rowFromMatrix", name=f"{aim_matrix}_row2")
        cmds.connectAttr(deconstruct_matrix_attribute, f"{aim_matrix_row2}.matrix")
        cmds.setAttr(f"{aim_matrix_row2}.input", 1)
        aim_matrix_row3 = cmds.createNode("rowFromMatrix", name=f"{aim_matrix}_row3")
        cmds.connectAttr(deconstruct_matrix_attribute, f"{aim_matrix_row3}.matrix")
        cmds.setAttr(f"{aim_matrix_row3}.input", 2)

        # Get tangent vector magnitude
        tangent_vector_length = cmds.createNode("length", name=f"{segment_name}_tangentVectorLength")
        cmds.connectAttr(f"{tangent_vector}.output", f"{tangent_vector_length}.input")
        tangent_vector_length_scaled = cmds.createNode(
            "multDoubleLinear", name=f"{segment_name}_tangentVectorLengthScaled"
        )
        cmds.connectAttr(f"{tangent_vector_length}.output", f"{tangent_vector_length_scaled}.input1")
        tangent_sample = cmds.pointOnCurve(curve_shape, tangent=True, parameter=parameter * spans)
        cmds.setAttr(
            f"{tangent_vector_length_scaled}.input2",
            1 / Vector3(tangent_sample[0], tangent_sample[1], tangent_sample[2]).length(),
        )

        # Create Nodes to re-apply scale
        x_scaled = cmds.createNode("multiplyDivide", name=f"{segment_name}_xScale")
        x_vector_attribute = f"{aim_matrix_row1}"
        x_scale_attribute = f"{blended_matrix_row1}.outputW"
        cmds.connectAttr(f"{x_vector_attribute}.outputX", f"{x_scaled}.input1X")
        cmds.connectAttr(f"{x_vector_attribute}.outputY", f"{x_scaled}.input1Y")
        cmds.connectAttr(f"{x_vector_attribute}.outputZ", f"{x_scaled}.input1Z")

        cmds.connectAttr(x_scale_attribute, f"{x_scaled}.input2X")
        cmds.connectAttr(x_scale_attribute, f"{x_scaled}.input2Y")
        cmds.connectAttr(x_scale_attribute, f"{x_scaled}.input2Z")

        y_scaled = cmds.createNode("multiplyDivide", name=f"{segment_name}_yScale")
        y_vector_attribute = f"{aim_matrix_row2}"
        y_scale_attribute = f"{tangent_vector_length_scaled}.output"
        cmds.connectAttr(f"{y_vector_attribute}.outputX", f"{y_scaled}.input1X")
        cmds.connectAttr(f"{y_vector_attribute}.outputY", f"{y_scaled}.input1Y")
        cmds.connectAttr(f"{y_vector_attribute}.outputZ", f"{y_scaled}.input1Z")

        cmds.connectAttr(y_scale_attribute, f"{y_scaled}.input2X")
        cmds.connectAttr(y_scale_attribute, f"{y_scaled}.input2Y")
        cmds.connectAttr(y_scale_attribute, f"{y_scaled}.input2Z")

        z_scaled = cmds.createNode("multiplyDivide", name=f"{segment_name}_zScale")
        z_vector_attribute = f"{aim_matrix_row3}"
        z_scale_attribute = f"{blended_matrix_row3}.outputW"
        cmds.connectAttr(f"{z_vector_attribute}.outputX", f"{z_scaled}.input1X")
        cmds.connectAttr(f"{z_vector_attribute}.outputY", f"{z_scaled}.input1Y")
        cmds.connectAttr(f"{z_vector_attribute}.outputZ", f"{z_scaled}.input1Z")

        cmds.connectAttr(z_scale_attribute, f"{z_scaled}.input2X")
        cmds.connectAttr(z_scale_attribute, f"{z_scaled}.input2Y")
        cmds.connectAttr(z_scale_attribute, f"{z_scaled}.input2Z")

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

        cmds.connectAttr(f"{blended_matrix_row4}.outputX", f"{output_matrix}.in30")
        cmds.connectAttr(f"{blended_matrix_row4}.outputY", f"{output_matrix}.in31")
        cmds.connectAttr(f"{blended_matrix_row4}.outputZ", f"{output_matrix}.in32")

        segment_transform = cmds.polyCube(name=segment_name)[0]
        cmds.connectAttr(f"{output_matrix}.output", f"{segment_transform}.offsetParentMatrix")
