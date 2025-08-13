"""
Functions for working with splines.

"""

from typing import Any

import maya.cmds as cmds
import numpy as np

from impulse.structs.transform import Vector3 as Vector3
from impulse.utils.control import Control, ControlShape, connect_control, make_control
from impulse.utils.transform import matrix_constraint


def generate_knots(count: int, degree: int = 3) -> list[float]:
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
    knots += [
        count - degree for i in range(degree)
    ]  # put degree number of the last knot value at the end
    return [float(knot) for knot in knots]


def get_knots(curve_shape: str) -> list[float]:
    # Refer to https://openusd.org/dev/api/class_usd_geom_nurbs_curves.html#details
    # The above only works with uniform knots, so this is generalized to higher order and non-uniform knots
    # based on info found here https://developer.rhino3d.com/guides/opennurbs/periodic-curves-and-surfaces/
    """
    Gets the knot vector for a given curve shape.
    Args:
        curve_shape(str): Name of curve shape node.
    Returns:
        list: A list of knot values. (aka knot vector)
    """
    curve_info = cmds.createNode("curveInfo", name="temp_curveInfo")
    cmds.connectAttr(f"{curve_shape}.worldSpace", f"{curve_info}.inputCurve")
    knots: list[float] = cmds.getAttr(f"{curve_info}.knots[*]")
    cmds.delete(curve_info)
    degree: int = cmds.getAttr(f"{curve_shape}.degree")
    periodic_indices = (degree * 2) - 1
    knots.insert(0, 0)
    knots.append(0)
    if cmds.getAttr(f"{curve_shape}.form") == 2:
        knots[0] = knots[1] - (knots[-(periodic_indices - 1)] - knots[-(periodic_indices)])
        knots[-1] = knots[-2] + (knots[periodic_indices] - knots[periodic_indices - 1])
    else:
        knots[0] = knots[1]
        knots[-1] = knots[-2]

    return knots


def get_cvs(curve_shape: str) -> list[Vector3]:
    """
    Gets the positions of all CVs for a given curve shape.
    Args:
        curve_shape(str): Name of curve shape node.
    Returns:
        list: A list of CV positions as Vector3s
    """
    curve_info = cmds.createNode("curveInfo", name="temp_curveInfo")
    cmds.connectAttr(f"{curve_shape}.worldSpace", f"{curve_info}.inputCurve")
    cv_list: list[tuple[float, float, float]] = cmds.getAttr(f"{curve_info}.controlPoints[*]")
    cmds.delete(curve_info)
    position_list = [Vector3(position[0], position[1], position[2]) for position in cv_list]
    return position_list


def get_cv_weights(curve_shape: str) -> list[float]:
    """
    Gets the weights of all CVs for a given curve shape.
    Args:
        curve_shape(str): Name of curve shape node.
    Returns:
        list: A list of CV weight values.
    """
    curve_info = cmds.createNode("curveInfo", name="temp_curveInfo")
    cmds.connectAttr(f"{curve_shape}.worldSpace", f"{curve_info}.inputCurve")
    weights: list[float] = cmds.getAttr(f"{curve_info}.weights[*]")
    cmds.delete(curve_info)
    return weights


def is_periodic_knot_vector(knots: list[float], degree: int = 3) -> bool:
    # Based on this equation k[(degree-1)+i+1] - k[(degree-1)+i] = k[(cv_count-1)+i+1] - k[(cv_count)+i]
    # See https://developer.rhino3d.com/guides/opennurbs/periodic-curves-and-surfaces/
    # Although there is a typo in the above doc, k[(cv_count)+i] should be k[(cv_count - 1)+i]
    # Don't ask how long it took me to find that out
    cv_count = len(knots) - (degree + 1)
    num_knots = len(knots)
    for i in range(-degree + 1, degree):
        if (
            knots[(degree - 1) + i + 1] - knots[(degree - 1) + i]
            != knots[(cv_count - 1) + i + 1] - knots[(cv_count - 1) + i]
        ):
            return False
    return True


def deBoor_setup(
    cvs: list[Any],
    t: float,
    degree: int = 3,
    knots: list[float] | None = None,
    normalize: bool = True,
) -> tuple[list[float], int, float, bool]:
    # Algorithm and code originally from Cole O'Brien. Modified to support periodic splines.
    # https://coleobrien.medium.com/matrix-splines-in-maya-ec17f3b3741
    # https://gist.github.com/obriencole11/354e6db8a55738cb479523f15f1fd367
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

    knots = knots or generate_knots(len(cvs), degree)  # Defaults to even knot distribution
    if len(knots) != len(cvs) + order:
        raise ValueError(
            "Not enough knots provided. Curves with %s cvs must have a knot vector of length %s. "
            "Received a knot vector of length %s: %s. "
            "Total knot count must equal len(cvs) + degree + 1."
            % (len(cvs), len(cvs) + order, len(knots), knots)
        )

    # Determine if curve is periodic
    periodic: bool = is_periodic_knot_vector(knots=knots, degree=degree)

    # Optional normalization of t
    domain_start = knots[degree]
    domain_end = knots[-degree - 1]
    domain_range = domain_end - domain_start

    if normalize:
        t = (t * domain_range) + domain_start

    if periodic:
        t = ((t - domain_start) % domain_range) + domain_start  # Wrap t into valid domain

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


def deBoor_weights(
    cvs: list[Any],
    knots: list[float],
    t: float,
    span: int,
    degree: int = 3,
    cv_weights: dict[Any, float] | None = None,
) -> dict[Any, float]:
    # Algorithm and code originally from Cole O'Brien
    # https://coleobrien.medium.com/matrix-splines-in-maya-ec17f3b3741
    # https://gist.github.com/obriencole11/354e6db8a55738cb479523f15f1fd367
    """
    Extracts information needed for DeBoors Algorithm
    Args:
        cvs(list): A list of cvs, these are used for the return value.
        t(float): A parameter value.
        span(int): Span index (can be retrieved with deBoorSetup)
        degree(int): The curve dimensions.
        knots(list): A list of knot values.
        weights(dict): A dictionary of CV:Weight values.
    Returns:
        dict: Dictionary with cv: weight mappings
    """
    if cv_weights is None:
        cv_weights = {cv: 1 for cv in cvs}

    # Run a modified version of de Boors algorithm
    cvBases = [{cv: 1.0} for cv in cvs]  # initialize basis weights with a value of 1 for every cv
    for r in range(1, degree + 1):  # Loop once per degree
        for j in range(degree, r - 1, -1):  # Loop backwards from degree to r
            right = j + 1 + span - r
            left = j + span - degree
            alpha = (t - knots[left]) / (
                knots[right] - knots[left]
            )  # Alpha is how much influence comes from the left vs right cv

            weights = {}
            for cv, weight in cvBases[j].items():
                weights[cv] = weight * alpha

            for cv, weight in cvBases[j - 1].items():
                if cv in weights:
                    weights[cv] += weight * (1 - alpha)
                else:
                    weights[cv] = weight * (1 - alpha)

            cvBases[j] = weights
    finalBases = cvBases[degree]

    # Multiply each CVs basis function by it's weight
    # see: https://en.wikipedia.org/wiki/Non-uniform_rational_B-spline#General_form_of_a_NURBS_curve
    numerator: dict[Any, float] = {i: finalBases[i] * cv_weights[i] for i in finalBases}

    # Sum all of the weights to normalize them such that they all total to 1
    denominator: float = sum(numerator.values())
    if denominator == 0:
        raise ZeroDivisionError("Zero sum of total weight values, unable to normalize.")

    # Actually do the normalization
    rational_weights: dict[Any, float] = {i: numerator[i] / denominator for i in numerator}

    return rational_weights


def point_on_spline_weights(
    cvs: list[Any],
    t: float,
    degree: int = 3,
    knots: list[float] | None = None,
    weights: list[float] | None = None,
    normalize: bool = True,
) -> list[tuple[Any, float]]:
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
        weights(list): A list of CV weight values.
        normalize(bool): When true, the curve is parameter is normalized from 0-1
    Returns:
        list: A list of control point, weight pairs.
    """

    curve_setup = deBoor_setup(cvs=cvs, t=t, degree=degree, knots=knots, normalize=normalize)
    knots = curve_setup[0]
    segment = curve_setup[1]
    t = curve_setup[2]
    periodic = curve_setup[3]

    # Convert cvs into hash-able indices
    _cvs = cvs
    cvs: list[int] = [i for i in range(len(cvs))]
    if weights:
        cv_weights = {cvs[i]: weights[i] for i in range(len(cvs))}
    else:
        cv_weights = None

    # Filter out cvs we won't be using
    cvs = [cvs[j + segment - degree] for j in range(0, degree + 1)]

    # Run a modified version of de Boors algorithm
    cvWeights = deBoor_weights(
        cvs=cvs, t=t, span=segment, degree=degree, knots=knots, cv_weights=cv_weights
    )
    return [(_cvs[index], weight) for index, weight in cvWeights.items() if weight != 0.0]


def get_weights_along_spline(
    cvs: list[Any],
    parameters: list[float],
    degree: int = 3,
    knots: list[float] | None = None,
    sample_points: int = 128,
) -> list[list[tuple[Any, float]]]:
    """
    Evaluates B-spline basis weights for a given list of parameters.
    Faster than calling point_on_spline_weights in a loop as this function uses a
        lookup table and interpolation. Will be much faster when passing
        a large number of parameter values such as when splitting skin weights on a dense mesh.
    Args:
        cvs(list): A list of cvs, these are used for the return value.
        parameters(list): List of parameters.
        degree: Degree of the B-spline.
        knots(list): Knot vector of the B-spline.
        sample_points: Number of samples to take for Lookup Table Interpolation,
            more samples will be more accurate but slower. Default value of 128 should be plenty.

    Returns:
        A (len(parameters), n_basis) matrix of spline weights.
    """
    if not knots:
        knots = generate_knots(len(cvs), degree=degree)

    result: list[list[tuple[Any, float]]] = []
    # If we have less points than samples don't bother using a lookup table
    if len(parameters) <= sample_points:
        for parameter in parameters:
            weights: list[tuple[Any, float]] = point_on_spline_weights(
                cvs=cvs, t=parameter, degree=degree, knots=knots, normalize=False
            )
            result.append(weights)
        return result

    # Precompute lookup table
    parameter_array = np.array(parameters, dtype=float)
    min_t, max_t = min(parameters), max(parameters)
    t_range = max_t - min_t
    if t_range == 0:
        # All parameters are the same, just calculate the one weight
        weights = point_on_spline_weights(
            cvs=cvs, t=min_t, degree=degree, knots=knots, normalize=False
        )
        return [weights for _ in parameters]

    # Get evenly spaced points from the minimum to maximum t value
    sample_params = np.linspace(min_t, max_t, sample_points, dtype=float)
    lut_weights = np.zeros((sample_points, len(cvs)), dtype=float)
    for sample_index, sample_parameter in enumerate(sample_params):
        weights: list[tuple[Any, float]] = point_on_spline_weights(
            cvs=cvs, t=sample_parameter, degree=degree, knots=knots, normalize=False
        )
        weight_dict = {cv: w for cv, w in weights}
        # Take the weights and put them into the correct row in the array
        lut_weights[sample_index, :] = [weight_dict.get(cv, 0.0) for cv in cvs]

    # Map each parameter to LUT index positions
    normalized_positions = (parameter_array - min_t) / t_range * (sample_points - 1)
    lower_indices = np.floor(normalized_positions).astype(int)
    upper_indices = np.clip(lower_indices + 1, 0, sample_points - 1)
    interpolation_alphas = (normalized_positions - lower_indices)[:, None]

    # Interpolate weights for all parameters in bulk
    interpolated_weight_array = (1 - interpolation_alphas) * lut_weights[
        lower_indices, :
    ] + interpolation_alphas * lut_weights[upper_indices, :]

    # Reattach CV references to each interpolated weight row
    for weight_row in interpolated_weight_array:
        result.append(list(zip(cvs, weight_row.tolist())))
    return result


def tangent_on_spline_weights(
    cvs: list[Any],
    t: float,
    degree: int = 3,
    knots: list[float] | None = None,
    normalize: bool = True,
) -> list[tuple[Any, int]]:
    # Algorithm and code originally from Cole O'Brien
    # https://coleobrien.medium.com/matrix-splines-in-maya-ec17f3b3741
    # https://gist.github.com/obriencole11/354e6db8a55738cb479523f15f1fd367

    # This cannot be used for full NURBS, only B-Splines (NURBS where every CV has a weight of 1)
    # as the derivative of a full NURB Spline cannot be expressed as a weighted sum of point positions
    """
    Creates a mapping of cvs to curve tangent weight values.
    While all cvs are required, only the cvs with non-zero weights will be returned.
    Args:
        cvs(list): A list of cvs, these are used for the return value.
        t(float): A parameter value.
        degree(int): The curve dimensions.
        knots(list): A list of knot values.
        normalize(bool): When true, the curve parameter is normalized from 0-1
    Returns:
        list: A list of control point, weight pairs.
    """

    curve_setup = deBoor_setup(cvs=cvs, t=t, degree=degree, knots=knots, normalize=normalize)
    knots: list[float] = curve_setup[0]
    segment: int = curve_setup[1]
    t: float = curve_setup[2]
    periodic: bool = curve_setup[3]

    # Convert cvs into hash-able indices
    _cvs: list[Any] = cvs
    cvs: list[Any] = [i for i in range(len(cvs))]

    # In order to find the tangent we need to find points on a lower degree curve
    degree: int = degree - 1
    weights: dict[Any, float] = deBoor_weights(
        cvs=cvs, t=t, span=segment, degree=degree, knots=knots
    )

    # Take the lower order weights and match them to our actual cvs
    remapped_weights: list[Any] = []
    for j in range(0, degree + 1):
        weight: float = weights[j]
        cv0: int = j + segment - degree
        cv1: int = j + segment - degree - 1
        alpha: float = (
            weight * (degree + 1) / (knots[j + segment + 1] - knots[j + segment - degree])
        )
        remapped_weights.append([cvs[cv0], alpha])
        remapped_weights.append([cvs[cv1], -alpha])

    # Add weights of corresponding CVs and only return those that are > 0
    deduplicated_weights = {i: 0 for i in cvs}
    for item in remapped_weights:
        deduplicated_weights[item[0]] += item[1]
    deduplicated_weights = {key: value for key, value in deduplicated_weights.items() if value != 0}

    return [(_cvs[index], weight) for index, weight in deduplicated_weights.items()]


def point_on_surface_weights(cvs, u, v, uKnots=None, vKnots=None, degree=3):
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
    matrixWeightRows = [point_on_spline_weights(row, u, degree, uKnots) for row in cvs]
    matrixWeightColumns = point_on_spline_weights(
        [i for i in range(len(matrixWeightRows))], v, degree, vKnots
    )
    surfaceMatrixWeights = []
    for index, weight in matrixWeightColumns:
        matrixWeights = matrixWeightRows[index]
        surfaceMatrixWeights.extend([[m, (w * weight)] for m, w in matrixWeights])

    return surfaceMatrixWeights


def tangent_u_on_surface_weights(cvs, u, v, uKnots=None, vKnots=None, degree=3):
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

    matrixWeightRows = [point_on_spline_weights(row, u, degree, uKnots) for row in cvs]
    matrixWeightColumns = tangent_on_spline_weights(
        [i for i in range(len(matrixWeightRows))], v, degree, vKnots
    )
    surfaceMatrixWeights = []
    for index, weight in matrixWeightColumns:
        matrixWeights = matrixWeightRows[index]
        surfaceMatrixWeights.extend([[m, (w * weight)] for m, w in matrixWeights])

    return surfaceMatrixWeights


def tangent_v_on_surface_weights(cvs, u, v, uKnots=None, vKnots=None, degree=3):
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
    return tangent_u_on_surface_weights(
        reorderedCvs, v, u, uKnots=vKnots, vKnots=uKnots, degree=degree
    )


def get_point_on_spline(
    cv_positions: list[Vector3],
    t: float,
    degree: int = 3,
    knots: list[float] | None = None,
    weights: list[float] | None = None,
) -> Vector3:
    position: Vector3 = Vector3()
    for control_point, weight in point_on_spline_weights(
        cvs=cv_positions, t=t, degree=degree, knots=knots, weights=weights
    ):
        position += control_point * weight
    return position


def get_tangent_on_spline(
    cv_positions: list[Vector3], t: float, degree: int = 3, knots: list[float] | None = None
) -> Vector3:
    tangent: Vector3 = Vector3()
    for control_point, weight in tangent_on_spline_weights(
        cvs=cv_positions, t=t, degree=degree, knots=knots
    ):
        tangent += control_point * weight
    return tangent


def resample(
    cv_positions: list[Vector3],
    number_of_points: int,
    degree: int = 3,
    knots: list[float] | None = None,
    weights: list[float] | None = None,
    padded: bool = True,
    arc_length: bool = True,
    sample_points: int = 256,
) -> list[float]:
    """
    Takes curve CV positions and returns the parameter of evenly spaced points along the curve.
    Args:
        cv_positions: list of vectors containing XYZ of the CV positions.
        number_of_points: Number of point positions along the curve.
        degree: Degree of the spline CVs
        knots(list): A list of knot values.
        weights(list): A list of CV weight values.
        padded(bool): When True, the points are returned such that the end points have half a segment of spacing from the ends of the curve.
        arc_length(bool): When True, the points are returned with even spacing according to arc length.
        sample_points: The number of points to sample along the curve to find even arc-length segments. More points will be more accurate/evenly spaced.
    Returns:
        list: List of the parameter values of the picked points along the curve.
    """

    if not arc_length:
        point_parameters: list[float] = []
        for i in range(number_of_points):
            if padded:
                u = (i + 0.5) / number_of_points
            else:
                u = i / (number_of_points - 1)
            point_parameters.append(u)
        return point_parameters

    # Arc length based resampling
    samples: list[Vector3] = []
    for i in range(sample_points):
        parameter: float = i * (1 / (sample_points - 1))
        sample_pos: Vector3 = get_point_on_spline(
            cv_positions=cv_positions, t=parameter, degree=degree, knots=knots, weights=weights
        )
        samples.append(sample_pos)

    arc_lengths: list[float] = []
    c_length: float = 0
    prev_sample: Vector3 | None = None
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
        if padded:
            u: float = (i + 0.5) / (number_of_points)
        else:
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

        # If the sample is exactly our target point return it, if it's the last, return the end point, otherwise interpolate between the closest samples
        if length_before == target_length:
            mapped_t = index / len(arc_lengths)
        elif i == number_of_points - 1:
            if padded:
                mapped_t = 1 - (0.5 / number_of_points)
            else:
                mapped_t = 1
        else:
            sample_distance = arc_lengths[index + 1] - arc_lengths[index]
            sample_fraction = (
                target_length - length_before
            ) / sample_distance  # How far we are along the current segment
            mapped_t = (index + sample_fraction) / len(arc_lengths)

        point_parameters.append(mapped_t)

    return point_parameters


class MatrixSpline:
    def __init__(
        self,
        cv_transforms: list[str],
        degree: int = 3,
        knots: list[float] | None = None,
        periodic: bool = False,
        name: str | None = None,
    ) -> None:
        self.periodic = periodic
        self.degree = degree
        number_of_cvs: int = len(cv_transforms) + (periodic * degree)
        if knots:
            self.knots = knots
        else:
            self.knots = generate_knots(count=number_of_cvs, degree=degree)
        if name:
            self.name = name
        else:
            self.name = "MatrixSpline"

        cv_matrices: list[str] = []
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

            # Rebuild the matrix but encode the scale into the empty values in the matrix
            # (this needs to be extracted after the weighted matrix sum)
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

        # If the curve is periodic there are we need to re-add CVs that move together.
        if periodic:
            for i in range(degree):
                cv_matrices.append(cv_matrices[i])

        self.cv_matrices = cv_matrices


def pin_to_matrix_spline(
    matrix_spline: MatrixSpline, pinned_transform: str, parameter: float, stretch: bool = True
) -> None:
    """
    Pins a transform to a matrix spline at a given parameter along the curve.

    Args:
        matrix_spline: The matrix spline data object.
        pinned_transform: Transform to pin to the spline.
        parameter: Position along the spline (0–1).
        stretch: Whether to apply automatic scaling along the spline tangent.

    Returns:
        None
    """
    cv_matrices: list[str] = matrix_spline.cv_matrices
    degree: int = matrix_spline.degree
    knots: list[float] = matrix_spline.knots
    segment_name: str = pinned_transform

    # Create node that blends the matrices based on the calculated DeBoor weights.
    blended_matrix = cmds.createNode("wtAddMatrix", name=f"{segment_name}_BaseMatrix")
    point_weights = point_on_spline_weights(
        cvs=cv_matrices, t=parameter, degree=degree, knots=knots
    )
    for index, point_weight in enumerate(point_weights):
        cmds.setAttr(f"{blended_matrix}.wtMatrix[{index}].weightIn", point_weight[1])
        cmds.connectAttr(f"{point_weight[0]}", f"{blended_matrix}.wtMatrix[{index}].matrixIn")

    blended_tangent_matrix = cmds.createNode("wtAddMatrix", name=f"{segment_name}_TangentMatrix")
    tangent_weights = tangent_on_spline_weights(
        cvs=cv_matrices, t=parameter, degree=degree, knots=knots
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

    tangent_vector = cmds.createNode(
        "pointMatrixMult", name=f"{blended_tangent_matrix}_TangentVector"
    )
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

    if stretch:
        # Get tangent vector magnitude
        tangent_vector_length = cmds.createNode(
            "length", name=f"{segment_name}_tangentVectorLength"
        )
        cmds.connectAttr(f"{tangent_vector}.output", f"{tangent_vector_length}.input")
        tangent_vector_length_scaled = cmds.createNode(
            "multDoubleLinear", name=f"{segment_name}_tangentVectorLengthScaled"
        )
        cmds.connectAttr(
            f"{tangent_vector_length}.output", f"{tangent_vector_length_scaled}.input1"
        )
        tangent_sample = cmds.getAttr(f"{tangent_vector}.output")[0]
        tangent_length = Vector3(tangent_sample[0], tangent_sample[1], tangent_sample[2]).length()
        if tangent_length == 0:
            raise RuntimeError(
                f"{pinned_transform} had a tangent magnitude of 0 and wasn't able to be pinned with stretching enabled."
            )
        cmds.setAttr(
            f"{tangent_vector_length_scaled}.input2",
            1 / tangent_length,
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
    if stretch:
        y_scale_attribute = f"{tangent_vector_length_scaled}.output"
    else:
        y_scale_attribute = f"{blended_matrix_row2}.outputW"

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

    cmds.connectAttr(f"{output_matrix}.output", f"{pinned_transform}.offsetParentMatrix")


def matrix_spline_from_curve(
    curve: str,
    segments: int,
    padded: bool = True,
    name: str | None = None,
    control_size: float = 0.1,
    control_shape: ControlShape = ControlShape.SPHERE,
    tweak_control_size: float = 1,
    tweak_control_shape: ControlShape = ControlShape.CUBE,
    tweak_control_height: float = 1,
    parent: str | None = None,
    stretch: bool = True,
    arc_length: bool = True,
) -> MatrixSpline:
    """
    Takes a curve shape and creates a matrix spline with controls and deformation joints.
    Args:
        curve: The curve transform.
        segments: Number of matrices to pin to the curve.
        padded: When True, segments are sampled such that the end points have half a segment of spacing from the ends of the spline.
        name: Name of the matrix spline group to be created.
        control_size: Size of generated controls.
        control_shape: Shape of primary controls.
        tweak_control_size: Size multiplier for tweak controls
        tweak_control_shape: Shape of tweak controls.
        tweak_control_size: Height multiplier for generated tweak controls.
        parent: Parent for the newly created matrix spline group.
        stretch: Whether to apply automatic scaling along the spline tangent.
        arc_length: When True, the parameters for the spline will be even according to arc length.

    Returns:
        matrix_spline: The resulting matrix spline.
    """
    # Retrieve the shape and ensure it is a NURBS curve.
    curve_shapes = cmds.listRelatives(curve, shapes=True) or []
    if not curve_shapes:
        cmds.error(f"No shape node found for {curve_shapes}")
    curve_shape = curve_shapes[0]

    if cmds.nodeType(curve_shape) != "nurbsCurve":
        cmds.error(f"Node {curve_shape} is not a nurbsCurve.")

    periodic: bool = cmds.getAttr(f"{curve}.form") == 2
    spans: int = cmds.getAttr(f"{curve}.spans")
    degree: int = cmds.getAttr(f"{curve}.degree")
    cv_positions: list[Vector3] = get_cvs(curve_shape)
    knots: list[float] = get_knots(curve_shape)
    weights: list[float] = get_cv_weights(curve_shape)
    if not name:
        name: str = curve

    if not parent:
        if cmds.listRelatives(curve, parent=True):
            curve_parent: str = cmds.listRelatives(curve, parent=True)[0]
        else:
            curve_parent: str = None
        if curve_parent:
            container_group: str = cmds.group(
                empty=True, parent=curve_parent, name=f"{name}_MatrixSpline_GRP"
            )
        else:
            container_group: str = cmds.group(empty=True, name=f"{name}_MatrixSpline_GRP")
    else:
        container_group: str = cmds.group(
            empty=True, parent=parent, name=f"{name}_MatrixSpline_GRP"
        )
    ctl_group: str = cmds.group(empty=True, parent=container_group, name=f"{name}_CTLS")
    mch_group: str = cmds.group(empty=True, parent=container_group, name=f"{name}_MCH")
    def_group: str = cmds.group(empty=True, parent=container_group, name=f"{name}_DEF")

    filtered_cv_positions: list[Vector3] = list(cv_positions)
    # If the curve is periodic there are duplicate CVs that move together. Remove them.
    if periodic:
        for i in range(degree):
            filtered_cv_positions.pop(-1)

    # Create CV Transforms
    cv_transforms: list[str] = []
    for i, cv_pos in enumerate(filtered_cv_positions):
        # Create Transform for CV and move it to the position of the CV on the curve
        control: Control = make_control(
            name=f"{curve}_CV{i}",
            position=(
                cv_pos.x,
                cv_pos.y,
                cv_pos.z,
            ),
            control_shape=control_shape,
            size=control_size,
        )
        cv_transform: str = cmds.group(name=f"{curve}_CV{i}", empty=True)
        connect_control(control=control, driven_name=cv_transform)
        cv_transforms.append(cv_transform)
        cmds.parent(cv_transform, mch_group)
        cmds.parent(control.offset_transform, ctl_group)
    matrix_spline: MatrixSpline = MatrixSpline(
        cv_transforms=cv_transforms, degree=degree, knots=knots, periodic=periodic, name=name
    )

    segment_parameters: list[float] = resample(
        cv_positions=cv_positions,
        number_of_points=segments,
        degree=degree,
        knots=knots,
        padded=padded,
        arc_length=arc_length,
    )

    for i in range(segments):
        segment_name = f"{matrix_spline.name}_matrixSpline_Segment{i + 1}"
        parameter = segment_parameters[i]

        segment_ctl: Control = make_control(
            name=segment_name,
            control_shape=tweak_control_shape,
            size=control_size * tweak_control_size,
            dimensions=(1, 1 * tweak_control_height, 1),
            parent=ctl_group,
        )
        segment_transform: str = cmds.joint(name=segment_name, scaleCompensate=False)
        cmds.parent(segment_transform, def_group, absolute=False)
        connect_control(control=segment_ctl, driven_name=segment_transform)
        pin_to_matrix_spline(
            matrix_spline=matrix_spline,
            pinned_transform=segment_ctl.offset_transform,
            parameter=parameter,
            stretch=stretch,
        )
    return matrix_spline


def matrix_spline_from_transforms(
    transforms: list[str],
    segments: int,
    periodic: bool = False,
    degree: int = 3,
    knots: list[str] | None = None,
    padded: bool = True,
    name: str | None = None,
    control_size: float = 0.1,
    control_shape: ControlShape = ControlShape.CUBE,
    control_height: float = 1,
    parent: str | None = None,
    stretch: bool = True,
    arc_length: bool = True,
    spline_group: str | None = None,
    ctl_group: str | None = None,
    def_group: str | None = None,
    def_chain: bool = False,
) -> MatrixSpline:
    """
    Takes a set of transforms (cvs) and creates a matrix spline with controls and deformation joints.
    Args:
        curve: The curve transform.
        segments: Number of matrices to pin to the curve.
        periodic: Whether the given transforms form a periodic curve or not (no need for repeated CVs)
        degree: Degree of the spline to be created.
        knots: The knot vector for the generated B-Spline.
        padded: When True, segments are sampled such that the end points have half a segment of spacing from the ends of the spline.
        name: Name of the matrix spline group to be created.
        control_size: Size of generated controls.
        control_shape: Shape of primary controls.
        tweak_control_size: Size multiplier for tweak controls
        tweak_control_shape: Shape of tweak controls.
        tweak_control_size: Height multiplier for generated tweak controls.
        parent: Parent for the newly created matrix spline group.
        stretch: Whether to apply automatic scaling along the spline tangent.
        arc_length: When True, the parameters for the spline will be even according to arc length.
        spline_group: The container group for all the generated subcontrols and joints.
        ctl_group: The container for the generated sub-controls.
        def_group: The container for the generated deformation joints.
        def_chain: When true, each of the generated deformation joints will be parented as a chain.
    Returns:
        matrix_spline: The resulting matrix spline.
    """

    cv_positions: list[Vector3] = []

    for transform in transforms:
        position = cmds.xform(transform, query=True, worldSpace=True, translation=True)
        cv_positions.append(Vector3(*position))
    if not spline_group:
        if not parent:
            if cmds.listRelatives(transforms[0], parent=True):
                curve_parent: str = cmds.listRelatives(transforms[0], parent=True)[0]
            else:
                curve_parent: str = None
            if curve_parent:
                container_group: str = cmds.group(
                    empty=True, parent=curve_parent, name=f"{name}_MatrixSpline_GRP"
                )
            else:
                container_group: str = cmds.group(empty=True, name=f"{name}_MatrixSpline_GRP")
        else:
            container_group: str = cmds.group(
                empty=True, parent=parent, name=f"{name}_MatrixSpline_GRP"
            )
    else:
        container_group = spline_group

    if not ctl_group:
        ctl_group: str = cmds.group(empty=True, parent=container_group, name=f"{name}_CTLS")
    mch_group: str = cmds.group(empty=True, parent=container_group, name=f"{name}_MCH")
    if not def_group:
        def_group: str = cmds.group(empty=True, parent=container_group, name=f"{name}_DEF")

    # Create CV Transforms
    cv_transforms: list[str] = []
    for i, transform in enumerate(transforms):
        cv_transform: str = cmds.group(name=f"{name}_CV{i}", empty=True)
        matrix_constraint(transform, cv_transform, keep_offset=False)
        cv_transforms.append(cv_transform)
        cmds.parent(cv_transform, mch_group)

    matrix_spline: MatrixSpline = MatrixSpline(
        cv_transforms=cv_transforms, degree=degree, periodic=periodic, name=name, knots=knots
    )

    segment_parameters: list[float] = resample(
        cv_positions=cv_positions,
        number_of_points=segments,
        degree=degree,
        knots=knots,
        padded=padded,
        arc_length=arc_length,
    )

    prev_segment: str | None = None
    for i in range(segments):
        segment_name = f"{matrix_spline.name}_matrixSpline_Segment{i + 1}"
        parameter = segment_parameters[i]

        segment_ctl: Control = make_control(
            name=segment_name,
            control_shape=control_shape,
            size=control_size,
            dimensions=(1, 1 * control_height, 1),
            parent=ctl_group,
        )
        segment_transform: str = cmds.joint(name=segment_name, scaleCompensate=False)
        if def_chain and prev_segment:
            cmds.parent(segment_transform, prev_segment, absolute=False)
        else:
            cmds.parent(segment_transform, def_group, absolute=False)

        prev_segment = segment_transform
        connect_control(control=segment_ctl, driven_name=segment_transform)
        pin_to_matrix_spline(
            matrix_spline=matrix_spline,
            pinned_transform=segment_ctl.offset_transform,
            parameter=parameter,
            stretch=stretch,
        )
    return matrix_spline
