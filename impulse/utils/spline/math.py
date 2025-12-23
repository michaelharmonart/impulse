from typing import TypeVar

import numpy as np

from impulse.structs.transform import Vector3

# CV can be anything: a Vector3, a transform name, etc.
CV = TypeVar("CV")

def generate_knots(count: int, degree: int = 3, periodic=False) -> list[float]:
    """
    Gets a default knot vector for a given number of cvs and degrees.
    Args:
        count(int): The number of cvs.
        degree(int): The curve degree.
        periodic: If true the knot vector will be for a periodic curve.
    Returns:
        list: A list of knot values. (aka knot vector)
    """

    if periodic:
        knots = [i for i in range(count + degree + 1)]
    else:
        clamp_start = [0] * degree
        clamp_end = [count - degree] * degree
        knots = clamp_start + [i for i in range(count - degree + 1)] + clamp_end

    return [float(knot) for knot in knots]


def is_periodic_knot_vector(knots: list[float], degree: int = 3) -> bool:
    # Based on this equation k[(degree-1)+i+1] - k[(degree-1)+i] = k[(cv_count-1)+i+1] - k[(cv_count)+i]
    # See https://developer.rhino3d.com/guides/opennurbs/periodic-curves-and-surfaces/
    # Although there is a typo in the above doc, k[(cv_count)+i] should be k[(cv_count - 1)+i]
    # Don't ask how long it took me to find that out
    cv_count = len(knots) - (degree + 1)
    for i in range(-degree + 1, degree):
        if (
            knots[(degree - 1) + i + 1] - knots[(degree - 1) + i]
            != knots[(cv_count - 1) + i + 1] - knots[(cv_count - 1) + i]
        ):
            return False
    return True


def deBoor_setup(
    cvs: list[CV],
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
    cvs: list[CV],
    knots: list[float],
    t: float,
    span: int,
    degree: int = 3,
    cv_weights: dict[CV, float] | None = None,
) -> dict[CV, float]:
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
    numerator = {i: finalBases[i] * cv_weights[i] for i in finalBases}

    # Sum all of the weights to normalize them such that they all total to 1
    denominator: float = sum(numerator.values())
    if denominator == 0:
        raise ZeroDivisionError("Zero sum of total weight values, unable to normalize.")

    # Actually do the normalization
    rational_weights = {i: numerator[i] / denominator for i in numerator}

    return rational_weights


def point_on_spline_weights(
    cvs: list[CV],
    t: float,
    degree: int = 3,
    knots: list[float] | None = None,
    weights: list[float] | None = None,
    normalize: bool = True,
) -> list[tuple[CV, float]]:
    # Algorithm and code originally from Cole O'Brien
    # https://coleobrien.medium.com/matrix-splines-in-maya-ec17f3b3741
    # https://gist.github.com/obriencole11/354e6db8a55738cb479523f15f1fd367
    """
    Creates a mapping of cvs to curve weight values on a spline curve.
    While all cvs are required, only the cvs with non-zero weights will be returned.
    This function is based on de Boor's algorithm for evaluating splines and has been modified to consolidate weights.
    Args:
        cvs: A list of cvs, these are used for the return value.
        t: A parameter value.
        degree: The curve dimensions.
        knots: A list of knot values.
        weights: A list of CV weight values.
        normalize: When true, the curve is parameter is normalized from 0-1
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
    cvs: list[CV],
    parameters: list[float],
    degree: int = 3,
    knots: list[float] | None = None,
    sample_points: int = 128,
) -> list[list[tuple[CV, float]]]:
    """
    Evaluates B-spline basis weights for a given list of parameters.
    Faster than calling point_on_spline_weights in a loop as this function uses a
    lookup table and interpolation. Will be much faster when passing a large number
    of parameter values such as when splitting skin weights on a dense mesh.

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

    result: list[list[tuple[CV, float]]] = []
    # If we have less points than samples don't bother using a lookup table
    if len(parameters) <= sample_points:
        for parameter in parameters:
            weights: list[tuple[CV, float]] = point_on_spline_weights(
                cvs=cvs, t=parameter, degree=degree, knots=knots, normalize=False
            )
            result.append(weights)
        return result

    # Precompute lookup table
    parameter_array = np.array(parameters, dtype=float)
    min_t, max_t = min(parameters), max(parameters)
    t_range: float = max_t - min_t
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
        weights: list[tuple[CV, float]] = point_on_spline_weights(
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
    cvs: list[CV],
    t: float,
    degree: int = 3,
    knots: list[float] | None = None,
    normalize: bool = True,
) -> list[tuple[CV, int]]:
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
    _cvs = cvs
    cvs = [i for i in range(len(cvs))]

    # In order to find the tangent we need to find points on a lower degree curve
    degree: int = degree - 1
    weights = deBoor_weights(
        cvs=cvs, t=t, span=segment, degree=degree, knots=knots
    )

    # Take the lower order weights and match them to our actual cvs
    remapped_weights: list[CV] = []
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


def get_point_on_spline(
    cv_positions: list[Vector3],
    t: float,
    degree: int = 3,
    knots: list[float] | None = None,
    weights: list[float] | None = None,
    normalize_parameter: bool = True,
) -> Vector3:
    position: Vector3 = Vector3()
    for control_point, weight in point_on_spline_weights(
        cvs=cv_positions,
        t=t,
        degree=degree,
        knots=knots,
        weights=weights,
        normalize=normalize_parameter,
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
    periodic=False,
    padded: bool = True,
    arc_length: bool = True,
    sample_points: int = 256,
    u_min: float | None = None,
    u_max: float | None = None,
    normalize_parameter: bool = True,
) -> list[float]:
    """
    Takes curve CV positions and returns the parameter of evenly spaced points along the curve.
    Args:
        cv_positions: list of vectors containing XYZ of the CV positions.
        number_of_points: Number of point positions along the curve.
        degree: Degree of the spline CVs
        knots(list): A list of knot values.
        weights(list): A list of CV weight values.
        periodic(bool): When True, the samples will be spaced evenly on the curve assuming it is periodic
        padded(bool): When True, the points are returned such that the end points have half a segment
            of spacing from the ends of the curve. Ignored if periodic.
        arc_length(bool): When True, the points are returned with even spacing according to arc length.
        sample_points: The number of points to sample along the curve to find even arc-length segments.
            More points will be more accurate/evenly spaced.
        u_min (float): The starting parameter value of the resampling range. Must be less than u_max.
        u_max (float): The ending parameter value of the resampling range. Must be greater than u_min.
    Returns:
        list: List of the parameter values of the picked points along the curve.
    """

    if not knots:
        knots: list[float] = generate_knots(
            count=len(cv_positions), degree=degree, periodic=periodic
        )

    if normalize_parameter:
        domain_start: float = 0.0
        domain_end: float = 1.0
    else:
        domain_start: float = knots[degree]
        domain_end: float = knots[-degree - 1]

    if not u_min:
        u_min: float = domain_start
    if not u_max:
        u_max: float = domain_end

    if not u_min < u_max:
        raise ValueError(
            f"The minimum U value ({u_min}) must be less than the maximum U value ({u_max})"
        )

    def get_normalized_u(index):
        if periodic:
            base_u = i / (number_of_points)
        else:
            if padded:
                base_u = (i + 0.5) / number_of_points
            else:
                base_u = i / (number_of_points - 1)
        return base_u

    def get_target_u(index: int) -> float:
        return u_min + (u_max - u_min) * get_normalized_u(index)

    if not arc_length:
        point_parameters: list[float] = []
        for i in range(number_of_points):
            u = get_target_u(i)
            point_parameters.append(u)
        return point_parameters

    # Arc length based resampling
    if sample_points < 2:
        raise ValueError("sample_points must be >= 2")

    sample_params: list[float] = [
        u_min + (u_max - u_min) * (i / (sample_points - 1)) for i in range(sample_points)
    ]

    samples: list[Vector3] = [
        get_point_on_spline(
            cv_positions=cv_positions,
            t=param,
            degree=degree,
            knots=knots,
            weights=weights,
            normalize_parameter=normalize_parameter,
        )
        for param in sample_params
    ]

    # cumulative arc lengths (arc_lengths[0] will be 0.0)
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

    total_length: float = arc_lengths[-1]

    point_parameters: list[float] = []
    n_samples: int = len(arc_lengths)  # equal to sample_points

    for i in range(number_of_points):
        normalized_u = get_normalized_u(i)
        mapped_t: float = get_target_u(i)
        target_length: float = normalized_u * total_length

        # Binary search to find the first point equal or greater than the target length
        low: int = 0
        high: int = n_samples - 1
        index: int = 0
        while low < high:
            mid = (low + high) // 2
            if arc_lengths[mid] < target_length:
                low = mid + 1
            else:
                high = mid
        index = low  # smallest index where arc_lengths[index] >= target_length.

        prev_index: int = max(0, index - 1)
        next_index: int = index

        # If the sample is exactly our target point return it, if it's the last, return the end point, otherwise interpolate between the closest samples
        if arc_lengths[prev_index] == target_length:
            mapped_t: float = sample_params[next_index]
        elif i == number_of_points - 1:
            mapped_t: float = get_target_u(number_of_points)
        else:
            length_before: float = arc_lengths[prev_index]
            sample_distance: float = arc_lengths[next_index] - arc_lengths[prev_index]
            if sample_distance == 0.0:
                sample_fraction: float = 0.0
            else:
                sample_fraction = (
                    target_length - length_before
                ) / sample_distance  # How far we are along the current segment
            mapped_t = sample_params[prev_index] + sample_fraction * (
                sample_params[next_index] - sample_params[prev_index]
            )

        point_parameters.append(mapped_t)

    return point_parameters
