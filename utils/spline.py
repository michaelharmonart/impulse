"""
Functions for working with splines.

This is mostly the work of Cole O'Brien
https://coleobrien.medium.com/matrix-splines-in-maya-ec17f3b3741
https://gist.github.com/obriencole11/354e6db8a55738cb479523f15f1fd367
"""

import maya.cmds as cmds

def generateKnots(count: int, degree: int=3) -> list[float]:
    """
    Gets a default knot vector for a given number of cvs and degrees.
    Args:
        count(int): The number of cvs. 
        degree(int): The curve degree. 
    Returns:
        list: A list of knot values. (aka knot vector)
    """
    knots = [0 for i in range(degree)] + [i for i in range(count - degree + 1)] #put degree number of 0s at the beginning
    knots += [count - degree for i in range(degree)] #put degree number of the last knot value at the end
    return [float(knot) for knot in knots]

def pointOnCurveWeights(cvs: list, t: float, degree: int = 3, knots: list[float] = None):
    """
    Creates a mapping of cvs to curve weight values on a spline curve.
    While all cvs are required, only the cvs with non-zero weights will be returned.
    This function is based on de Boor's algorithm for evaluating splines and has been modified to consolidate weights.
    Args:
        cvs(list): A list of cvs, these are used for the return value.
        t(float): A parameter value. 
        degree(int): The curve dimensions. 
        knots(list): A list of knot values. 
    Returns:
        list: A list of control point, weight pairs.
    """

    order = degree + 1  # Our functions often use order instead of degree
    if len(cvs) <= degree:
        raise ValueError('Curves of degree %s require at least %s cvs' % (degree, degree + 1))

    knots = knots or generateKnots(len(cvs), degree)  # Defaults to even knot distribution
    if len(knots) != len(cvs) + order:
        raise ValueError('Not enough knots provided. Curves with %s cvs must have a knot vector of length %s. '
                             'Received a knot vector of length %s: %s. '
                             'Total knot count must equal len(cvs) + degree + 1.' % (len(cvs), len(cvs) + order,
                                                                                     len(knots), knots))

    # Convert cvs into hash-able indices
    _cvs = cvs
    cvs: list[int] = [i for i in range(len(cvs))]

    # Remap the t value to the range of knot values.
    min = knots[order] - 1
    max = knots[len(knots) - 1 - order] + 1
    t = (t * (max - min)) + min

    # Determine which segment the t lies in
    segment = degree
    for index, knot in enumerate(knots[order:len(knots) - order]): #slice the knot list starting at order
        if knot <= t:
            segment = index + order # Set segment to position in the knot list, taking into acount the order offset we took out earlier

    # Filter out cvs we won't be using
    cvs = [cvs[j + segment - degree] for j in range(0, degree + 1)]

    # Run a modified version of de Boors algorithm
    cvWeights = [{cv: 1.0} for cv in cvs] # initialize weights with a value of 1 for every cv
    for r in range(1, degree + 1): # Loop once per degree
        for j in range(degree, r - 1, -1): # Loop backwards from degree to r
            right = j + 1 + segment - r
            left = j + segment - degree
            alpha = (t - knots[left]) / (knots[right] - knots[left]) # Alpha is how much influence comes from the left vs right cv

            weights = {}
            for cv, weight in cvWeights[j].items():
                weights[cv] = weight * alpha

            for cv, weight in cvWeights[j - 1].iteritems():
                if cv in weights:
                    weights[cv] += weight * (1 - alpha)
                else:
                    weights[cv] = weight * (1 - alpha)

            cvWeights[j] = weights

    cvWeights = cvWeights[degree]
    return [[_cvs[index], weight] for index, weight in cvWeights.items()]


def tangentOnCurveWeights(cvs: list, t: float, degree:int = 3, knots: list[float]=None):
    """
    Creates a mapping of cvs to curve tangent weight values.
    While all cvs are required, only the cvs with non-zero weights will be returned.
    Args:
        cvs(list): A list of cvs, these are used for the return value.
        t(float): A parameter value. 
        degree(int): The curve dimensions. 
        knots(list): A list of knot values. 
    Returns:
        list: A list of control point, weight pairs.
    """

    order = degree + 1  # Our functions often use order instead of degree
    if len(cvs) <= degree:
        raise ValueError('Curves of degree %s require at least %s cvs' % (degree, degree + 1))

    knots = knots or generateKnots(len(cvs), degree)  # Defaults to even knot distribution
    if len(knots) != len(cvs) + order:
        raise ValueError('Not enough knots provided. Curves with %s cvs must have a knot vector of length %s. '
                             'Received a knot vector of length %s: %s. '
                             'Total knot count must equal len(cvs) + degree + 1.' % (len(cvs), len(cvs) + order,
                                                                                     len(knots), knots))

    # Remap the t value to the range of knot values.
    min = knots[order] - 1
    max = knots[len(knots) - 1 - order] + 1
    t = (t * (max - min)) + min

    # Determine which segment the t lies in
    segment = degree
    for index, knot in enumerate(knots[order:len(knots) - order]):
        if knot <= t:
            segment = index + order

    # Convert cvs into hash-able indices
    _cvs = cvs
    cvs = [i for i in range(len(cvs))]

    # In order to find the tangent we need to find points on a lower degree curve
    degree = degree - 1
    qWeights = [{cv: 1.0} for cv in range(0, degree + 1)]

    # Get the DeBoor weights for this lower degree curve
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            right = j + 1 + segment - r
            left = j + segment - degree
            alpha = (t - knots[left]) / (knots[right] - knots[left])

            weights = {}
            for cv, weight in qWeights[j].items():
                weights[cv] = weight * alpha

            for cv, weight in qWeights[j - 1].items():
                if cv in weights:
                    weights[cv] += weight * (1 - alpha)
                else:
                    weights[cv] = weight * (1 - alpha)

            qWeights[j] = weights
    weights = qWeights[degree]

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