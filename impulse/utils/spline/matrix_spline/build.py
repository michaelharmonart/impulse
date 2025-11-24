import maya.cmds as cmds

from impulse.structs.transform import Vector3
from impulse.utils.control import Control, ControlShape, connect_control, make_control
from impulse.utils.spline.math import (
    generate_knots,
    resample,
)
from impulse.utils.spline.matrix_spline.core import (
    MatrixSpline,
    bound_curve_from_matrix_spline,
    pin_to_matrix_spline,
)
from impulse.utils.spline.maya_query import get_cv_weights, get_cvs, get_knots
from impulse.utils.transform import clean_parent, matrix_constraint


def matrix_spline_from_curve(
    curve: str,
    segments: int,
    transforms_to_pin: list[str] = [],
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
    primary_axis: tuple[int, int, int] | None = (0, 1, 0),
    secondary_axis: tuple[int, int, int] | None = (0, 0, 1),
    twist: bool = True,
    align_tangent: bool = True,
    create_curve: bool = False,
) -> MatrixSpline:
    """
    Takes a curve shape and creates a matrix spline with controls and deformation joints.
    Args:
        curve: The curve transform.
        segments: Number of matrices to pin to the curve.
        transforms_to_pin: These transforms will be constrained to the spline as a normal segment.
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
        create_curve (bool, optional): If True, creates and binds a NURBS curve to the MatrixSpline.
            This is useful for arc length calculation.
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
        cv_transforms=cv_transforms,
        degree=degree,
        knots=knots,
        periodic=periodic,
        name=name,
    )
    if create_curve:
        matrix_spline.curve = bound_curve_from_matrix_spline(
            matrix_spline=matrix_spline, curve_parent=container_group
        )

    segment_parameters: list[float] = resample(
        cv_positions=cv_positions,
        number_of_points=segments,
        degree=degree,
        knots=knots,
        periodic=periodic,
        padded=padded,
        arc_length=arc_length,
        normalize_parameter=False,
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
        clean_parent(segment_transform, def_group)
        connect_control(control=segment_ctl, driven_name=segment_transform)
        pin_to_matrix_spline(
            matrix_spline=matrix_spline,
            pinned_transform=segment_ctl.offset_transform,
            parameter=parameter,
            stretch=stretch,
            primary_axis=primary_axis,
            secondary_axis=secondary_axis,
            normalize_parameter=False,
            twist=twist,
            align_tangent=align_tangent,
        )
    if transforms_to_pin is not None:
        pin_parameters: list[float] = resample(
            cv_positions=cv_positions,
            number_of_points=len(transforms_to_pin),
            degree=degree,
            knots=knots,
            periodic=periodic,
            padded=padded,
            arc_length=arc_length,
            normalize_parameter=False,
        )
        for index, transform in enumerate(transforms_to_pin):
            segment_pin: str = cmds.group(
                name=f"{transform}_Pin", empty=True, parent=container_group
            )
            matrix_constraint(segment_pin, transform, keep_offset=False)
            pin_to_matrix_spline(
                matrix_spline=matrix_spline,
                pinned_transform=segment_pin,
                parameter=pin_parameters[index],
                stretch=stretch,
                primary_axis=primary_axis,
                secondary_axis=secondary_axis,
                normalize_parameter=False,
                twist=twist,
                align_tangent=align_tangent,
            )
    return matrix_spline


def matrix_spline_from_transforms(
    transforms: list[str],
    segments: int,
    transforms_to_pin: list[str] | None = [],
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
    primary_axis: tuple[int, int, int] | None = (0, 1, 0),
    secondary_axis: tuple[int, int, int] | None = (0, 0, 1),
    twist: bool = True,
    align_tangent: bool = True,
    create_curve: bool = False,
) -> MatrixSpline:
    """
    Takes a set of transforms (cvs) and creates a matrix spline with controls and deformation joints.
    Args:
        transforms: The transforms (CVS) that will drive the spline.
        transforms_to_pin: These transforms will be constrained to the spline as a normal segment.
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
        create_curve (bool, optional): If True, creates and binds a NURBS curve to the MatrixSpline.
            This is useful for arc length calculation.
    Returns:
        matrix_spline: The resulting matrix spline.
    """
    num_cvs: int = len(transforms)
    if not knots:
        if periodic:
            knots = generate_knots(num_cvs + degree, degree=degree, periodic=True)
        else:
            knots = generate_knots(num_cvs, degree=degree)
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
        cv_transforms=cv_transforms,
        degree=degree,
        periodic=periodic,
        name=name,
        knots=knots,
    )
    if create_curve:
        matrix_spline.curve = bound_curve_from_matrix_spline(
            matrix_spline=matrix_spline, curve_parent=container_group
        )

    extended_cv_positions: list[Vector3] = list(cv_positions)

    if periodic:
        for i in range(degree):
            extended_cv_positions.append(cv_positions[i])

    segment_parameters: list[float] = resample(
        cv_positions=extended_cv_positions,
        number_of_points=segments,
        degree=degree,
        knots=knots,
        periodic=periodic,
        padded=padded,
        arc_length=arc_length,
        normalize_parameter=False,
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
            clean_parent(segment_transform, prev_segment)
        else:
            clean_parent(segment_transform, def_group)

        prev_segment = segment_transform
        connect_control(control=segment_ctl, driven_name=segment_transform)
        pin_to_matrix_spline(
            matrix_spline=matrix_spline,
            pinned_transform=segment_ctl.offset_transform,
            parameter=parameter,
            stretch=stretch,
            primary_axis=primary_axis,
            secondary_axis=secondary_axis,
            normalize_parameter=False,
            twist=twist,
            align_tangent=align_tangent,
        )
    if transforms_to_pin is not None:
        pin_parameters: list[float] = resample(
            cv_positions=extended_cv_positions,
            number_of_points=len(transforms_to_pin),
            degree=degree,
            knots=knots,
            periodic=periodic,
            padded=padded,
            arc_length=arc_length,
            normalize_parameter=False,
        )
        for index, transform in enumerate(transforms_to_pin):
            segment_pin: str = cmds.group(
                name=f"{transform}_Pin", empty=True, parent=container_group
            )
            matrix_constraint(segment_pin, transform, keep_offset=False)
            pin_to_matrix_spline(
                matrix_spline=matrix_spline,
                pinned_transform=segment_pin,
                parameter=pin_parameters[index],
                stretch=stretch,
                primary_axis=primary_axis,
                secondary_axis=secondary_axis,
                normalize_parameter=False,
                twist=twist,
                align_tangent=align_tangent,
            )
    return matrix_spline
