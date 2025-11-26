from impulse.utils.spline import closest_point_on_matrix_spline
import maya.cmds as cmds

from impulse.structs.transform import Vector3
from impulse.utils.control import Control, ControlShape, connect_control, make_control
from impulse.utils.spline.math import generate_knots
from impulse.utils.spline.matrix_spline import (
    MatrixSpline,
    bound_curve_from_matrix_spline,
    pin_to_matrix_spline,
)
from impulse.utils.transform import clean_parent, match_transform, matrix_constraint


def create_eyelid_spline(
    cv_transforms: list[str],
    segment_guides: list[int],
    degree: int = 3,
    knots: list[str] | None = None,
    padded: bool = False,
    name: str | None = None,
    control_size: float = 0.1,
    control_shape: ControlShape = ControlShape.CUBE,
    control_height: float = 1,
    parent: str | None = None,
    spline_group: str | None = None,
    ctl_group: str | None = None,
    def_group: str | None = None,
    create_curve: bool = False,
) -> MatrixSpline:
    """
    Takes a set of transforms (cvs) and creates a matrix spline with controls and deformation joints.
    Args:
        cv_transforms: The transforms (CVS) that will drive the spline.
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
        arc_length: When True, the parameters for the spline will be even according to arc length.
        spline_group: The container group for all the generated subcontrols and joints.
        ctl_group: The container for the generated sub-controls.
        def_group: The container for the generated deformation joints.
        def_chain: When true, each of the generated deformation joints will be parented as a chain.
        create_curve (bool, optional): If True, creates and binds a NURBS curve to the MatrixSpline.
            This is useful for arc length calculation.
    Returns:
        matrix_spline: The resulting matrix spline.
    """
    num_cvs: int = len(cv_transforms)
    if not knots:
        knots = generate_knots(num_cvs, degree=degree)
    cv_positions: list[Vector3] = [
        Vector3(*cmds.xform(transform, query=True, worldSpace=True, translation=True))
        for transform in cv_transforms
    ]
    guide_positions: list[Vector3] = [
        cmds.xform(transform, query=True, worldSpace=True, translation=True)
        for transform in segment_guides
    ]

    if not spline_group:
        if not parent:
            if cmds.listRelatives(cv_transforms[0], parent=True):
                curve_parent: str = cmds.listRelatives(cv_transforms[0], parent=True)[0]
            else:
                curve_parent: str = None
            if curve_parent:
                container_group: str = cmds.group(
                    empty=True, parent=curve_parent, name=f"{name}_GRP"
                )
            else:
                container_group: str = cmds.group(empty=True, name=f"{name}_GRP")
        else:
            container_group: str = cmds.group(empty=True, parent=parent, name=f"{name}_GRP")
    else:
        container_group = spline_group

    if not ctl_group:
        ctl_group: str = cmds.group(empty=True, parent=container_group, name=f"{name}_CTLS")
    mch_group: str = cmds.group(empty=True, parent=container_group, name=f"{name}_MCH")
    if not def_group:
        def_group: str = cmds.group(empty=True, parent=container_group, name=f"{name}_DEF")

    # Create CV Transforms
    driven_cv_transforms: list[str] = []
    for i, transform in enumerate(cv_transforms):
        driven_cv_transform: str = cmds.group(name=f"{name}_CV{i}", empty=True)
        matrix_constraint(transform, driven_cv_transform, keep_offset=False)
        driven_cv_transforms.append(driven_cv_transform)
        cmds.parent(driven_cv_transform, mch_group)

    matrix_spline: MatrixSpline = MatrixSpline(
        cv_transforms=driven_cv_transforms,
        degree=degree,
        name=name,
        knots=knots,
    )
    if create_curve:
        matrix_spline.curve = bound_curve_from_matrix_spline(
            matrix_spline=matrix_spline, curve_parent=container_group
        )

    segment_parameters: list[float] = [
        closest_point_on_matrix_spline(matrix_spline, position) for position in guide_positions
    ]

    for i, segment_guide in enumerate(segment_guides):
        segment_name = f"{matrix_spline.name}_Segment{i + 1:02d}"
        parameter = segment_parameters[i]

        segment_ctl: Control = make_control(
            name=segment_name,
            control_shape=control_shape,
            size=control_size,
            dimensions=(1, 1 * control_height, 1),
            parent=ctl_group,
        )
        segment_transform: str = cmds.joint(name=segment_name, scaleCompensate=False)
        clean_parent(segment_transform, def_group)

        connect_control(control=segment_ctl, driven_name=segment_transform)
        pin_to_matrix_spline(
            matrix_spline=matrix_spline,
            pinned_transform=segment_ctl.offset_transform,
            parameter=parameter,
            normalize_parameter=False,
            align_tangent=False,
        )
        matrix_spline.pinned_drivers.append(segment_transform)

    return matrix_spline


def create_eyelid_drivers(
    name: str, parent: str, cv_transforms: list[str], loop_guides: list[str] | None = None, degree: int = 3, aim_vector: tuple[float, float, float] = (0,1,0),
):
    part_name = name
    group = cmds.group(empty=True, name=f"{part_name}_GRP", parent=parent)
    def_group = cmds.group(empty=True, name=f"{part_name}_DEF", parent=group)
    mch_group = cmds.group(empty=True, name=f"{part_name}_MCH", parent=group)

    eyelid_spline: MatrixSpline = create_eyelid_spline(
        cv_transforms=cv_transforms,
        name=f"{name}_Spline",
        segment_guides=loop_guides,
        degree=degree,
        parent=mch_group,
    )

    for i, lid_driver in enumerate(eyelid_spline.pinned_drivers):
        driver = cmds.group(empty=True, name=f"{part_name}_Driver{i:02d}", parent=mch_group)
        joint: str = cmds.joint(name=f"{part_name}_{i:02d}_DEF")
        cmds.parent(joint, def_group, relative=True)
        cmds.aimConstraint(lid_driver, driver, maintainOffset=False, worldUpType="none", aimVector=aim_vector)
        match_transform(joint, lid_driver)
        matrix_constraint(driver, joint, keep_offset=True)
        
