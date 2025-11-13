from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import maya.cmds as cmds
from maya.api.OpenMaya import MMatrix, MPoint, MQuaternion, MSpace, MTransformationMatrix

from impulse.utils.transform import get_local_matrix


@dataclass
class PoseDriver:
    """Represents a driver transform used for pose interpolation.

    Args:
        transform (str): Name of the Maya transform node used as the driver.
        twist_axis (Literal["x", "y", "z"], optional): Axis used for twist computation. Defaults to "y".
        euler_twist (bool, optional): Whether to use Euler twist instead of quaternion. Defaults to False.
    """

    transform: str
    twist_axis: Literal["x", "y", "z"] | int = "y"
    euler_twist: bool = False

    def __post_init__(self) -> None:
        axis_map: dict[str, int] = {"x": 0, "y": 1, "z": 2}
        if isinstance(self.twist_axis, int):
            self.twist_axis_num = self.twist_axis
        else:
            self.twist_axis_num = axis_map[self.twist_axis]


class PoseType(Enum):
    """Defines the type of pose for interpolation."""

    SWING_AND_TWIST = 0
    SWING_ONLY = 1
    TWIST_ONLY = 2


@dataclass
class Pose:
    """Stores pose data for a pose interpolator.

    Args:
        name (str): Name of the pose.
        rotations (list[MQuaternion], optional): List of pose rotations. Defaults to identity.
        translations (list[MPoint], optional): List of pose translations. Defaults to origin.
        matrices (list[MMatrix] | None, optional): Optional list of transformation matrices,
            when set these will override the pose rotations and pose translations.
        independent (bool, optional): Whether the pose is independent. Defaults to False.
        rotation_falloff (float, optional): Angular falloff in degrees. Defaults to 180.
        translation_falloff (float, optional): Translation falloff distance. Defaults to 0.
        gaussian_falloff (float, optional): Gaussian smoothing factor. Defaults to 1.
        enabled (bool, optional): Whether the pose is active. Defaults to True.
    """

    name: str
    rotations: list[MQuaternion] | None = field(default_factory=lambda: [MQuaternion.kIdentity])
    translations: list[MPoint] | None = field(default_factory=lambda: [MPoint.kOrigin])
    matrices: list[MMatrix] | None = None
    independent: bool = False
    rotation_falloff: float = 180
    translation_falloff: float = 0.001
    pose_type: PoseType | int = PoseType.SWING_ONLY
    gaussian_falloff: float = 1
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.matrices is not None:
            transform_matrices: list[MTransformationMatrix] = [
                MTransformationMatrix(matrix) for matrix in self.matrices
            ]
            self.rotations = [matrix.rotation(asQuaternion=True) for matrix in transform_matrices]
            self.translations = [MPoint(matrix.translation(MSpace.kTransform)) for matrix in transform_matrices]
        if not isinstance(self.pose_type, PoseType):
            self.pose_type = PoseType(self.pose_type)
        self.index: int | None = None


@dataclass
class PoseInterpolator:
    """Creates and manages a Maya poseInterpolator node with connected drivers.

    Args:
        name (str): Name for the new poseInterpolator node.
        drivers (list[PoseDriver]): List of driver definitions.
        poses (list[Pose], optional): List of pose definitions.
        parent (str | None, optional): Optional parent node for the interpolator.
        regularization (float, optional): Regularization strength for RBF solving.
        gaussian_interpolation (bool, optional): Use Gaussian instead of linear interpolation.
        allow_negative_weights (bool, optional): Pose weights may be solved to negative values or values greater than 1.0.
            This is the default because it provides the smoothest interpolation.
            However, if you want to prevent having negative values, turn off this option.
            This may cause sharp steps in the output because negative values will be clamped.
        output_smoothing (float, optional): Remaps the output weights to a smooth interpolation.
            This is useful with Linear interpolations because it give you the stability of a linear interpolation
            with the smoothness of a Gaussian falloff.
        enable_translation (bool, optional): When this is on, the orientation of the joint is used when solving the pose.
        enable_rotation (bool, optional): When this is on, the translation of the joint is used when solving the pose.
    """

    name: str
    drivers: list[PoseDriver]
    poses: list[Pose] | None = None
    parent: str | None = None
    regularization: float = 0
    gaussian_interpolation: bool = False
    allow_negative_weights: bool = True
    output_smoothing: float = 0
    enable_translation: bool = False
    enable_rotation: bool = True

    def __post_init__(self) -> None:
        pose_interpolator_node: str = cmds.createNode("poseInterpolator", name=f"{self.name}Shape")
        transform_node: str = cmds.listRelatives(pose_interpolator_node, parent=True)[0]
        self.transform: str = cmds.rename(transform_node, self.name)
        self.pose_interpolator = cmds.listRelatives(self.transform, children=True, shapes=True)[0]
        if self.parent is not None:
            cmds.parent(self.transform, self.parent)

        drivers_to_add: list[PoseDriver] = self.drivers.copy()
        self.drivers = []
        for driver in drivers_to_add:
            self.add_driver(driver)
        self.neutral_pose: Pose = self._get_neutral_pose()
        self.pose_indices: set[int] = set()
        if self.poses is None:
            self.poses: list[Pose] = []
            self.add_pose(self.neutral_pose)
        else:
            poses_to_add: list[Pose] = self.poses.copy()
            self.poses: list[Pose] = []
            self.add_pose(self.neutral_pose)
            for pose in poses_to_add:
                self.add_pose(pose)

        cmds.setAttr(f"{self.pose_interpolator}.regularization", self.regularization)
        cmds.setAttr(
            f"{self.pose_interpolator}.interpolation", 1 if self.gaussian_interpolation else 0
        )
        cmds.setAttr(f"{self.pose_interpolator}.outputSmoothing", self.output_smoothing)
        cmds.setAttr(
            f"{self.pose_interpolator}.allowNegativeWeights",
            1 if self.allow_negative_weights else 0,
        )
        cmds.setAttr(f"{self.pose_interpolator}.enableRotation", 1 if self.enable_rotation else 0)
        cmds.setAttr(
            f"{self.pose_interpolator}.enableTranslation", 1 if self.enable_translation else 0
        )

    def add_driver(self, driver: PoseDriver) -> None:
        next_driver_index: int = len(self.drivers)
        driver_attr = f"{self.pose_interpolator}.driver[{next_driver_index}]"
        cmds.connectAttr(f"{driver.transform}.matrix", f"{driver_attr}.driverMatrix")
        cmds.connectAttr(f"{driver.transform}.rotateAxis", f"{driver_attr}.driverRotateAxis")
        cmds.connectAttr(f"{driver.transform}.rotateOrder", f"{driver_attr}.driverRotateOrder")
        cmds.setAttr(f"{driver_attr}.driverTwistAxis", driver.twist_axis_num)
        cmds.setAttr(f"{driver_attr}.driverEulerTwist", 1 if driver.euler_twist else 0)
        if cmds.nodeType(driver.transform) == "joint":
            cmds.connectAttr(f"{driver.transform}.jointOrient", f"{driver_attr}.driverOrient")
        self.drivers.append(driver)

    def _get_neutral_pose(self) -> Pose:
        matrices: list[MMatrix] = [get_local_matrix(driver.transform) for driver in self.drivers]
        return Pose(name="neutral", matrices=matrices, gaussian_falloff=0.5)

    def _next_free_pose_index(self) -> int:
        index = 0
        while index in self.pose_indices:
            index += 1
        return index

    def add_pose(self, pose: Pose) -> None:
        next_pose_index: int = self._next_free_pose_index()
        pose_attr = f"{self.pose_interpolator}.pose[{next_pose_index}]"
        cmds.setAttr(f"{pose_attr}.poseName", pose.name, type="string")
        for index, rotation in enumerate(pose.rotations):
            cmds.setAttr(
                f"{pose_attr}.poseRotation[{index}]",
                (rotation.x, rotation.y, rotation.z, rotation.w),
                type="doubleArray",
            )
        for index, translation in enumerate(pose.translations):
            cmds.setAttr(
                f"{pose_attr}.poseTranslation[{index}]",
                (translation.x, translation.y, translation.z),
                type="doubleArray",
            )
        cmds.setAttr(f"{pose_attr}.isIndependent", 1 if pose.independent else 0)
        cmds.setAttr(f"{pose_attr}.poseRotationFalloff", pose.rotation_falloff)
        cmds.setAttr(f"{pose_attr}.poseTranslationFalloff", pose.translation_falloff)
        cmds.setAttr(f"{pose_attr}.poseType", pose.pose_type.value)
        cmds.setAttr(f"{pose_attr}.poseFalloff", pose.gaussian_falloff)
        cmds.setAttr(f"{pose_attr}.isEnabled", 1 if pose.enabled else 0)
        pose.index = next_pose_index
        self.pose_indices.add(next_pose_index)
        self.poses.append(pose)
