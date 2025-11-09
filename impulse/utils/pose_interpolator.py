from dataclasses import dataclass, field
from typing import Literal


import maya.cmds as cmds


@dataclass
class PoseDriver:
    transform: str
    twist_axis: Literal["x", "y", "z"] = "y"
    euler_twist: bool = False

    def __post_init__(self) -> None:
        axis_map: dict[str, int] = {"x": 0, "y": 1, "z": 2}
        self.twist_axis_num = axis_map[self.twist_axis]


@dataclass
class Pose:
    pose_name: str
    pose_rotation: list[tuple[float, float, float]] = field(default_factory=lambda: [(0, 0, 0, 1)])
    pose_translation: list[tuple[float, float, float]] = field(default_factory=lambda: [(0, 0, 0)])
    independent: bool = False
    rotation_falloff: float = 180
    translation_falloff: float = 0
    gaussian_falloff: float = 1
    enabled: bool = True


@dataclass
class PoseInterpolator:
    name: str
    drivers: list[PoseDriver]
    parent: str | None = None
    regularization: float = 0
    gaussian_interpolation: bool = False
    allow_negative_weights: bool = False
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
        self.drivers: list[str] = []
        for driver in self.drivers:
            self.add_driver(driver)

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
