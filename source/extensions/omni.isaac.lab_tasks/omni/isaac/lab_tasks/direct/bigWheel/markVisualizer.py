# import math
import torch
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.isaac.lab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)
import omni.isaac.lab.sim as sim_utils


class MarkVisualizer:
    def __init__(self) -> None:
        # add markers
        goal_vel_visualizer_cfg: VisualizationMarkersCfg = (
            GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
        )
        current_vel_visualizer_cfg: VisualizationMarkersCfg = (
            BLUE_ARROW_X_MARKER_CFG.replace(
                prim_path="/Visuals/Command/velocity_current"
            )
        )
        current_height_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/height_current",
            markers={
                "marker": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0)
                    ),
                )
            },
        )
        goal_height_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/height_goal",
            markers={
                "marker": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0)
                    ),
                )
            },
        )
        goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
        current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
        self.goal_marker = VisualizationMarkers(goal_vel_visualizer_cfg)
        self.current_marker = VisualizationMarkers(current_vel_visualizer_cfg)
        self.goal_height_marker = VisualizationMarkers(goal_height_visualizer_cfg)
        self.current_height_marker = VisualizationMarkers(current_height_visualizer_cfg)

        self.MAX_LIN_VEL_X = 10.0

    def update_mark(
        self,
        command: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_rot: torch.Tensor,
        robot_vel: torch.Tensor,
        num_envs: int,
        _device: torch.device,
    ):
        goal_marker_translations = robot_pos + torch.tensor(
            [0, 0, 0.5], device=_device
        ).repeat(num_envs, 1)
        current_marker_translations = robot_pos + torch.tensor(
            [0, 0, 0.4], device=_device
        ).repeat(num_envs, 1)
        height_goal_transelations = robot_pos + torch.tensor(
            [-0.1, 0.05, 0], device=_device
        ).repeat(num_envs, 1)
        height_current_transelations = robot_pos + torch.tensor(
            [-0.1, -0.05, 0], device=_device
        ).repeat(num_envs, 1)
        marker_orientations = robot_rot
        goal_scale = torch.tensor([0.1, 0.1, 0.1], device=_device).repeat(num_envs, 1)
        current_scale = torch.tensor([0.1, 0.1, 0.1], device=_device).repeat(num_envs, 1)
        goal_scale[:, 0] = (command[:, 0] / self.MAX_LIN_VEL_X) * 0.5
        current_scale[:, 0] = (robot_vel / self.MAX_LIN_VEL_X) * 0.5
        height_goal_transelations[:, 2] = command[:, 2]
        height_current_transelations[:, 2] = robot_pos[:, 2]
        self.goal_marker.visualize(
            goal_marker_translations, marker_orientations, goal_scale
        )
        self.current_marker.visualize(
            current_marker_translations, marker_orientations, current_scale
        )
        self.goal_height_marker.visualize(height_goal_transelations)
        self.current_height_marker.visualize(height_current_transelations)
