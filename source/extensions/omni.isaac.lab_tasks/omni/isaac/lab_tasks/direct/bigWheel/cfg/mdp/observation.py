from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import ObservationTermCfg
from omni.isaac.lab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def euler_angles(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    euler_angle = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    euler_angle = torch.stack(euler_angle, dim=1)
    return euler_angle


def base_lin_vel_x(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def inner_theta_vel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    theta_vel = torch.zeros(env.num_envs, 2, device=env.device)
    theta_vel[:, 0] = (
        asset.data.joint_vel[:, 0].squeeze() + asset.data.root_ang_vel_b[:, 1].squeeze()
    )
    theta_vel[:, 1] = (
        asset.data.joint_vel[:, 1].squeeze() + asset.data.root_ang_vel_b[:, 1].squeeze()
    )
    return theta_vel


def inner_theta(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    euler_angle = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    theta = torch.zeros(env.num_envs, 2, device=env.device)
    theta[:, 0] = asset.data.joint_pos[:, 0].squeeze() + euler_angle[1].squeeze()
    theta[:, 1] = asset.data.joint_pos[:, 1].squeeze() + euler_angle[1].squeeze()
    return theta
