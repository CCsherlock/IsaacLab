# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# import math
import torch
from collections.abc import Sequence

# import gymnasium as gym
from omni.isaac.lab_assets.bigWheel import BIGWHEEL_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass

# from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.sensors import ContactSensorCfg, ContactSensor
import omni.isaac.lab.utils.math as math_utils


@configclass
class BigWheelEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale_inner = 4.0  # [N]
    action_scale_outer = 5.0  # [N]
    action_space = 4  # 驱动数
    observation_space = 25  # 观测数
    state_space = 0
    episode_length_s = 10.0  # 最大存活时间
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # robot
    robot_cfg: ArticulationCfg = BIGWHEEL_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=False,
    )
    innerWheel_left_dof_name = "innerWheelLeft_joint"
    innerWheel_right_dof_name = "innerWheelRight_joint"
    outerWheel_left_dof_name = "outerWheelLeft_joint"
    outerWheel_right_dof_name = "outerWheelRight_joint"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=4.0, replicate_physics=True
    )

    # reward scales //TODO: Update reward scales
    lin_vel_reward_scale = 1.5  # 线速度奖励
    yaw_rate_reward_scale = 0.1  # yaw 转向速度奖励
    z_vel_reward_scale = -0.5  #  垂直速度奖励
    ang_vel_reward_scale = -0.5  # pitch roll角速度奖励
    joint_torque_reward_scale_inner = -2.5e-5  # 内轮关节扭矩奖励
    joint_accel_reward_scale_inner = -2.5e-7  # 内轮关节加速度奖励
    joint_torque_reward_scale_outer = -2.5e-5  # 外轮关节扭矩奖励
    joint_accel_reward_scale_outer = -2.5e-7  # 外轮关节加速度奖励
    action_rate_reward_scale_inner = -0.005  # 内轮动作速率奖励
    action_rate_reward_scale_outer = -0.005  # 外轮动作速率奖励
    flat_orientation_reward_scale = -1.0  # 平面方向奖励


class BigWheelEnv(DirectRLEnv):
    cfg: BigWheelEnvCfg
    LEFT = 0
    RIGHT = 1
    # 物理参数
    outerWheel_radius = 0.225  # 外轮半径
    chassis_width = 0.32  # 底盘宽度
    outerWheel_radio = 5  # 外轮减速比

    def __init__(self, cfg: BigWheelEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._innerWheel_left_dof_idx, _ = self.bigWheel.find_joints(
            self.cfg.innerWheel_left_dof_name
        )
        self._innerWheel_right_dof_idx, _ = self.bigWheel.find_joints(
            self.cfg.innerWheel_right_dof_name
        )
        self._outerWheel_left_dof_idx, _ = self.bigWheel.find_joints(
            self.cfg.outerWheel_left_dof_name
        )
        self._outerWheel_right_dof_idx, _ = self.bigWheel.find_joints(
            self.cfg.outerWheel_right_dof_name
        )

        self.actions = torch.zeros(
            self.num_envs,
            4,
            device=self.device,
        )
        self.previous_actions = torch.zeros(
            self.num_envs,
            4,
            device=self.device,
        )

        self.action_scale_inner = self.cfg.action_scale_inner
        self.action_scale_outer = self.cfg.action_scale_outer

        self.joint_pos = self.bigWheel.data.joint_pos
        self.joint_vel = self.bigWheel.data.joint_vel

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 2, device=self.device)
        self.theta = torch.zeros(self.num_envs, 2, device=self.device)
        self.pitch = torch.zeros(self.num_envs, 1, device=self.device)
        self.velocity_chassis = torch.zeros(
            self.num_envs, 1, device=self.device
        )  # chassis 速度 x方向
        self._base_id, _ = self._contact_sensor.find_bodies("body")

    def _setup_scene(self):
        self.bigWheel = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["bigWheel"] = self.bigWheel
        # add sensors
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions[:, :2] = self.action_scale_inner * actions.clone()[:, :2]
        self.actions[:, 2:] = self.action_scale_outer * actions.clone()[:, 2:]

    def _apply_action(self) -> None:
        # self.actions = torch.zeros_like(self.actions)
        self.bigWheel.set_joint_effort_target(self.actions)

    def _get_observations(self) -> dict:
        self.previous_actions = self.actions.clone()

        # 计算内轮抬升角
        euler_angles = math_utils.euler_xyz_from_quat(
            self.bigWheel.data.root_quat_w
        )  # 获取欧拉角元组
        self.pitch = euler_angles[1].squeeze()  # 提取第二列并去除多余维度
        self.theta[:, self.LEFT] = (
            self.bigWheel.data.joint_pos[:, self._innerWheel_left_dof_idx].squeeze()
            + self.pitch
        )
        self.theta[:, self.RIGHT] = (
            self.bigWheel.data.joint_pos[:, self._innerWheel_right_dof_idx].squeeze()
            + self.pitch
        )
        # 计算底盘速度 = (左外轮速度 + 右外轮速度) * 外轮半径 * 0.5
        self.velocity_chassis = (
            (
                self.bigWheel.data.joint_vel[
                    :, self._outerWheel_left_dof_idx
                ].unsqueeze(1)
                + self.bigWheel.data.joint_vel[
                    :, self._outerWheel_right_dof_idx
                ].unsqueeze(1)
            )
            * self.outerWheel_radius
            * 0.5
        )
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.bigWheel.data.root_lin_vel_b,  # 3 机体线速度
                    self.bigWheel.data.root_ang_vel_b,  # 3 机体角速度
                    self.bigWheel.data.projected_gravity_b,  # 3 重力投影
                    self.bigWheel.data.joint_pos,  # 4 关节位置
                    self.bigWheel.data.joint_vel,  # 4 关节速度
                    self.theta,  # 2 内轮抬升角
                    self._commands,  # 2 指令
                    self.actions,  # 4 动作
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking x direction
        lin_vel_error = torch.square(
            self._commands[:, 0] - self.bigWheel.data.root_lin_vel_b[:, 0]
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(
            self._commands[:, 1] - self.bigWheel.data.root_ang_vel_b[:, 2]
        )
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self.bigWheel.data.root_lin_vel_b[:, 2])
        # angular velocity x/y pitch/roll
        ang_vel_error = torch.sum(
            torch.square(self.bigWheel.data.root_ang_vel_b[:, :2]), dim=1
        )
        # joint torques
        joint_torques_inner = torch.sum(
            torch.square(self.bigWheel.data.applied_torque[:, :2]), dim=1
        )
        joint_torques_outer = torch.sum(
            torch.square(self.bigWheel.data.applied_torque[:, 2:]), dim=1
        )
        # joint acceleration
        joint_accel_inner = torch.sum(
            torch.square(self.bigWheel.data.joint_acc[:, :2]), dim=1
        )
        joint_accel_outer = torch.sum(
            torch.square(self.bigWheel.data.joint_acc[:, 2:]), dim=1
        )
        # action rate
        action_rate_inner = torch.sum(
            torch.square(self.actions[:, :2] - self.previous_actions[:, :2]), dim=1
        )
        action_rate_outer = torch.sum(
            torch.square(self.actions[:, 2:] - self.previous_actions[:, 2:]), dim=1
        )
        # flat orientation pitch/roll
        flat_orientation = torch.sum(
            torch.square(self.bigWheel.data.projected_gravity_b[:, :2]), dim=1
        )

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped
            * self.cfg.lin_vel_reward_scale
            * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped
            * self.cfg.yaw_rate_reward_scale
            * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error
            * self.cfg.ang_vel_reward_scale
            * self.step_dt,
            "dof_torques_inner_l2": joint_torques_inner
            * self.cfg.joint_torque_reward_scale_inner
            * self.step_dt,
            "dof_torques_outer_l2": joint_torques_outer
            * self.cfg.joint_torque_reward_scale_outer
            * self.step_dt,
            "dof_acc_inner_l2": joint_accel_inner
            * self.cfg.joint_accel_reward_scale_inner
            * self.step_dt,
            "dof_acc_outer_l2": joint_accel_outer
            * self.cfg.joint_accel_reward_scale_outer
            * self.step_dt,
            "action_rate_inner_l2": action_rate_inner
            * self.cfg.action_rate_reward_scale_inner
            * self.step_dt,
            "action_rate_outer_l2": action_rate_outer
            * self.cfg.action_rate_reward_scale_outer
            * self.step_dt,
            "flat_orientation_l2": flat_orientation
            * self.cfg.flat_orientation_reward_scale
            * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(
            torch.max(
                torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1
            )[0]
            > 1.0,
            dim=1,
        )
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.bigWheel._ALL_INDICES
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        # Sample new commands
        self._commands[env_ids, 0] = (
            torch.zeros_like(self._commands[env_ids, 0])
            .uniform_(2.0, 7.0) *  # 生成范围在 [5, 10] 的随机数
            (torch.randint(0, 2, self._commands[env_ids, 0].shape, device=self._commands.device) * 2 - 1)  # 随机生成正负号
        ) # x

        self._commands[env_ids, 1] = torch.zeros_like(
            self._commands[env_ids, 1]
        ).uniform_(-0.5, 0.5)  # w
        # Reset robot state
        joint_pos = self.bigWheel.data.default_joint_pos[env_ids]
        joint_vel = self.bigWheel.data.default_joint_vel[env_ids]
        default_root_state = self.bigWheel.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.bigWheel.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.bigWheel.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.bigWheel.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
