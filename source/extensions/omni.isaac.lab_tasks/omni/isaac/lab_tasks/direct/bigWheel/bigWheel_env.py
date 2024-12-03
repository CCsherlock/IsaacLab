# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# import math
import torch
from collections.abc import Sequence
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.sim.spawners.from_files import (
    GroundPlaneCfg,  # noqa: F401
    spawn_ground_plane,  # noqa: F401
)
from omni.isaac.lab.devices.keyboard import Se3Keyboard

# from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.sensors import ContactSensor
import omni.isaac.lab.utils.math as math_utils
from cfg.bigWheel_env_cfg import (
    BigWheelCFlatEnvCfg,
    BigWheelCRoughEnvCfg,
    BigWheelEnvCfg,  # noqa: F401
)
from cfg.trainRewardsCfg import TrainRewards
from visualize.markVisualizer import MarkVisualizer


class BigWheelEnv(DirectRLEnv):
    cfg: BigWheelCFlatEnvCfg | BigWheelCRoughEnvCfg
    LEFT = 0
    RIGHT = 1
    PLAY = 0
    MAX_LIN_VEL_X = 10.0
    MAX_ANGLE_VEL_Z = 0.1
    MAX_HEIGHT = 0.4
    MIN_HEIGHT = 0.065
    # 物理参数
    outerWheel_radius = 0.225  # 外轮半径
    chassis_width = 0.32  # 底盘宽度
    outerWheel_radio = 5  # 外轮减速比

    def __init__(
        self,
        cfg: BigWheelCFlatEnvCfg | BigWheelCRoughEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
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
        self.actions = torch.zeros(self.num_envs, 4, device=self.device)
        self.previous_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self.action_scale_inner = self.cfg.action_scale_inner
        self.action_scale_outer = self.cfg.action_scale_outer
        self.joint_pos = self.bigWheel.data.joint_pos
        self.joint_vel = self.bigWheel.data.joint_vel
        # X linear velocity and yaw angular velocity commands hight
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.theta = torch.zeros(self.num_envs, 2, device=self.device)
        self.theta_vel = torch.zeros(self.num_envs, 2, device=self.device)
        self.pitch = torch.zeros(self.num_envs, 1, device=self.device)
        self.euler_angle = torch.zeros(self.num_envs, 3, device=self.device)
        self.velocity_chassis = torch.zeros(self.num_envs, 1, device=self.device)
        self._base_id, _ = self._contact_sensor.find_bodies("body")
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                # "track_velocity_coupling_reward",
                # "track_height_reward",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_inner_l2",
                "dof_torques_outer_l2",
                "dof_acc_inner_l2",
                "dof_acc_outer_l2",
                "action_rate_inner_l2",
                "action_rate_outer_l2",
                "flat_orientation_l2",
            ]
        }

    def _setup_scene(self):
        self.bigWheel = Articulation(self.cfg.robot)
        # add ground plane
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
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

        self.keyboard_controller = Se3Keyboard(
            pos_sensitivity=0.05, rot_sensitivity=0.1
        )
        # 添加指令UI
        self.markVisualizer = MarkVisualizer()
        # 获取奖励配置
        self.rwd = TrainRewards()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions[:, :2] = self.action_scale_inner * actions.clone()[:, :2]
        self.actions[:, 2:] = self.action_scale_outer * actions.clone()[:, 2:]
        if self.PLAY:
            delta_pose, gripper_action = self.keyboard_controller.advance()
            self._commands[:, 0] += delta_pose[0] * 0.5
            self._commands[:, 1] += delta_pose[1] * 0.5
            if gripper_action:
                self._commands[:, 2] = 0.065
            else:
                self._commands[:, 2] = self.MAX_HEIGHT
            print(f"commands: {self._commands[0]}")
            # print(f"state: {self.bigWheel.data.root_vel_w[:,0], self.bigWheel.data.root_ang_vel_b[:,2]},self.bigWheel.data.root_pos_w[:,2]")
        #! 添加指令UI
        self.markVisualizer.update_mark(
            self._commands,
            self.bigWheel.data.root_pos_w,
            self.bigWheel.data.root_quat_w,
            self.bigWheel.data.root_lin_vel_b[:, 0],
            self.num_envs,
            self.device,
        )

    def _apply_action(self) -> None:
        # self.actions = torch.zeros_like(self.actions)
        self.bigWheel.set_joint_effort_target(self.actions)

    def _get_observations(self) -> dict:
        self.previous_actions = self.actions.clone()
        #! 计算内轮抬升角
        euler_angles = math_utils.euler_xyz_from_quat(self.bigWheel.data.root_quat_w)
        #! 获取欧拉角元组
        self.euler_angle = torch.stack(euler_angles, dim=1)
        #! 获取pitch角
        self.pitch = euler_angles[1].squeeze()  # 提取第二列并去除多余维度
        #! 计算抬升角
        self.theta[:, self.LEFT] = (
            self.bigWheel.data.joint_pos[:, self._innerWheel_left_dof_idx].squeeze()
            + self.pitch
        )
        self.theta[:, self.RIGHT] = (
            self.bigWheel.data.joint_pos[:, self._innerWheel_right_dof_idx].squeeze()
            + self.pitch
        )
        #! 计算抬升角速度
        self.theta_vel[:, self.LEFT] = (
            self.bigWheel.data.joint_vel[:, self._innerWheel_left_dof_idx].squeeze()
            + self.bigWheel.data.root_ang_vel_b[:, 1].squeeze()
        )
        self.theta_vel[:, self.RIGHT] = (
            self.bigWheel.data.joint_vel[:, self._innerWheel_right_dof_idx].squeeze()
            + self.bigWheel.data.root_ang_vel_b[:, 1].squeeze()
        )
        #! 计算底盘速度 = (左外轮速度 + 右外轮速度) * 外轮半径 * 0.5
        # self.velocity_chassis = (
        #     (
        #         self.bigWheel.data.joint_vel[
        #             :, self._outerWheel_left_dof_idx
        #         ].unsqueeze(1)
        #         + self.bigWheel.data.joint_vel[
        #             :, self._outerWheel_right_dof_idx
        #         ].unsqueeze(1)
        #     )
        #     * self.outerWheel_radius
        #     * 0.5
        # )
        # self.velocity_chassis = self.velocity_chassis.squeeze(1)
        self.velocity_chassis = self.bigWheel.data.root_lin_vel_b[:, 0].unsqueeze(1)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.euler_angle,  # 3 欧拉角 roll pitch yaw
                    self.bigWheel.data.projected_gravity_b,  # 3 重力投影
                    self.velocity_chassis,  # 1 机体线速度x
                    self.bigWheel.data.root_ang_vel_b,  # 3 机体角速度 roll pitch yaw
                    self.theta_vel,  # 2 内轮抬升角速度
                    self.theta,  # 2 内轮抬升角
                    self._commands,  # 3 指令
                    self.actions,  # 4 动作
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        rewards = self.rwd.getReward(
            self._commands,
            self.bigWheel,
            self.actions,
            self.previous_actions,
            self.step_dt,
        )
        #! 汇总奖励
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        #! 存活时间超过最大存活时间时，机体死亡
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        #! 机体接触力大于 1.0 时，机体死亡
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
            # 分散重置时间以免所有机体同时死亡
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        # 新的 episode 时，随机生成指令
        if self.PLAY == 0:
            #! x方向速度生成的随机数
            self._commands[env_ids, 0] = torch.zeros_like(
                self._commands[env_ids, 1]
            ).uniform_(-self.MAX_LIN_VEL_X, self.MAX_LIN_VEL_X)
            #! w方向速度生成随机数
            self._commands[env_ids, 1] = torch.zeros_like(
                self._commands[env_ids, 1]
            ).uniform_(-self.MAX_ANGLE_VEL_Z, self.MAX_ANGLE_VEL_Z)
            #! 高度生成随机数
            # self._commands[env_ids, 2] = torch.tensor(
            #     [self.MIN_HEIGHT, self.MAX_HEIGHT], device=self.device
            # )[torch.randint(0, 2, (len(env_ids),))]

        # 重置内轮关节位置
        joint_pos = self.bigWheel.data.default_joint_pos[env_ids]
        joint_vel = self.bigWheel.data.default_joint_vel[env_ids]
        default_root_state = self.bigWheel.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.bigWheel.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.bigWheel.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.bigWheel.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"].update(extras)

