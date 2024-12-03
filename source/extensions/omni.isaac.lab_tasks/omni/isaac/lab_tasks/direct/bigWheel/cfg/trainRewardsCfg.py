# import math
import torch
from omni.isaac.lab.assets import Articulation


class TrainRewards:
    def __init__(self):
        # reward scales
        self.lin_vel_reward_scale = 10.0  # 线速度奖励
        self.yaw_rate_reward_scale = 0.2  # yaw 转向速度奖励
        self.velocity_coupling_reward_scale = 0.0  # 速度耦合奖励
        self.height_reward_scale = 5.0  # 高度奖励
        self.z_vel_reward_scale = -0.01  # 垂直速度奖励
        self.ang_vel_reward_scale = -0.1  # pitch roll角速度奖励
        self.joint_torque_reward_scale_inner = -5e-6  # 内轮关节扭矩奖励
        self.joint_accel_reward_scale_inner = -2.5e-7  # 内轮关节加速度奖励
        self.joint_torque_reward_scale_outer = -2.5e-6  # 外轮关节扭矩奖励
        self.joint_accel_reward_scale_outer = -2.5e-7  # 外轮关节加速度奖励
        self.action_rate_reward_scale_inner = -0.001  # 内轮动作速率奖励 扭矩变化率
        self.action_rate_reward_scale_outer = -0.001  # 外轮动作速率奖励 扭矩变化率
        self.flat_orientation_reward_scale = -3.5  # 重力投影方向奖励

        self.height_threshold = 0.1  # 高度阈值

    def getReward(
        self,
        _commands: torch.Tensor,
        bigWheel: Articulation,
        actions: torch.Tensor,
        previous_actions: torch.Tensor,
        step_dt: float,
    ) -> dict:
        #! 线速度跟踪
        lin_vel_error = torch.square(
            _commands[:, 0] - bigWheel.data.root_lin_vel_b[:, 0]
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        #! yaw 角速度跟踪
        yaw_rate_error = torch.square(
            _commands[:, 1] - bigWheel.data.root_ang_vel_b[:, 2]
        )
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.5)
        # # #! 高度跟踪 计算目标高度与实际高度的误差
        # height_error = torch.abs(_commands[:, 2] - bigWheel.data.root_pos_w)
        # #! 将误差线性映射到奖励范围
        # height_error_mapped = 1.0 - height_error / 0.1575

        # #! 将奖励值限制在 [0.0, 1.0] 范围内
        # height_error_mapped = torch.clamp(height_error_mapped, 0.0, 1.0)

        #! 垂直速度惩罚
        z_vel_error = torch.square(bigWheel.data.root_lin_vel_b[:, 2]) * 0.5

        #! pitch/roll 角速度惩罚
        ang_vel_error = torch.sum(
            torch.square(bigWheel.data.root_ang_vel_b[:, :2]), dim=1
        )
        #! 内轮关节扭矩惩罚
        joint_torques_inner = torch.sum(
            torch.square(bigWheel.data.applied_torque[:, :2]), dim=1
        )
        #! 外轮关节扭矩惩罚
        joint_torques_outer = torch.sum(
            torch.square(bigWheel.data.applied_torque[:, 2:]), dim=1
        )
        #! 内轮关节加速度惩罚
        joint_accel_inner = torch.sum(
            torch.square(bigWheel.data.joint_acc[:, :2]), dim=1
        )
        #! 外轮关节加速度惩罚
        joint_accel_outer = torch.sum(
            torch.square(bigWheel.data.joint_acc[:, 2:]), dim=1
        )
        #! 内轮动作变化速率惩罚
        action_rate_inner = torch.sum(
            torch.square(actions[:, :2] - previous_actions[:, :2]), dim=1
        )
        #! 外轮动作变化速率惩罚
        action_rate_outer = torch.sum(
            torch.square(actions[:, 2:] - previous_actions[:, 2:]), dim=1
        )
        #! pitch/roll 与地面角度惩罚
        flat_orientation = torch.sum(
            torch.square(bigWheel.data.projected_gravity_b[:, :2]), dim=1
        )
        # # 判断目标高度是否超过阈值
        # height_threshold_met = _commands[:, 2] > self.height_threshold

        # height_reward = torch.where(
        #     height_threshold_met,  # 条件：目标高度是否超过阈值
        #     height_error_mapped
        #     * self.height_reward_scale
        #     * step_dt,  # 如果满足条件，计算高度奖励
        #     height_error_mapped
        #     * self.height_reward_scale
        #     * (1.0 - lin_vel_error_mapped)
        #     * step_dt,  # 否则计算动态调整的奖励
        # )
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped
            * self.lin_vel_reward_scale
            * step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped
            * self.yaw_rate_reward_scale
            * step_dt,
            # "track_velocity_coupling_reward": velocity_coupling_reward
            # * self.velocity_coupling_reward_scale
            # * step_dt,
            # "track_height_reward": height_reward,
            "lin_vel_z_l2": z_vel_error * self.z_vel_reward_scale * step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.ang_vel_reward_scale * step_dt,
            "dof_torques_inner_l2": joint_torques_inner
            * self.joint_torque_reward_scale_inner
            * step_dt,
            "dof_torques_outer_l2": joint_torques_outer
            * self.joint_torque_reward_scale_outer
            * step_dt,
            "dof_acc_inner_l2": joint_accel_inner
            * self.joint_accel_reward_scale_inner
            * step_dt,
            "dof_acc_outer_l2": joint_accel_outer
            * self.joint_accel_reward_scale_outer
            * step_dt,
            "action_rate_inner_l2": action_rate_inner
            * self.action_rate_reward_scale_inner
            * step_dt,
            "action_rate_outer_l2": action_rate_outer
            * self.action_rate_reward_scale_outer
            * step_dt,
            "flat_orientation_l2": flat_orientation
            * self.flat_orientation_reward_scale
            * step_dt,
        }
        return rewards
