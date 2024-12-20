class DataRecordCfg:

    def __init__(self) -> None:
        self.dict = {
            "time",
            "euler_angle_roll",
            "euler_angle_pitch",
            "euler_angle_yaw",
            "projected_gravity_b_x",
            "projected_gravity_b_y",
            "projected_gravity_b_z",
            "velocity_chassis",
            "root_ang_vel_b_x",
            "root_ang_vel_b_y",
            "root_ang_vel_b_z",
            "theta_vel_left",
            "theta_vel_right",
            "theta_left",
            "theta_right",
            "command_x",
            "command_w",
            "command_z",
            "action_in_left",
            "action_in_right",
            "action_out_left",
            "action_out_right",
        }
        self.record_lists = {key: [] for key in self.dict}
