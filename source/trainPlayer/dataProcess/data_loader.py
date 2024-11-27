import torch
from communication.message_handler import MessageReceive
class DataLoader:
    def __init__(self, source: str):
        self.source = source
        # 这里可以根据你的数据源初始化不同的接口
        self.obs_data = torch.zeros(1, 21)
    def load_data(self):
        # 例如从文件或传感器获取数据
        # 这里的代码会返回一个张量，假设是观测值
        if self.source == "random": 
            obs = torch.randn(1, 21)  # 示例数据
        if self.source == "sensor_data_source":
            obs = torch.randn(1, 21)  # 示例数据
        return obs
    def get_data(self,data:MessageReceive):
        data_list = (
            data.euler_angle + 
            data.projected_gravity_b + 
            data.velocity_chassis + 
            data.root_ang_vel_b + 
            data.theta_vel + 
            data.theta + 
            data.commands + 
            data.actions
        )
        # 转换为 torch 张量并调整为 (1, 21) 形状
        self.obs_data = torch.tensor(data_list, dtype=torch.float32).unsqueeze(0)
        return self.obs_data