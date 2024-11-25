import torch

class DataLoader:
    def __init__(self, source: str):
        self.source = source
        # 这里可以根据你的数据源初始化不同的接口

    def load_data(self):
        # 例如从文件或传感器获取数据
        # 这里的代码会返回一个张量，假设是观测值
        if self.source == "random": 
            obs = torch.randn(1, 26)  # 示例数据
        if self.source == "sensor_data_source":
            obs = torch.randn(1, 26)  # 示例数据
        return obs
