import pandas as pd
from .dataRecorderCfg import DataRecordCfg
import time
import os
from datetime import datetime


class DataRecorder:
    def __init__(self, path, cfg: DataRecordCfg, save_interval=10):
        # 配置数据
        self.dataConfig = cfg
        # 获取当前脚本所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 获取上一级目录（父级目录）
        parent_dir = os.path.dirname(current_dir)
        # 定义 log 文件夹路径
        log_dir = os.path.join(parent_dir, "log")
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )  # 获取当前时间戳，格式为：年-月-日_时-分-秒
        # 构建完整的文件路径
        self.path = os.path.join(log_dir, f"{path}_{timestamp}.csv")

        # 初始化一个空的 DataFrame
        self.data = pd.DataFrame(columns=self.dataConfig.record_lists)

        # 设置定期保存的时间间隔（单位：秒）
        self.save_interval = save_interval

        # 记录开始时间
        self.last_save_time = time.time()

    def record(self, data: dict):
        # 逐行添加数据
        new_data = {key: [data[key]] for key in self.dataConfig.dict}
        # 将新数据追加到 DataFrame 中
        new_df = pd.DataFrame(new_data)
        self.data = pd.concat([self.data, new_df], ignore_index=True)

        # # 检查是否到了保存的时间间隔
        # if time.time() - self.last_save_time >= self.save_interval:
        #     self.save()
        #     self.last_save_time = time.time()

    def save(self):
        # 将数据保存到 CSV 文件中
        self.data.to_csv(self.path, index=False)
        print(f"数据已保存到 {self.path}")

    def load(self):
        # 加载已保存的数据
        self.data = pd.read_csv(self.path)
        print(f"数据已从 {self.path} 加载")
