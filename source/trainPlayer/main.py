from dataProcess.data_loader import DataLoader
from inference.model import InferenceModel
from communication.message_handler import MessageHandler
import time


def main():
    # 配置文件和模型权重路径
    config_file = "./config/rl_games_ppo_cfg.yaml"
    checkpoint_file = "./bigWheel_direct.pth"

    # 初始化各个模块
    data_loader = DataLoader("sensor_data_source")
    model = InferenceModel(config_file, checkpoint_file)
    message_handler = MessageHandler("219.216.98.54", 24)

    while True:
        # 获取数据
        obs = data_loader.load_data()
        # 推理
        res_dict = model.infer(obs)
        # 发送推理结果
        message_handler.send_message(str(res_dict["actions"]))
        time.sleep(1)  # 等待接收线程运行一段时间

    # 关闭连接
    message_handler.close()


if __name__ == "__main__":
    main()
