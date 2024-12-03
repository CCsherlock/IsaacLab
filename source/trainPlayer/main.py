from trainPlayer.dataProcess.data_loader import DataLoader
from trainPlayer.inference.model import InferenceModel
from trainPlayer.communication.message_handler import MessageHandler
from trainPlayer.dataRecord.dataRecorder import DataRecorder, DataRecordCfg

# import time
import torch


def main():
    # 配置文件和模型权重路径
    config_file = "trainPlayer/config/rl_games_ppo_cfg.yaml"
    checkpoint_file = "trainPlayer/bigWheel_direct.pth"

    # 初始化各个模块
    data_loader = DataLoader("sensor_data_source")
    model = InferenceModel(config_file, checkpoint_file)
    message_handler = MessageHandler("219.216.98.54", 9090)
    feedBackData = DataRecordCfg()
    dataRecord = DataRecorder("FBdata", feedBackData, save_interval=10)
    recordCnt = 0
    try:
        while True:
            # 获取数据
            if message_handler.receiveFlag:
                # 记录收到的数据
                if recordCnt % 100 == 0:
                    dataRecord.record(message_handler.msg_dict.dict)
                    recordCnt = 0
                recordCnt += 1
                obs = data_loader.get_data(message_handler.msg)
                # 推理
                res_dict = model.infer(obs)
                # 发送推理结果
                actions = res_dict.get("actions")
                if isinstance(actions, torch.Tensor):
                    actions = actions.squeeze().tolist()  # 去除多余的维度，并转换为列表
                if len(actions) == 4:
                    # 拆解成四个独立的 float 值
                    action1, action2, action3, action4 = actions
                    print(f"推理结果: {action1}, {action2}, {action3}, {action4}")
                message_handler.send_message(actions)
                message_handler.receiveFlag = False
            # time.sleep(1)  # 等待接收线程运行一段时间
    except KeyboardInterrupt:
        print("\n程序被中断，正在关闭连接...")
        # 保存数据
        dataRecord.save()
        # 关闭连接
        message_handler.close()


if __name__ == "__main__":
    main()
