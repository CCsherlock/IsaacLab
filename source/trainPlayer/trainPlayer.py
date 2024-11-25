import torch
from omegaconf import OmegaConf
from rl_games.algos_torch import model_builder


# 加载模型函数
def load_model(config_file: str, checkpoint_file: str) -> torch.nn.Module:
    try:
        # 读取配置文件
        with open(config_file, "r") as f:
            config_data = f.read()
        config = OmegaConf.create(config_data)

        # 构建网络
        builder = model_builder.ModelBuilder()
        network = builder.load(config["params"])

        # 配置模型参数
        model_config = {
            "actions_num": 4,  # 输出动作维度
            "input_shape": [26],  # 输入观测维度
            "num_seqs": 4,
            "value_size": 1,
            "normalize_value": True,
            "normalize_input": True,
        }
        model = network.build(model_config)
        model.eval()  # 切换为评估模式

        # 加载权重
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        model.load_state_dict(checkpoint["model"])

        print("模型加载成功！")
        return model
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        raise


# 推理函数
def infer(model: torch.nn.Module, input_data: torch.Tensor, rnn_states=None):
    try:
        input_dict = {
            "is_train": False,  # 非训练模式
            "prev_actions": None,  # 上一次的动作（此处为 None）
            "obs": input_data,  # 当前输入观测
            "rnn_states": rnn_states,  # RNN 状态
        }

        # 推理
        with torch.no_grad():
            res_dict = model(input_dict)

        return res_dict
    except Exception as e:
        print(f"推理时发生错误: {e}")
        raise


# 主函数
def main():
    # 配置文件和模型权重路径
    config_file = "./rl_games_ppo_cfg.yaml"
    checkpoint_file = "./bigWheel_direct.pth"

    # 加载模型
    model = load_model(config_file, checkpoint_file)

    # 创建观测值
    root_lin_vel_b = torch.randn(3)  # 3 机体线速度
    root_ang_vel_b = torch.randn(3)  # 3 机体角速度
    projected_gravity_b = torch.randn(3)  # 3 重力投影
    joint_pos = torch.randn(4)  # 4 关节位置
    joint_vel = torch.randn(4)  # 4 关节速度
    theta = torch.randn(2)  # 2 内轮抬升角
    commands = torch.randn(3)  # 3 指令
    actions = torch.randn(4)  # 4 动作

    # 将各个张量连接成一个大观测张量
    obs = torch.cat(
        [
            tensor
            for tensor in (
                root_lin_vel_b,  # 3 机体线速度
                root_ang_vel_b,  # 3 机体角速度
                projected_gravity_b,  # 3 重力投影
                joint_pos,  # 4 关节位置
                joint_vel,  # 4 关节速度
                theta,  # 2 内轮抬升角
                commands,  # 3 指令
                actions,  # 4 动作
            )
            if tensor is not None
        ],
        dim=-1,
    )

    # RNN 状态设置为 None（如果不使用 RNN）
    rnn_states = None

    # 推理
    res_dict = infer(model, obs.unsqueeze(0), rnn_states)  # 扩展维度以适配 batch size

    # 输出结果
    print("\n推理结果：")
    print("Actions (动作输出):", res_dict["actions"])
    print("其他输出:", res_dict)


# 运行主函数
if __name__ == "__main__":
    main()
