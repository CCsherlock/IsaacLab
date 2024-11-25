import torch
from omegaconf import OmegaConf
from rl_games.algos_torch import model_builder

class InferenceModel:
    def __init__(self, config_file: str, checkpoint_file: str):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.model = self.load_model()

    def load_model(self):
        try:
            # 读取配置文件
            with open(self.config_file, "r") as f:
                config_data = f.read()
            config = OmegaConf.create(config_data)

            builder = model_builder.ModelBuilder()
            network = builder.load(config["params"])

            model_config = {
                "actions_num": 4,
                "input_shape": [26],
                "num_seqs": 4,
                "value_size": 1,
                "normalize_value": True,
                "normalize_input": True,
            }

            model = network.build(model_config)
            model.eval()

            checkpoint = torch.load(self.checkpoint_file, weights_only=False)
            model.load_state_dict(checkpoint["model"])

            print("模型加载成功！")
            return model
        except Exception as e:
            print(f"加载模型时发生错误: {e}")
            raise

    def infer(self, input_data: torch.Tensor, rnn_states=None):
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": input_data,
            "rnn_states": rnn_states,
        }
        
        with torch.no_grad():
            res_dict = self.model(input_dict)
        return res_dict
