import torch
import torch.nn as nn

# 定义模型架构
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.fc = nn.Linear(26, 4)  # 根据实际模型调整

    def forward(self, x):
        return self.fc(x)

def load_model(model_path):
    try:
        # 加载包含额外信息的模型
        checkpoint = torch.load(model_path)
        
        # 提取模型的权重部分
        model_weights = checkpoint['model']  # 这里假设保存时使用了 'model' 键

        # 创建模型并加载权重
        model = YourModel()
        model.load_state_dict(model_weights)
        model.eval()
        print("模型加载成功。")
        return model
        
    except Exception as e:
        print(f"加载模型时出错: {e}")

# 假设模型路径是 "bigWheel.pth"
model = load_model("bigWheel.pth")
