import torch
import numpy as np
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-L-14", device=device, download_root='/workspace/ai-model/clip_cn/')
model.eval()

# 输入的文本
text_input = ["雷峰塔"]
text = clip.tokenize(text_input).to(device)

# 计算文本特征向量
with torch.no_grad():
    text_features = model.encode_text(text)
    # 对特征进行归一化处理
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 将特征向量转换为 numpy 数组
text_features_np = text_features.cpu().numpy()

# 将向量结果保存为文本文件
np.savetxt("text_features.txt", text_features_np, delimiter=",")

print("Text features have been saved to text_features.txt")
