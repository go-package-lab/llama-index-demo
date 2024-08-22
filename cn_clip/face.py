# 搜索不准确，clip 不是用于人脸搜索的
import torch
import numpy as np
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-L-14", device=device, download_root='/workspace/ai-model/clip_cn/')
model.eval()

def get_image_features(image_path):
    image = preprocess(Image.open("images/"+image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

# 准备数据库中的人脸图像
database_images = ["face-one.png", "face2.png", "face3.png"]  # 这些是你的人脸图像
database_features = [get_image_features(img) for img in database_images]

# 查询图像
query_image = "face1.png"
query_feature = get_image_features(query_image)

# 计算相似度
similarities = [np.dot(query_feature, db_feature.T) for db_feature in database_features]

# 找到最相似的图像
most_similar_index = np.argmax(similarities)
most_similar_image = database_images[most_similar_index]

print(f"The most similar image is: {most_similar_image}")
