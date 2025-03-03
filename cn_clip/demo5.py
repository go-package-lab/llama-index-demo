import torch
import hashlib
import os
import time
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

print("Available models:", available_models())

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-L-14", device=device, download_root='/workspace/ai-model/clip_cn/')
model.eval()

# 计算文件路径的MD5值
def md5_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# 保存特征到文件
def save_features(file_path, features):
    torch.save(features, file_path)

# 从文件加载特征
def load_features(file_path):
    return torch.load(file_path, weights_only=True)

# 内存特征缓存
memory_cache = {}

def get_image_features(image_path, model, preprocess, device):
    feature_key = md5_hash(image_path)

    # 先检查内存缓存
    if feature_key in memory_cache:
        print(f"Loading features from memory for {image_path}")
        return memory_cache[feature_key]

    # 再检查文件缓存
    feature_file = f"features/{feature_key}.pt"
    if os.path.exists(feature_file):
        print(f"Loading features from {feature_file}")
        features = load_features(feature_file)
        memory_cache[feature_key] = features  # 存入内存缓存
        return features

    # 计算新特征
    print(f"Extracting features for {image_path}")
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # 归一化
    save_features(feature_file, image_features)
    memory_cache[feature_key] = image_features  # 存入内存缓存
    return image_features

# 计算两张图片的相似度
def calculate_similarity(image_path1, image_path2, model, preprocess, device):
    features1 = get_image_features(image_path1, model, preprocess, device)
    features2 = get_image_features(image_path2, model, preprocess, device)

    similarity = torch.mm(features1, features2.t()).item()
    return similarity

# 示例：计算两张图片的相似度
os.makedirs("features", exist_ok=True)  # 创建存储特征的文件夹

image_path1 = "images/a1.png"
image_path2 = "images/a2.png"

# 计算两张图片的相似度，并记录耗时
def calculate_similarity_with_timing(image_path1, image_path2, model, preprocess, device):
    start_time = time.time()  # 记录开始时间

    similarity_score = calculate_similarity(image_path1, image_path2, model, preprocess, device)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算耗时

    return similarity_score, elapsed_time

# 示例：计算两张图片的相似度和耗时
image_path1 = "images/a1.png"
image_path2 = "images/a2.png"

similarity_score, elapsed_time = calculate_similarity_with_timing(image_path1, image_path2, model, preprocess, device)

print("Image Similarity Score:", similarity_score)
print(f"Time Taken: {elapsed_time:.4f} seconds")