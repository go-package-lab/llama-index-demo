from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import hashlib
import os
import requests
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
from io import BytesIO

app = FastAPI()

# 配置模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-L-14", device=device, download_root='/workspace/ai-model/clip_cn/')
model.eval()

# MD5 哈希计算
def md5_hash(image_bytes):
    hash_md5 = hashlib.md5()
    hash_md5.update(image_bytes)
    return hash_md5.hexdigest()

# 保存和加载特征
def save_features(file_path, features):
    torch.save(features, file_path)

def load_features(file_path):
    return torch.load(file_path, map_location=device)

# 内存缓存
memory_cache = {}

def get_image_features(image_bytes, model, preprocess):
    feature_key = md5_hash(image_bytes)

    if feature_key in memory_cache:
        return memory_cache[feature_key]

    feature_file = f"features/{feature_key}.pt"
    if os.path.exists(feature_file):
        features = load_features(feature_file)
        memory_cache[feature_key] = features
        return features

    image = preprocess(Image.open(BytesIO(image_bytes)).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
        features /= features.norm(dim=-1, keepdim=True)

    save_features(feature_file, features)
    memory_cache[feature_key] = features
    return features

def calculate_similarity(feature1, feature2):
    return torch.mm(feature1, feature2.t()).item()

class ImageURL(BaseModel):
    url: str

@app.post("/tools/match-image")
async def match_image(image_url: ImageURL):
    try:
        response = requests.get(image_url.url)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")

    if response.headers['Content-Type'] not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    query_features = get_image_features(response.content, model, preprocess)

    best_match = None
    best_score = float('-inf')

    # 遍历已有图片
    for image_name in os.listdir("match-images"):
        image_path = os.path.join("match-images", image_name)
        with open(image_path, "rb") as f:
            image_features = get_image_features(f.read(), model, preprocess)
        similarity = calculate_similarity(query_features, image_features)

        if similarity > best_score:
            best_score = similarity
            best_match = image_name

    return {"matched_image": best_match, "similarity_score": best_score}