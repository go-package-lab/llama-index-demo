from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
import requests
from io import BytesIO
import uvicorn
import argparse
import os
from typing import List

app = FastAPI(
    title="Image-Text Similarity API",
    description="API to compute similarity between multiple texts and a single image using CN-CLIP",
    version="1.0.0"
)

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-L-14", device=device, download_root='/workspace/ai-model/clip_cn/')
model.eval()

# 定义输入模型
class SimilarityRequest(BaseModel):
    texts: List[str]  # 输入的多个文本列表
    image_url: str  # 输入的单个图片URL

# 处理函数
def process_image_and_texts(texts: list[str], image_url: str):
    try:
        # 下载图片
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        # 预处理图片和文本
        processed_image = preprocess(image).unsqueeze(0).to(device)
        processed_text = clip.tokenize(texts).to(device)

        # 获取特征和相似度
        with torch.no_grad():
            image_features = model.encode_image(processed_image)
            text_features = model.encode_text(processed_text)

            # 归一化
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # 计算相似度
            logits_per_image, _ = model.get_similarity(processed_image, processed_text)
            # probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]  # 相似度百分比
            probs = logits_per_image.cpu().numpy()[0]  # 原始logits值

        # 构造结果，同时包含logits和相似度
        result = [
            {
                "text": text,
                # "logits": float(logits[i]),  # 转换为float以确保可序列化
                "similarity": round(float(prob)/100, 3)  # 转换为百分比并保留3位小数
            }
            for i, (text, prob) in enumerate(zip(texts, probs))
        ]

        return result

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# API 端点
@app.post("/compute_similarity",
          summary="Compute similarity between multiple texts and a single image",
          response_description="Returns both logits and similarity scores between each text and the provided image")
async def compute_similarity(request: SimilarityRequest):
    result = process_image_and_texts(request.texts, request.image_url)
    return {
        "status": "success",
        "results": result,
        "message": "Similarity and logits computed successfully"
    }

def get_port():
    """获取端口号，优先级：命令行参数 > 环境变量 > 默认值"""
    parser = argparse.ArgumentParser(description='Run FastAPI server with custom port')
    parser.add_argument('--port', type=int, default=None, help='Port to run the server on')
    args = parser.parse_args()

    # 优先使用命令行参数
    if args.port is not None:
        return args.port

    # 次优先使用环境变量
    env_port = os.environ.get('PORT')
    if env_port is not None:
        try:
            return int(env_port)
        except ValueError:
            raise ValueError("PORT environment variable must be an integer")

    # 默认端口
    return 8000

# 启动服务
if __name__ == "__main__":
    try:
        port = get_port()
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Failed to start server: {str(e)}")
        exit(1)