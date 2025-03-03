from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from gme_inference import GmeQwen2VL
from typing import List
import uvicorn
import argparse
import numpy as np

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 初始化 FastAPI 应用
app = FastAPI(
    title="Image-Text Similarity API",
    description="API to compute similarity between multiple texts and a single image using GmeQwen2VL model",
    version="1.0.0"
)


# 定义请求体的数据模型
class SimilarityRequest(BaseModel):
    texts: List[str]  # 输入的多个文本列表
    image_url: str  # 输入的单个图片URL（改为字符串）


# 初始化模型（全局加载，避免每次请求都重新加载）
model = GmeQwen2VL("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")


@app.post("/compute_similarity",
          summary="Compute similarity between multiple texts and a single image",
          response_description="Returns the similarity scores between each text and the provided image")
async def compute_similarity(request: SimilarityRequest):
    """
    计算多个文本与一张图片之间的相似度

    Args:
        request: SimilarityRequest object containing texts and image_url

    Returns:
        dict: Contains similarity scores for each text and status
    """
    try:
        # 获取文本和图片URL
        texts = request.texts
        image_url = request.image_url

        # 验证输入
        if not texts:
            raise HTTPException(status_code=400, detail="Texts cannot be empty")

        if not image_url:
            raise HTTPException(status_code=400, detail="Image URL must be provided")

        # 将单个URL转换为列表以适配模型接口
        image_urls = [image_url]

        # 计算嵌入
        e_text = model.get_text_embeddings(texts=texts)  # shape: (n_texts, embedding_dim)
        e_image = model.get_image_embeddings(images=image_urls)  # shape: (1, embedding_dim)

        # 计算每个文本与图片的相似度
        similarity = (e_text * e_image).sum(-1).tolist()  # shape: (n_texts,)

        # 构造结果
        result = [
            {"text": text, "similarity": round(float(sim), 3)}
            for text, sim in zip(texts, similarity)
        ]

        return {
            "status": "success",
            "similarity_scores": result,
            "message": "Similarity computed successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing similarity: {str(e)}")


@app.get("/health", summary="Check API health")
async def health_check():
    return {"status": "healthy"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run FastAPI server with specified port")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 获取端口号：优先级 命令行 > 环境变量 > 默认值
    port = args.port
    if "PORT" in os.environ:
        port = int(os.environ["PORT"])

    # 运行服务
    uvicorn.run(app, host="0.0.0.0", port=port)