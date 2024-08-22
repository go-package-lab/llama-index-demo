import os
import torch
from PIL import Image
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

# 初始化Elasticsearch客户端
es = Elasticsearch("http://localhost:9200")

# 检查Elasticsearch是否连接成功
if not es.ping():
    raise ValueError("连接到Elasticsearch失败")

# 定义索引名称
index_name = "image_index"

# 创建索引，如果索引不存在
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body={
        "mappings": {
            "properties": {
                "image_id": {"type": "keyword"},
                "vector": {"type": "dense_vector", "dims": 768}  # 根据模型输出的维度设置
            }
        }
    })

# 加载模型和预处理
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-L-14", device=device, download_root='/workspace/ai-model/clip_cn/')
model.eval()


def process_image(image_path):
    """
    处理单张图片，生成向量。
    """
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        image_vector = image_features.cpu().numpy().flatten().tolist()

    return image_vector


def index_images_from_directory(directory_path):
    """
    从指定目录读取所有图片并将其索引到Elasticsearch。
    """
    actions = []

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(directory_path, filename)
            image_id = os.path.splitext(filename)[0]  # 使用文件名作为图片的唯一标识

            # 检查文档是否已经存在
            if es.exists(index=index_name, id=image_id):
                print(f"图片 {image_id} 已存在于索引中，跳过处理。")
                continue

            image_vector = process_image(image_path)

            doc = {
                "image_id": image_id,
                "vector": image_vector
            }

            action = {
                "_index": index_name,
                "_id": image_id,
                "_source": doc
            }

            actions.append(action)

    # 批量索引所有图片
    if actions:
        bulk(es, actions)
        print(f"成功索引 {len(actions)} 张图片。")
    else:
        print("没有找到需要索引的图片。")


# 指定图片目录
image_directory = "images"
index_images_from_directory(image_directory)
