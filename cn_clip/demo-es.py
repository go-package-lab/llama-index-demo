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

# 加载图片并生成向量
image = preprocess(Image.open("images/cat.png")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    image_vector = image_features.cpu().numpy().flatten().tolist()

# 定义要存入Elasticsearch的数据
doc = {
    "image_id": "cat",  # 图片的唯一标识
    "vector": image_vector
}

# 将数据存入Elasticsearch
res = es.index(index=index_name, id=doc["image_id"], body=doc)
print("Document indexed:", res)
