import os

from gme_inference import GmeQwen2VL

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 在导入模型前设置

# 定义美食相关的中文文本和图片链接
texts = [
    "冈仁波齐？",      # Text 2
]
images = [
    'https://imgaliyuncdn.miaopai.com/aigc/image/1071871043043458105_ygHUEM8Iwu.jpeg',  # 美食图片1（假设为火锅）
]

# 初始化模型
gme = GmeQwen2VL("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")

# 1. 单模态嵌入 (Single-modal embedding)
e_text = gme.get_text_embeddings(texts=texts)
e_image = gme.get_image_embeddings(images=images)
similarity_single = (e_text * e_image).sum(-1)
print("单模态文本-图片相似度:", similarity_single,e_text)
