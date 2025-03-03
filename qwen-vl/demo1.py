import os

from gme_inference import GmeQwen2VL

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 在导入模型前设置

# 定义美食相关的中文文本和图片链接
texts = [
    "冈仁波齐？",      # Text 2
    "大山？",      # Text 2
    "比卡丘",      # Text 2
]
images = [
    'https://imgaliyuncdn.miaopai.com/aigc/image/1071871043043458105_ygHUEM8Iwu.jpeg',  # 美食图片1（假设为火锅）
    'https://imgaliyuncdn.miaopai.com/aigc/image/1071871043043458105_ygHUEM8Iwu.jpeg',  # 美食图片1（假设为火锅）
    'https://imgaliyuncdn.miaopai.com/aigc/image/1071871043043458105_ygHUEM8Iwu.jpeg',  # 美食图片1（假设为火锅）
]

# 初始化模型
gme = GmeQwen2VL("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct", use_fast=True)

# 1. 单模态嵌入 (Single-modal embedding)
e_text = gme.get_text_embeddings(texts=texts)
e_image = gme.get_image_embeddings(images=images)
similarity_single = (e_text * e_image).sum(-1)
print("单模态文本-图片相似度:", similarity_single)
# 预期输出示例: tensor([0.2281, 0.6001], dtype=torch.float16)

# 2. 带指令的嵌入 (Instruction-based embedding)
e_query = gme.get_text_embeddings(texts=texts, instruction='找到与给定文本匹配的图片。')
e_corpus = gme.get_image_embeddings(images=images, is_query=False)
similarity_instruction = (e_query * e_corpus).sum(-1)
print("指令调整后的文本-图片相似度:", similarity_instruction)
# 预期输出示例: tensor([0.2433, 0.7051], dtype=torch.float16)

# 3. 融合模态嵌入 (Fused-modal embedding)
e_fused = gme.get_fused_embeddings(texts=texts, images=images)
print("融合嵌入的形状:", e_fused.shape)  # 检查形状，预期可能是 [2, embedding_dim]
# 计算两个文本-图片对之间的相似度
if len(e_fused) >= 2:  # 确保有足够的嵌入进行比较
    similarity_fused = (e_fused[0] * e_fused[1]).sum()
    print("融合嵌入之间的相似度:", similarity_fused)
else:
    print("只有一个融合嵌入，无法计算相似度:", e_fused)
# 预期输出示例: tensor(0.6108, dtype=torch.float16)（如果有两个嵌入）

# 4. 可选：融合嵌入与单模态嵌入的比较
print("融合嵌入与文本嵌入的相似度:", (e_fused[0] * e_text[0]).sum(-1))