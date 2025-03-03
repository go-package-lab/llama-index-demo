import torch
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

print("Available models:", available_models())
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-L-14", device=device, download_root='/workspace/ai-model/clip_cn/')
model.eval()
labels = ["杰尼龟", "孙悟空", "唐僧", "皮卡丘"]
image = preprocess(Image.open("/workspace/py/llama-index-demo/cn_clip/images/sunwukong1.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(labels).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]  # 取第一行，去掉多余维度

# 输出百分比形式的相似度

for label, prob in zip(labels, probs):
    percentage = prob * 100  # 将概率转换为百分比
    print(f"{label}: {percentage:.2f}%")