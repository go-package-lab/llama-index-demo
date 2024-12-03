import torch
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

print("Available models:", available_models())
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-L-14", device=device, download_root='/workspace/ai-model/clip_cn/')
model.eval()

def preprocess_image(image_path, preprocess, device):
    return preprocess(Image.open(image_path)).unsqueeze(0).to(device)

def calculate_similarity(image_path1, image_path2, model, preprocess, device):
    image1 = preprocess_image(image_path1, preprocess, device)
    image2 = preprocess_image(image_path2, preprocess, device)

    with torch.no_grad():
        image_features1 = model.encode_image(image1)
        image_features2 = model.encode_image(image2)

        # 归一化
        image_features1 /= image_features1.norm(dim=-1, keepdim=True)
        image_features2 /= image_features2.norm(dim=-1, keepdim=True)

        # 计算余弦相似度
        similarity = torch.mm(image_features1, image_features2.t()).item()

    return similarity

similarity_score = calculate_similarity("images/a1.png", "images/a2.png", model, preprocess, device)
print("Image Similarity Score:", similarity_score)

similarity_score = calculate_similarity("images/a1.png", "images/公园1.jpg", model, preprocess, device)
print("Image Similarity Score:", similarity_score)