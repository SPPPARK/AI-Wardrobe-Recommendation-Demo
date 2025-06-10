import os
import torch
import numpy as np
from PIL import Image
import tqdm
import open_clip

IMAGE_FOLDER = "data/images"
OUTPUT_EMBEDDINGS = "data/clip_embeddings.npy"
OUTPUT_FILENAMES = "data/image_filenames.npy"

#Load Clip Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device)
model.eval()

# Extract Clip Functions
def extract_clip_features(image_folder):
    embeddings = []
    filenames = []

    for filename in tqdm.tqdm(os.listdir(image_folder)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(image_folder, filename)
        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(image).cpu().numpy()
                embeddings.append(emb)
                filenames.append(filename)
        except Exception as e:
            print(f"跳过图像 {filename}，错误：{e}")

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, filenames

# Main Function
if __name__ == "__main__":
    print("Extracting...")
    emb, names = extract_clip_features(IMAGE_FOLDER)
    np.save(OUTPUT_EMBEDDINGS, emb)
    np.save(OUTPUT_FILENAMES, names)
    print(f"Total {len(emb)} Image")