import torch
import numpy as np
from PIL import Image
import open_clip
import os
import heapq

IMAGE_DIR = "data/images"
EMBEDDING_FILE = "data/clip_embeddings.npy"
FILENAMES_FILE = "data/image_filenames.npy"
TOP_K = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device)
model.eval()

# Load Library Vector
image_features = np.load(EMBEDDING_FILE)
image_filenames = np.load(FILENAMES_FILE)

# Calculate Similarity
def find_similar_images(query_path, top_k=TOP_K):
    image = preprocess(Image.open(query_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        query_feature = model.encode_image(image).cpu().numpy()[0]

    # Normalization
    query_feature /= np.linalg.norm(query_feature)
    image_norm = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)

    # Calculate Cos Similarity
    sims = np.dot(image_norm, query_feature)

    # Find the index of Top-K Similarity
    top_k_idx = heapq.nlargest(top_k, range(len(sims)), sims.__getitem__)
    return [(image_filenames[i], sims[i]) for i in top_k_idx]

# Main Function
if __name__ == "__main__":

    query_image = "data\Example\example.jpg"

    results = find_similar_images(query_image)
    print(f"\n🔍 Searching:{os.path.basename(query_image)}")
    print("🎯 Recommendation:")
    for name, score in results:
        print(f"  - {name} (Similarity Rate: {score:.4f})")