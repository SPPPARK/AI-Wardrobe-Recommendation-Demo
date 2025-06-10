import streamlit as st
import os
from PIL import Image
import torch
import open_clip
import numpy as np
import requests

IMAGE_DIR = "data/images"
EMBEDDING_FILE = "data/clip_embeddings.npy"
FILENAMES_FILE = "data/image_filenames.npy"
TOP_K = 5

HF_TOKEN = "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"
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Load open_clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device).eval()

# Load embeddings 
@st.cache_data(show_spinner=False)
def load_embeddings():
    return np.load(EMBEDDING_FILE), np.load(FILENAMES_FILE)

image_features, image_filenames = load_embeddings()

# Generate explanation
def generate_reason(query_name, rec_name):
    prompt = (
        f"The user likes the clothing item named {query_name}, and the recommended item is {rec_name}. "
        "Write one short, natural-sounding sentence explaining why these two items are stylistically similar and suitable for the user."
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 100,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=45)
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()
        elif isinstance(result, dict) and "error" in result:
            return f"(API error: {result['error']})"
        else:
            return str(result)

    except Exception as e:
        return f"(Generation failed: {str(e)})"

# Similar image search
def search_similar(uploaded_img):
    image = preprocess(uploaded_img).unsqueeze(0).to(device)
    with torch.no_grad():
        q = model.encode_image(image).cpu().numpy()[0]
    q = q / np.linalg.norm(q)
    db = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
    scores = np.dot(db, q)
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    return [(image_filenames[i], scores[i]) for i in top_idx]

# Streamlit UI 
st.set_page_config(page_title="AI Clothing Style Recommender", layout="wide")
st.title("👗 AI Clothing Style Recommender")
st.markdown("Upload a clothing image, and we will recommend visually similar items and provide a natural language explanation.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Query Image", width=300)

    with st.spinner("Searching for similar styles..."):
        results = search_similar(img)

    st.subheader(f"🎯 Top {TOP_K} Recommendations")

    for name, score in results:
        rec_path = os.path.join(IMAGE_DIR, name)
        col1, col2 = st.columns([1, 3])

        with col1:
            st.image(rec_path, caption=name, use_column_width=True)

        with col2:
            with st.spinner("Generating explanation..."):
                reason = generate_reason(uploaded.name, name)
            st.markdown(f"Explanation: {reason}")
            st.markdown(f"Similarity Score: `{score:.4f}`")