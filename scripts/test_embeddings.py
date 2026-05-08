import sys
import os
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from models.clip_model import load_clip_model, generate_image_embedding

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_FOLDER = os.path.join(DATA_DIR, "fashion-dataset", "images")

def test_embeddings():
    model, processor, device = load_clip_model()

    # Test on first 5 images
    files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")][:5]
    embeddings = []

    for file in files:
        path = os.path.join(IMAGE_FOLDER, file)
        try:
            image = Image.open(path).convert("RGB")
            embedding = generate_image_embedding(image, model, processor, device=device)
            embeddings.append(embedding)
            print(f"Processed {file}: embedding shape {embedding.shape}")
        except Exception as e:
            print(f"Error processing {file}: {e}")

    embeddings = np.array(embeddings)
    print(f"Test embeddings shape: {embeddings.shape}")
    print("Embedding test successful!")

if __name__ == "__main__":
    test_embeddings()