"""
Generate CLIP embeddings for the FULL fashion dataset.
Builds a normalized FAISS index for cosine similarity search.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm

from models.clip_model import load_clip_model, generate_image_embedding


DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_FOLDER = os.path.join(DATA_DIR, "fashion-dataset", "images")


def main():
    model, processor, device = load_clip_model()

    # Check for existing data to resume
    embeddings_path = os.path.join(DATA_DIR, "embeddings.npy")
    image_paths_path = os.path.join(DATA_DIR, "image_paths.npy")

    if os.path.exists(embeddings_path) and os.path.exists(image_paths_path):
        print("Existing embeddings found. Loading to resume...")
        existing_embeddings = np.load(embeddings_path)
        existing_image_paths = np.load(image_paths_path)
        processed_paths = set(existing_image_paths)
        print(f"Resuming from {len(processed_paths)} already processed images.")
    else:
        existing_embeddings = np.empty((0, 512), dtype=np.float32)  # Assuming 512 dim
        existing_image_paths = np.array([], dtype=str)
        processed_paths = set()
        print("Starting fresh embedding generation.")

    image_paths = list(existing_image_paths)
    embeddings = list(existing_embeddings)

    files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")]
    print(f"Found {len(files)} images to process.")

    new_count = 0
    # Process in batches for memory efficiency
    for file in tqdm(files, desc="Generating embeddings"):
        rel_path = os.path.join("data", "fashion-dataset", "images", file)
        if rel_path in processed_paths:
            continue  # Skip already processed
        path = os.path.join(IMAGE_FOLDER, file)
        try:
            image = Image.open(path).convert("RGB")
            embedding = generate_image_embedding(image, model, processor, device=device)
            embeddings.append(embedding)
            image_paths.append(rel_path)
            new_count += 1
            # Write progress every 1000 new images
            if new_count % 1000 == 0:
                with open(os.path.join(DATA_DIR, "progress.txt"), "w") as f:
                    f.write(f"Processed: {len(embeddings)} total embeddings\n")
        except Exception as e:
            print(f"Skipping {file}: {e}")

    embeddings = np.array(embeddings).astype("float32")
    image_paths = np.array(image_paths)
    print(f"\nTotal indexed: {len(embeddings)} images, shape: {embeddings.shape} (added {new_count} new)")

    # L2 normalize all embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build FAISS index (Inner Product on normalized vectors = cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    # Save everything
    np.save(embeddings_path, embeddings)
    np.save(image_paths_path, image_paths)
    faiss.write_index(index, os.path.join(DATA_DIR, "faiss_index.index"))

    print("✅ All files saved successfully.")


if __name__ == "__main__":
    main()