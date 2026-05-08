from pathlib import Path

import numpy as np
import faiss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
INDEX_PATH = DATA_DIR / "faiss_index.index"

# load embeddings
embeddings = np.load(EMBEDDINGS_PATH)

# convert to float32 (FAISS requirement)
embeddings = embeddings.astype("float32")

# get dimension
dimension = embeddings.shape[1]

# create index (cosine similarity)
index = faiss.IndexFlatIP(dimension)

# normalize vectors (important for cosine similarity)
faiss.normalize_L2(embeddings)

# add vectors to index
index.add(embeddings)

# save index
faiss.write_index(index, str(INDEX_PATH))

print(f"FAISS index built and saved to {INDEX_PATH}!")