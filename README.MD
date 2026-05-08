# 🛒 ShopLens — AI-Powered Fashion Search Engine

ShopLens is a state-of-the-art fashion discovery engine that combines visual similarity search (using CLIP) with intelligent metadata-aware re-ranking.

## 🚀 Quick Start

To run this project on your machine, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/Manan0802/shoplens-updated
cd shoplens-updated
```

### 2. Install Dependencies
Make sure you have Python 3.9+ installed. Install the required libraries using:
```bash
pip install -r requirements.txt
```

### 3. Run the App
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

---

## 📊 Current Dataset Status

The engine is currently pre-indexed and ready to search:
- **Total Embeddings:** 45,441 products
- **Model:** CLIP (ViT-B/32) + YOLOv8n
- **Signals:** Visual Similarity, Category Match, Color Consistency, Gender Filtering, and Usage Context.

---

## 🛠️ Project Structure

- `app.py`: Main Streamlit application.
- `data/`: Contains the FAISS index, embeddings, and metadata.
- `models/`: CLIP model utilities and logic.
- `utils/`: 
  - `faiss_utils.py`: Vector search logic.
  - `yolo_utils.py`: Fashion region detection.
  - `ranker.py`: Multi-signal re-ranking engine.
  - `metadata_utils.py`: CSV handling and lookup.
- `yolov8n.pt`: Pre-trained YOLOv8 model for object detection.
- `requirements.txt`: List of all required Python packages.

---

## 💡 Features

- **Visual Search:** Upload any fashion image to find similar items.
- **Smart Text Prompting:** Describe what you're looking for to refine the visual search.
- **Auto-Detection:** Automatically detects and isolates fashion items in a photo (e.g., searching for just the shoes in a full-body outfit).
- **Multi-Signal Re-ranking:** Results are sorted not just by how they look, but by matching color, category, and gender.
- **Complete the Look:** Recommends style-coherent matching items (e.g., "This shirt goes well with these jeans").
- **Shopping Links:** Instant links to Amazon, Flipkart, Myntra, and Ajio.
