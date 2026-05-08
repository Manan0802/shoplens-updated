"""
ShopLens — AI-Powered Fashion Search Engine
Phase 1: Full 44K index + bug fixes
Phase 2: Multi-signal re-ranking
Phase 3: Expanded classification + smart prompts + style-coherent outfits
"""
import os
import json
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

from models.clip_model import (
    load_clip_model,
    generate_image_embedding,
    generate_text_embedding,
    classify_region,
    build_smart_prompt,
    FASHION_LABELS,
)
from utils.faiss_utils import load_faiss_index, search_similar, search_similar_with_scores
from utils.metadata_utils import load_metadata, get_product_name
from utils.yolo_utils import detect_fashion_regions
from utils.ranker import rerank_results, color_similarity_score


# ─────────────────────────────────────────────────
# Paths & Config
# ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAV_FILE = os.path.join(BASE_DIR, "favorites.json")


# ─────────────────────────────────────────────────
# Favorites Persistence
# ─────────────────────────────────────────────────
def load_favorites():
    if not os.path.exists(FAV_FILE):
        return []
    try:
        with open(FAV_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_favorites(favs):
    try:
        with open(FAV_FILE, "w") as f:
            json.dump(favs, f)
    except Exception:
        pass


# ─────────────────────────────────────────────────
# Dominant Color Extraction (K-Means on center pixels)
# ─────────────────────────────────────────────────
def get_dominant_color(pil_image, n_colors=3):
    """
    Extract the dominant color using K-Means clustering on center-cropped pixels.
    Returns the closest named color from the dataset palette.
    """
    # Center-crop to focus on the garment, not background
    w, h = pil_image.size
    margin_x, margin_y = int(w * 0.2), int(h * 0.15)
    center = pil_image.crop((margin_x, margin_y, w - margin_x, h - margin_y))
    
    img = np.array(center.resize((80, 80)))
    pixels = img.reshape(-1, 3).astype(float)

    # Remove near-white and near-black pixels (likely background)
    mask = (pixels.sum(axis=1) > 60) & (pixels.sum(axis=1) < 700)
    pixels = pixels[mask]

    if len(pixels) < 10:
        return "Other"

    try:
        kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), n_init=3, random_state=42)
        kmeans.fit(pixels)
        # Dominant = largest cluster
        counts = np.bincount(kmeans.labels_)
        dominant_rgb = kmeans.cluster_centers_[counts.argmax()]
    except Exception:
        dominant_rgb = pixels.mean(axis=0)

    # Map to nearest named color
    return _rgb_to_color_name(dominant_rgb)


# Named color centroids (matches styles.csv baseColour values)
_COLOR_MAP = {
    "Black":     (30, 30, 30),
    "White":     (235, 235, 235),
    "Grey":      (150, 150, 150),
    "Red":       (200, 40, 40),
    "Blue":      (40, 70, 180),
    "Navy Blue": (20, 30, 100),
    "Green":     (40, 160, 60),
    "Brown":     (140, 90, 50),
    "Pink":      (230, 130, 160),
    "Purple":    (130, 50, 160),
    "Yellow":    (230, 210, 50),
    "Orange":    (230, 140, 30),
    "Beige":     (220, 200, 170),
    "Maroon":    (120, 20, 40),
    "Gold":      (200, 180, 60),
    "Silver":    (190, 190, 200),
    "Teal":      (40, 160, 160),
    "Olive":     (130, 140, 60),
    "Cream":     (240, 230, 210),
}


def _rgb_to_color_name(rgb):
    """Find the nearest named color for a given RGB value."""
    best_name, best_dist = "Other", float("inf")
    for name, centroid in _COLOR_MAP.items():
        dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, centroid)))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    # If the closest named color is still very far, call it "Other"
    return best_name if best_dist < 120 else "Other"


# ─────────────────────────────────────────────────
# Category & Outfit Maps (expanded)
# ─────────────────────────────────────────────────
CATEGORY_MAP = {
    "shirt":        ["shirts", "casual shirts", "formal shirts"],
    "casual shirt": ["shirts", "casual shirts"],
    "formal shirt": ["shirts", "formal shirts"],
    "t-shirt":      ["tshirts"],
    "top":          ["tops", "tshirts"],
    "jeans":        ["jeans", "jeggings"],
    "trousers":     ["trousers", "formal trousers"],
    "pants":        ["trousers", "track pants", "jeans"],
    "shorts":       ["shorts"],
    "track pants":  ["track pants"],
    "leggings":     ["leggings", "jeggings"],
    "shoes":        ["casual shoes", "formal shoes", "sports shoes"],
    "formal shoes": ["formal shoes"],
    "sports shoes": ["sports shoes"],
    "sneakers":     ["casual shoes", "sports shoes"],
    "sandals":      ["sandals", "sports sandals"],
    "heels":        ["heels"],
    "flip flops":   ["flip flops"],
    "boots":        ["boots"],
    "watch":        ["watches"],
    "belt":         ["belts"],
    "bag":          ["handbags", "clutches", "laptop bag", "messenger bag"],
    "handbag":      ["handbags"],
    "backpack":     ["backpacks"],
    "wallet":       ["wallets"],
    "sunglasses":   ["sunglasses"],
    "jacket":       ["jackets"],
    "blazer":       ["blazers"],
    "sweater":      ["sweaters"],
    "sweatshirt":   ["sweatshirts"],
    "kurta":        ["kurtas", "kurta sets"],
    "saree":        ["sarees"],
    "dress":        ["dresses"],
    "scarf":        ["scarves", "stoles"],
}

OUTFIT_MAP = {
    "shirt":        ["jeans", "trousers", "shoes", "belt", "watch"],
    "casual shirt": ["jeans", "sneakers", "watch"],
    "formal shirt": ["trousers", "formal shoes", "belt", "watch"],
    "t-shirt":      ["jeans", "shorts", "sneakers", "watch", "backpack"],
    "top":          ["jeans", "shorts", "heels", "bag"],
    "jeans":        ["shirt", "t-shirt", "shoes", "belt", "jacket"],
    "trousers":     ["shirt", "formal shoes", "belt", "watch", "blazer"],
    "shorts":       ["t-shirt", "sneakers", "sandals"],
    "shoes":        ["jeans", "belt", "watch"],
    "sneakers":     ["jeans", "shorts", "t-shirt", "backpack"],
    "sandals":      ["kurta", "shorts", "dress"],
    "heels":        ["dress", "bag", "watch"],
    "jacket":       ["jeans", "t-shirt", "shoes", "belt"],
    "blazer":       ["trousers", "formal shoes", "watch"],
    "kurta":        ["trousers", "sandals", "watch"],
    "dress":        ["heels", "bag", "watch"],
    "watch":        ["shirt", "jeans"],
    "belt":         ["jeans", "trousers", "shirt"],
    "bag":          ["dress", "heels"],
    "sunglasses":   ["t-shirt", "jeans"],
}


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────
def generate_links(product_name):
    q = product_name.replace(" ", "+")
    return {
        "Amazon":   f"https://www.amazon.in/s?k={q}",
        "Flipkart": f"https://www.flipkart.com/search?q={q}",
        "Myntra":   f"https://www.myntra.com/{q}",
        "Ajio":     f"https://www.ajio.com/search/?text={q}",
    }


def apply_filters(df, gender, category, color, usage):
    fdf = df.copy()
    if gender   != "All": fdf = fdf[fdf["gender"]         == gender]
    if category != "All": fdf = fdf[fdf["masterCategory"] == category]
    if color    != "All": fdf = fdf[fdf["baseColour"]     == color]
    if usage    != "All": fdf = fdf[fdf["usage"]          == usage]
    return fdf


def get_product_name_fast(lookup, path):
    try:
        iid = int(os.path.basename(path).split(".")[0])
        row = lookup.get(iid)
        if row is not None:
            return str(row.get("productDisplayName", "Unknown Product"))
    except Exception:
        pass
    return "Unknown Product"


def render_product_card(col, image_path, product_name, favorites_list, section_key):
    """Render a product card with image, name, save button, and shopping links."""
    try:
        pid = int(os.path.basename(image_path).split(".")[0])
    except Exception:
        pid = 0

    with col:
        st.image(str(image_path), use_container_width=True)
        short = product_name[:35] + "…" if len(product_name) > 35 else product_name
        st.caption(f"**{short}**")

        is_saved = pid in favorites_list
        if st.button("❤️ Saved" if is_saved else "🤍 Save", key=f"fav_{section_key}_{pid}"):
            if is_saved:
                while pid in favorites_list:
                    favorites_list.remove(pid)
            else:
                favorites_list.append(pid)
            save_favorites(favorites_list)
            st.rerun()

        links = generate_links(product_name)
        st.markdown(
            f'<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:4px;">'
            f'<a href="{links["Amazon"]}" target="_blank"><button style="font-size:10px;padding:2px 5px;cursor:pointer;">🛒 Amazon</button></a>'
            f'<a href="{links["Flipkart"]}" target="_blank"><button style="font-size:10px;padding:2px 5px;cursor:pointer;">🛍 Flipkart</button></a>'
            f'<a href="{links["Myntra"]}" target="_blank"><button style="font-size:10px;padding:2px 5px;cursor:pointer;">👕 Myntra</button></a>'
            f'<a href="{links["Ajio"]}" target="_blank"><button style="font-size:10px;padding:2px 5px;cursor:pointer;">👜 Ajio</button></a>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────
# Cached Resource Loading
# ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    pack = load_clip_model()
    return pack[0], pack[1], pack[2]


@st.cache_resource
def load_search_data():
    index = load_faiss_index(os.path.join(DATA_DIR, "faiss_index.index"))
    raw_paths = np.load(os.path.join(DATA_DIR, "image_paths.npy"), allow_pickle=True)
    abs_paths, ids = [], []
    for p in raw_paths:
        ap = p if os.path.isabs(p) else os.path.join(BASE_DIR, p)
        abs_paths.append(ap)
        try:
            ids.append(int(os.path.basename(ap).split(".")[0]))
        except Exception:
            ids.append(-1)
    return index, np.array(abs_paths), np.array(ids, dtype=np.int64)


@st.cache_resource
def load_metadata_cached():
    return load_metadata(os.path.join(DATA_DIR, "fashion-dataset", "styles.csv"))


@st.cache_resource
def build_lookup(_df_id):
    df = load_metadata_cached()
    lookup = {}
    for _, row in df.iterrows():
        try:
            lookup[int(row["id"])] = row
        except Exception:
            pass
    return lookup


# ─────────────────────────────────────────────────
# Initialize
# ─────────────────────────────────────────────────
model, processor, device = load_model()
index, image_paths, image_ids = load_search_data()
metadata_df  = load_metadata_cached()
meta_lookup  = build_lookup(id(metadata_df))
favorites    = load_favorites()


# ─────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────
st.set_page_config(page_title="ShopLens", page_icon="🛒", layout="wide")
st.markdown("<h1 style='text-align:center;'>🛒 ShopLens</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:gray;'>AI-Powered Fashion Search · "
    f"{index.ntotal:,} products indexed</p>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload a fashion image", type=["jpg", "png", "jpeg"])
text_query    = st.text_input("Or describe what you're looking for (e.g. 'red formal shirt')")
weight        = st.slider("Text influence on search", 0.0, 1.0, 0.3)


# ─────────────────────────────────────────────────
# State Defaults
# ─────────────────────────────────────────────────
embedding     = None
detections    = []
det           = {}
cropped_image = None
query_label   = None
image_color   = "Other"
query_meta    = {}


# ─────────────────────────────────────────────────
# Image Upload Path
# ─────────────────────────────────────────────────
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="Uploaded Image", width=300)

    # Color detection (K-Means based)
    image_color = get_dominant_color(pil_image)
    st.info(f"🎨 Detected Color: **{image_color}**")

    # YOLO detection
    with st.spinner("Detecting fashion items…"):
        detections = detect_fashion_regions(np.array(pil_image))

    # CLIP classification of each region
    if detections:
        with st.spinner("AI classifying regions…"):
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                crop = pil_image.crop((x1, y1, x2, y2))
                lbl = classify_region(crop, model, processor)
                if lbl and lbl != "unknown":
                    d["label"] = lbl
            detections = [d for d in detections if d.get("label") and d["label"] != "unknown"]

    if detections:
        st.subheader("🎯 Detected Fashion Items")
        selected_idx = st.selectbox(
            "Select item to search:",
            range(len(detections)),
            format_func=lambda x: f"{x+1}. {detections[x]['label'].capitalize()}"
        )
        det = detections[selected_idx]
        query_label = det["label"].lower()
        x1, y1, x2, y2 = det["bbox"]
        cropped_image = pil_image.crop((x1, y1, x2, y2))
        st.info(f"Searching for: **{query_label}**")
        st.image(cropped_image, caption=f"Focused: {query_label}", width=200)

        # Build query metadata for re-ranker
        query_meta = {
            "articleType": query_label,
            "baseColour":  image_color,
            "gender":      "",
            "usage":       "",
            "season":      "",
        }
    else:
        st.info("No clear fashion items detected — searching the full image.")
        cropped_image = pil_image

    # Build embedding
    img_emb = generate_image_embedding(cropped_image, model, processor, device=device)

    if text_query.strip():
        # Phase 3: Smart prompt construction
        smart_prompt = build_smart_prompt(query_label, text_query)
        txt_emb = generate_text_embedding(smart_prompt, model, processor, device=device)

        img_norm = np.linalg.norm(img_emb)
        txt_norm = np.linalg.norm(txt_emb)
        if img_norm > 0: img_emb = img_emb / img_norm
        if txt_norm > 0: txt_emb = txt_emb / txt_norm

        embedding = (1 - weight) * img_emb + weight * txt_emb
    else:
        embedding = img_emb

# ─────────────────────────────────────────────────
# Text-Only Path
# ─────────────────────────────────────────────────
elif text_query.strip():
    with st.spinner("Searching…"):
        smart_prompt = build_smart_prompt(None, text_query)
        embedding = generate_text_embedding(smart_prompt, model, processor, device=device)

# ─────────────────────────────────────────────────
# Search + Results
# ─────────────────────────────────────────────────
if embedding is not None:
    # Phase 2: Get candidates WITH scores for re-ranking
    raw_indices, raw_distances = search_similar_with_scores(index, embedding, k=200)

    st.markdown("---")

    # Manual filters
    st.subheader("🔍 Refine Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1: sel_gender   = st.selectbox("Gender",   ["All", "Men", "Women", "Boys", "Girls", "Unisex"])
    with c2: sel_category = st.selectbox("Category", ["All"] + sorted(metadata_df["masterCategory"].dropna().unique().tolist()))
    with c3: sel_color    = st.selectbox("Color",    ["All"] + sorted(metadata_df["baseColour"].dropna().unique().tolist()))
    with c4: sel_usage    = st.selectbox("Usage",    ["All"] + sorted(metadata_df["usage"].dropna().unique().tolist()))

    manual_valid_ids = set(
        apply_filters(metadata_df, sel_gender, sel_category, sel_color, sel_usage)["id"].astype(int).values
    )

    # Phase 2: Multi-signal re-ranking
    # First filter by manual selections, then re-rank
    pre_filtered_indices = []
    pre_filtered_distances = []

    allowed_kws = CATEGORY_MAP.get(query_label, []) if query_label else []

    for rank, idx in enumerate(raw_indices):
        try:
            iid = int(image_ids[idx])
            if iid == -1 or iid not in manual_valid_ids:
                continue
            # Category filter (soft — only if we have a query_label)
            if allowed_kws:
                row = meta_lookup.get(iid)
                if row:
                    article = str(row.get("articleType", "")).lower()
                    if not any(kw in article for kw in allowed_kws):
                        continue
            pre_filtered_indices.append(idx)
            pre_filtered_distances.append(float(raw_distances[rank]))
        except Exception:
            continue

    # Fallback: relax category filter
    if len(pre_filtered_indices) < 5:
        pre_filtered_indices = []
        pre_filtered_distances = []
        for rank, idx in enumerate(raw_indices):
            try:
                iid = int(image_ids[idx])
                if iid != -1 and iid in manual_valid_ids:
                    pre_filtered_indices.append(idx)
                    pre_filtered_distances.append(float(raw_distances[rank]))
            except Exception:
                continue

    # Re-rank using multi-signal scoring
    if query_meta and pre_filtered_indices:
        ranked = rerank_results(
            pre_filtered_indices, pre_filtered_distances,
            query_meta, meta_lookup, image_ids
        )
        final_indices = [idx for idx, score in ranked]
    else:
        final_indices = pre_filtered_indices

    # Display Similar Products
    st.subheader("🛍️ Similar Products")
    display = final_indices[:10]

    if not display:
        st.warning("No results found 😢 — try relaxing your filters.")
    else:
        for row_start in range(0, len(display), 5):
            batch = display[row_start: row_start + 5]
            cols = st.columns(5)
            for i, idx in enumerate(batch):
                pname = get_product_name_fast(meta_lookup, image_paths[idx])
                render_product_card(cols[i], image_paths[idx], pname, favorites, f"sim_{row_start + i}")

    # ── Complete the Look (Style-Coherent) ───────
    all_detected_labels = list({d["label"] for d in detections if d.get("label")})

    if all_detected_labels:
        outfit_pool_indices, outfit_pool_dists = search_similar_with_scores(index, embedding, k=500)

        # Phase 3: Style-coherent outfit filtering
        # Respect gender + usage of the query item
        query_gender = ""
        query_usage  = ""
        if query_meta.get("gender"):
            query_gender = query_meta["gender"]
        elif sel_gender != "All":
            query_gender = sel_gender

        if query_meta.get("usage"):
            query_usage = query_meta["usage"]
        elif sel_usage != "All":
            query_usage = sel_usage

        outfit_sections = {}
        for label in all_detected_labels:
            for cat in OUTFIT_MAP.get(label, []):
                key = f"{label.capitalize()} → {cat.capitalize()}"
                if key in outfit_sections:
                    continue

                cat_kws = CATEGORY_MAP.get(cat, [cat])
                recs = []

                for rank, idx in enumerate(outfit_pool_indices):
                    try:
                        iid = int(image_ids[idx])
                        if iid == -1:
                            continue
                        row = meta_lookup.get(iid)
                        if not row:
                            continue

                        article = str(row.get("articleType", "")).lower()
                        if not any(kw in article for kw in cat_kws):
                            continue

                        # Style coherence: filter by gender
                        if query_gender:
                            item_gender = str(row.get("gender", ""))
                            if item_gender and item_gender != query_gender and item_gender != "Unisex":
                                continue

                        # Style coherence: filter by usage
                        if query_usage:
                            item_usage = str(row.get("usage", ""))
                            if item_usage and item_usage != query_usage:
                                continue

                        recs.append(idx)
                        if len(recs) >= 5:
                            break
                    except Exception:
                        continue

                # Relax usage filter if too few results
                if len(recs) < 3:
                    recs = []
                    for rank, idx in enumerate(outfit_pool_indices):
                        try:
                            iid = int(image_ids[idx])
                            if iid == -1:
                                continue
                            row = meta_lookup.get(iid)
                            if not row:
                                continue
                            article = str(row.get("articleType", "")).lower()
                            if not any(kw in article for kw in cat_kws):
                                continue
                            if query_gender:
                                item_gender = str(row.get("gender", ""))
                                if item_gender and item_gender != query_gender and item_gender != "Unisex":
                                    continue
                            recs.append(idx)
                            if len(recs) >= 5:
                                break
                        except Exception:
                            continue

                if recs:
                    outfit_sections[key] = recs

        if outfit_sections:
            st.markdown("---")
            st.subheader("👗 Complete the Look")
            counter = 0
            for key, idx_list in outfit_sections.items():
                st.markdown(f"##### {key}")
                cols = st.columns(min(len(idx_list), 5))
                for i, idx in enumerate(idx_list[:5]):
                    pname = get_product_name_fast(meta_lookup, image_paths[idx])
                    render_product_card(cols[i], image_paths[idx], pname, favorites, f"outfit_{counter}")
                    counter += 1


# ─────────────────────────────────────────────────
# Favorites Section
# ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("❤️ Your Saved Items")

if not favorites:
    st.info("No saved items yet. Click 'Save' on products you love!")
else:
    valid_favs = []
    for fid in favorites:
        path = os.path.join(DATA_DIR, "fashion-dataset", "images", f"{fid}.jpg")
        if os.path.exists(path):
            valid_favs.append((fid, path))

    if not valid_favs:
        st.info("Your saved items are currently unavailable.")
    else:
        for row_start in range(0, len(valid_favs), 5):
            batch = valid_favs[row_start: row_start + 5]
            cols = st.columns(5)
            for i, (fid, path) in enumerate(batch):
                pname = get_product_name_fast(meta_lookup, path)
                render_product_card(cols[i], path, pname, favorites, f"favs_{row_start + i}")