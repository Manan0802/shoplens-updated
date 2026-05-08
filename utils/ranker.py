"""
Multi-Signal Re-Ranker for ShopLens.
Takes FAISS candidates and re-ranks them using metadata signals
(color, gender, usage, season, category) combined with CLIP similarity.
"""
import numpy as np


# ─────────────────────────────────────────────────
# Color distance using simple RGB mapping
# ─────────────────────────────────────────────────
# Pre-defined color RGB centroids for perceptual matching
COLOR_RGB = {
    "black":      (20, 20, 20),
    "white":      (240, 240, 240),
    "grey":       (150, 150, 150),
    "red":        (200, 40, 40),
    "blue":       (40, 60, 180),
    "navy blue":  (20, 30, 100),
    "green":      (40, 160, 60),
    "brown":      (140, 90, 50),
    "pink":       (230, 130, 160),
    "purple":     (130, 50, 160),
    "yellow":     (230, 210, 50),
    "orange":     (230, 140, 30),
    "beige":      (220, 200, 170),
    "maroon":     (120, 20, 40),
    "gold":       (200, 180, 60),
    "silver":     (190, 190, 200),
    "teal":       (40, 160, 160),
    "olive":      (130, 140, 60),
    "cream":      (240, 230, 210),
    "charcoal":   (60, 60, 60),
    "magenta":    (200, 40, 140),
    "peach":      (240, 190, 160),
    "khaki":      (180, 170, 120),
    "lavender":   (180, 160, 220),
    "rust":       (180, 70, 30),
    "turquoise blue": (40, 190, 200),
    "coral":      (230, 120, 100),
    "tan":        (200, 170, 120),
    "nude":       (220, 195, 170),
    "off white":  (235, 230, 220),
    "multi":      (128, 128, 128),  # neutral
    "lime green": (80, 200, 50),
    "fluorescent green": (60, 240, 60),
    "sea green":  (60, 180, 120),
    "mauve":      (180, 130, 170),
    "mushroom brown": (160, 140, 115),
    "burgundy":   (100, 20, 40),
    "taupe":      (160, 140, 120),
    "steel":      (140, 150, 165),
    "coffee brown": (110, 70, 40),
    "copper":     (180, 110, 60),
    "rose":       (220, 140, 160),
    "skin":       (210, 175, 150),
    "grey melange": (170, 170, 175),
    "metallic":   (160, 165, 170),
}


def _color_distance(rgb1, rgb2):
    """Simple Euclidean distance in RGB space."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))


def color_similarity_score(query_color_name, candidate_color_name):
    """
    Returns a score 0.0-1.0 where 1.0 = identical color, 0.0 = max distance.
    """
    q = query_color_name.lower().strip() if query_color_name else ""
    c = candidate_color_name.lower().strip() if candidate_color_name else ""

    # Exact match
    if q == c:
        return 1.0

    # If either is unknown, neutral score
    if not q or not c or q == "other" or c == "other":
        return 0.5

    q_rgb = COLOR_RGB.get(q)
    c_rgb = COLOR_RGB.get(c)

    if q_rgb is None or c_rgb is None:
        # Partial string match fallback
        if q in c or c in q:
            return 0.7
        return 0.3

    dist = _color_distance(q_rgb, c_rgb)
    # Max possible distance in RGB = sqrt(3 * 255^2) ≈ 441
    score = max(0.0, 1.0 - (dist / 300.0))
    return score


def category_match_score(query_article, candidate_article):
    """
    Returns 1.0 for same article type, 0.5 for same family, 0.0 for different.
    """
    if not query_article or not candidate_article:
        return 0.3

    q = query_article.lower().strip()
    c = candidate_article.lower().strip()

    if q == c:
        return 1.0

    # Define article families
    families = [
        {"tshirts", "shirts", "tops", "casual shirts", "formal shirts", "kurtas", "tunics"},
        {"jeans", "trousers", "track pants", "shorts", "capris", "jeggings", "leggings"},
        {"casual shoes", "sports shoes", "formal shoes", "sneakers", "flats", "heels"},
        {"sandals", "flip flops", "sports sandals"},
        {"watches"},
        {"belts"},
        {"backpacks", "handbags", "clutches", "laptop bag", "messenger bag", "wallets"},
        {"jackets", "blazers", "waistcoat", "sweaters", "sweatshirts"},
        {"sunglasses"},
        {"kurtas", "kurta sets", "salwar", "sarees"},
        {"briefs", "trunk", "boxers"},
        {"socks"},
    ]

    for family in families:
        if q in family and c in family:
            return 0.6

    return 0.0


def binary_match(val1, val2):
    """Returns 1.0 if match, 0.0 if mismatch, 0.5 if either is missing."""
    if not val1 or not val2:
        return 0.5
    return 1.0 if str(val1).lower().strip() == str(val2).lower().strip() else 0.0


def compute_match_score(query_meta, candidate_meta, faiss_similarity):
    """
    Multi-signal composite score.

    query_meta / candidate_meta = {
        "articleType": str,
        "baseColour": str,
        "gender": str,
        "usage": str,
        "season": str,
    }

    faiss_similarity: float (cosine similarity from FAISS, already 0-1 range)

    Returns: float (0.0 - 1.0), higher = better match
    """
    # CLIP visual similarity (the core signal)
    clip_score = max(0.0, min(1.0, faiss_similarity))

    # Category match
    cat_score = category_match_score(
        query_meta.get("articleType", ""),
        candidate_meta.get("articleType", "")
    )

    # Color similarity
    col_score = color_similarity_score(
        query_meta.get("baseColour", ""),
        candidate_meta.get("baseColour", "")
    )

    # Gender match
    gender_score = binary_match(
        query_meta.get("gender", ""),
        candidate_meta.get("gender", "")
    )

    # Usage match (casual/formal/sports/ethnic)
    usage_score = binary_match(
        query_meta.get("usage", ""),
        candidate_meta.get("usage", "")
    )

    # Season match
    season_score = binary_match(
        query_meta.get("season", ""),
        candidate_meta.get("season", "")
    )

    # Weighted combination
    final = (
        0.40 * clip_score     +
        0.20 * cat_score      +
        0.15 * col_score      +
        0.10 * gender_score   +
        0.10 * usage_score    +
        0.05 * season_score
    )

    return final


def rerank_results(raw_indices, raw_distances, query_meta, meta_lookup, image_ids):
    """
    Takes raw FAISS results and re-ranks them using multi-signal scoring.

    Returns: list of (index, score) sorted by score descending.
    """
    scored = []

    for rank, idx in enumerate(raw_indices):
        try:
            iid = int(image_ids[idx])
            if iid == -1:
                continue

            row = meta_lookup.get(iid)
            if row is None:
                continue

            candidate_meta = {
                "articleType": str(row.get("articleType", "")),
                "baseColour":  str(row.get("baseColour", "")),
                "gender":      str(row.get("gender", "")),
                "usage":       str(row.get("usage", "")),
                "season":      str(row.get("season", "")),
            }

            faiss_sim = float(raw_distances[rank]) if rank < len(raw_distances) else 0.5

            score = compute_match_score(query_meta, candidate_meta, faiss_sim)
            scored.append((idx, score))

        except Exception:
            continue

    # Sort by composite score, descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
