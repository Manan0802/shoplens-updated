"""
CLIP model utilities for ShopLens.
- Image/text embedding generation
- Zero-shot fashion classification with expanded label set
- Smart prompt construction for better text search
"""
import torch
from transformers import CLIPProcessor, CLIPModel


def load_clip_model(device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device


def _tensor_to_numpy(model_output):
    # CLIP text/image features may come back as raw tensors, or as a ModelOutput wrapper.
    if hasattr(model_output, "image_embeds"):
        tensor = model_output.image_embeds
    elif hasattr(model_output, "text_embeds"):
        tensor = model_output.text_embeds
    elif hasattr(model_output, "pooler_output"):
        tensor = model_output.pooler_output
    elif isinstance(model_output, torch.Tensor):
        tensor = model_output
    else:
        raise ValueError("Unexpected model output type, cannot extract embedding")

    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor.numpy()


def generate_image_embedding(image, model, processor, device=None):
    device = device or next(model.parameters()).device
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    embedding = _tensor_to_numpy(image_features)[0]
    return embedding


def generate_text_embedding(text, model, processor, device=None):
    device = device or next(model.parameters()).device
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    embedding = _tensor_to_numpy(text_features)[0]
    return embedding


def build_smart_prompt(query_label, user_text):
    """
    Construct a CLIP-optimized text prompt that gives much better embeddings
    than raw user text alone.
    
    Examples:
        build_smart_prompt("shirt", "red formal") → "a photo of a red formal shirt"
        build_smart_prompt(None, "blue jeans")     → "a photo of blue jeans"
        build_smart_prompt("shoes", "")            → "a photo of shoes"
    """
    parts = []
    if user_text and user_text.strip():
        parts.append(user_text.strip())
    if query_label:
        parts.append(query_label)
    
    if parts:
        description = " ".join(parts)
        return f"a photo of {description}"
    return ""


# ─────────────────────────────────────────────────
# Expanded fashion labels for zero-shot classification
# Covers the top articleTypes in the dataset
# ─────────────────────────────────────────────────
FASHION_LABELS = [
    # Topwear
    "shirt", "t-shirt", "casual shirt", "formal shirt",
    "top", "kurta", "jacket", "blazer", "sweater", "sweatshirt",
    # Bottomwear
    "jeans", "trousers", "pants", "shorts", "track pants", "leggings",
    # Footwear
    "shoes", "sneakers", "sandals", "heels", "flip flops",
    "formal shoes", "sports shoes", "boots",
    # Accessories
    "watch", "belt", "bag", "handbag", "backpack", "wallet",
    "sunglasses", "scarf",
    # Ethnic
    "kurta", "saree", "dress",
]

# Deduplicate while preserving order
FASHION_LABELS = list(dict.fromkeys(FASHION_LABELS))


def classify_region(image, model, processor):
    """
    Classify a cropped fashion region using CLIP zero-shot classification.
    Uses an expanded label set covering all major fashion categories.
    Returns the best-matching label or None if confidence is too low.
    """
    # Use descriptive prompts for better CLIP accuracy
    prompt_labels = [f"a photo of a {label}" for label in FASHION_LABELS]

    inputs = processor(
        text=prompt_labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    best_idx = probs.argmax()
    best_score = probs[best_idx]

    # Confidence threshold
    if best_score < 0.15:
        return None

    return FASHION_LABELS[best_idx]