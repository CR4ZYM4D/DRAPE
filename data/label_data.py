# library imports
import os
import sys
import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# utility imports
from dotenv import load_dotenv
from logger.logger import logging
from exception.exception import ProjectError
from config.attribute_schema import (TOTAL_DIMS, SCHEMA_SLICES, ATTR_TO_IDX, CONTEXTS, STYLES)
from indexer.embed_image import get_embedder

# load datastore path
load_dotenv()
DATASTORE_PATH = os.getenv("DATASTORE_PATH", "data/raw/images")

# Approximate RGB values for the 15 colors in our schema
COLOR_RGB_MAP = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 128, 0),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "indigo": (75, 0, 130),
    "pink": (255, 192, 203),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "grey": (128, 128, 128),
    "brown": (165, 42, 42),
    "beige": (245, 245, 220),
    "teal": (0, 128, 128),
    "burgundy": (128, 0, 32),
}

# context text prompts for SigLIP zero-shot classification.
# garment categories can't reliably predict location, so we classify the actual image pixels instead.

CONTEXT_PROMPTS = {
    "office": "a photo taken inside an office",
    "park": "a photo taken in a park",
    "street": "a photo taken on an urban street",
    "home": "a photo taken at home",
    "gym": "a photo taken inside a gym",
    "beach": "a photo taken at the beach",
    "restaurant": "a photo taken inside a restaurant",
}

# style text prompts for SigLIP zero-shot classification.
# same reasoning as CONTEXTS:
# aesthetic/vibe judgments that don't map
# cleanly onto Fashionpedia's structural garment categories.

STYLE_PROMPTS = {
    "minimalist": "a photo of a minimalist, clean, understated outfit",
    "tailored": "a photo of a tailored, structured outfit",
    "sporty": "a photo of a sporty, athletic outfit",
    "relaxed": "a photo of a relaxed, casual, loose-fitting outfit",
    "dramatic": "a photo of a dramatic, statement outfit",
    "bold": "a photo of a bold, vibrant, eye-catching outfit",
    "elegant": "a photo of an elegant, refined outfit",
}

CONTEXT_SIM_THRESHOLD = 0.15
STYLE_SIM_THRESHOLD = 0.15

def get_closest_color(rgb_val):
    """Finds the closest color name in COLOR_RGB_MAP using Euclidean distance."""
    min_dist = float("inf")
    closest_color = None
    for color_name, color_rgb in COLOR_RGB_MAP.items():
        dist = np.linalg.norm(np.array(rgb_val) - np.array(color_rgb))
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name
    return closest_color


def extract_dominant_colors(image_path, bbox, k=2):
    """
    Crops image to a SINGLE bbox, runs K-Means, returns top-k color names.
    Called once per garment instance from process_labels().
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.warning(f"Error opening image {image_path}: {e}")
        return []

    if bbox:
        
        try:
            # handle list of bboxes or a single bbox
            box = bbox[0] if isinstance(bbox[0], list) else bbox
            # Make sure it's valid coordinates
            if len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
                img = img.crop((box[0], box[1], box[2], box[3]))
        except Exception as e:
            logging.warning(f"Warning: Failed to crop {image_path} with bbox {bbox}: {e}")
            pass
    
    # crop to 100x100 for not spending eternity in K means
    img.thumbnail((100, 100))

    # reshape img array from 100x100x3 to 10000x3
    pixels = np.array(img).reshape(-1, 3)

    # lesser pixels  than colors
    if len(pixels) < k:
        return []

    # use k-means to find the most prominent colors
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    dominant_colors = []
    for center in kmeans.cluster_centers_:
        color_name = get_closest_color(center)
        if color_name not in dominant_colors:
            dominant_colors.append(color_name)

    return dominant_colors


def _assign_via_siglip(image_path, embedder, label_names, text_embeds, sim_threshold, top_k=2):
    """
    Shared zero-shot classification routine — used for both CONTEXTS and
    STYLES. Classifies the actual image against a fixed set of text prompts
    via SigLIP cosine similarity, returns up to top_k labels above threshold.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.warning(f"Error opening image {image_path} for zero-shot classification: {e}")
        return []

    # get the siglip embeddings of the image
    image_emb = np.array(embedder.embed_image(img))
    # matrix mutliply to get cos similarity
    sims = text_embeds @ image_emb

    # find k most strong matches
    ranked_idx = np.argsort(sims)[::-1]
    labels_found = [label_names[idx] for idx in ranked_idx[:top_k] if sims[idx] >= sim_threshold]

    if not labels_found:
        # fallback: take the single best match even if below threshold so no
        # image is left with zero signal on this attribute group
        labels_found.append(label_names[ranked_idx[0]])

    return labels_found


def process_labels(annotations_file="data/raw/annotations.json", output_file="data/labels.npy"):
    """Reads annotations, extracts colors, synthesizes context/style, and saves 75-dim vectors."""
    try:
        if not os.path.exists(annotations_file):
            logging.error(f"Error: {annotations_file} not found. Run download_dataset.py first.")
            return

        with open(annotations_file, "r") as f:
            annotations = json.load(f)

        logging.info(f" ----- Processing {len(annotations)} annotations ----- ")

        # Pre-compute SigLIP text embeddings for context + style prompts once,
        # reused across every image instead of recomputing per-image.
        embedder = get_embedder()
        context_text_embeds = np.array(
            embedder.embed_text([CONTEXT_PROMPTS[c] for c in CONTEXTS])
        )
        style_text_embeds = np.array(
            embedder.embed_text([STYLE_PROMPTS[s] for s in STYLES])
        )

        all_labels = {}

        # loop through donwloaded images
        for idx, (image_id, data) in enumerate(annotations.items()):
            image_path = data.get("image_path")
            boxes = data.get("boxes", [])
            categories = data.get("categories", [])

            # init target vector
            label_vec = np.zeros(TOTAL_DIMS, dtype=np.float32)

            # label garment types as is from fashionpedia labels
            for cat_id in categories:
                if isinstance(cat_id, int) and cat_id < SCHEMA_SLICES["garment_type"][1]:
                    label_vec[cat_id] = 1.0

            # label colors from k means clusters
            for box in boxes:
                dominant_colors = extract_dominant_colors(image_path, box)
                for color in dominant_colors:
                    if color in ATTR_TO_IDX:
                        label_vec[ATTR_TO_IDX[color]] = 1.0

            # label context fronm siglip embeddings
            contexts_found = _assign_via_siglip(
                image_path, embedder, CONTEXTS, context_text_embeds, CONTEXT_SIM_THRESHOLD
            )
            for c in contexts_found:
                if c in ATTR_TO_IDX:
                    label_vec[ATTR_TO_IDX[c]] = 1.0

            # label style like context
            styles_found = _assign_via_siglip(
                image_path, embedder, STYLES, style_text_embeds, STYLE_SIM_THRESHOLD
            )
            for s in styles_found:
                if s in ATTR_TO_IDX:
                    label_vec[ATTR_TO_IDX[s]] = 1.0

            all_labels[image_id] = label_vec

            if (idx + 1) % 100 == 0:
                logging.info(f"Processed {idx + 1} labels.")
                
        # save as numpy dict
        np.save(output_file, all_labels)
        logging.info(f" ----- Saved labels to {output_file} ----- ")
    
    except Exception as e:
        logging.error(f" ----- Error in process_labels: {str(e)} -----")
        raise ProjectError(str(e), sys)


if __name__ == "__main__":
    process_labels()