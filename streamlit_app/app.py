import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import hnswlib
from ultralytics import YOLO
import cv2

# ------------------------------------------------
# Page Config
# ------------------------------------------------

st.set_page_config(
    page_title="Visual Product Search",
    layout="wide"
)

st.title("Visual Product Search Engine")

st.markdown("""
This application performs:
- YOLO-based product localization
- CLIP embedding generation
- HNSW approximate nearest neighbor retrieval
- Fashion similarity search on DeepFashion dataset
""")

# ------------------------------------------------
# Project Paths
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDINGS_DIR = BASE_DIR / "embeddings"

INDEX_DIR = BASE_DIR / "indexes"

DATASET_ROOT = (
    BASE_DIR
    / "data"
    / "DeepFashion"
    / "Img"
)

# ------------------------------------------------
# Load CLIP
# ------------------------------------------------

@st.cache_resource
def load_clip():

    model_name = "openai/clip-vit-base-patch32"

    model = CLIPModel.from_pretrained(
        model_name
    )

    processor = CLIPProcessor.from_pretrained(
        model_name
    )

    return model, processor


model, processor = load_clip()

st.success("CLIP loaded successfully")

# ------------------------------------------------
# Load YOLO
# ------------------------------------------------

@st.cache_resource
def load_yolo():

    model = YOLO("yolov8n.pt")

    return model


yolo_model = load_yolo()

st.success("YOLO loaded successfully")

# ------------------------------------------------
# Load Embeddings
# ------------------------------------------------

embeddings_path = (
    EMBEDDINGS_DIR
    / "gallery_embeddings_full.npy"
)

metadata_path = (
    EMBEDDINGS_DIR
    / "gallery_metadata_full.csv"
)

if not embeddings_path.exists():

    st.error(
        f"Embeddings file not found:\n{embeddings_path}"
    )

    st.stop()

if not metadata_path.exists():

    st.error(
        f"Metadata file not found:\n{metadata_path}"
    )

    st.stop()

embeddings = np.load(
    embeddings_path
)

metadata = pd.read_csv(
    metadata_path
)

st.success("Embeddings loaded successfully")

# ------------------------------------------------
# Load HNSW Index
# ------------------------------------------------

index_path = (
    INDEX_DIR
    / "fashion_hnsw_full.index"
)

if not index_path.exists():

    st.error(
        f"HNSW index not found:\n{index_path}"
    )

    st.stop()

dim = embeddings.shape[1]

index = hnswlib.Index(
    space='cosine',
    dim=dim
)

index.load_index(
    str(index_path)
)

index.set_ef(50)

st.success("HNSW index loaded successfully")

# ------------------------------------------------
# Upload Image
# ------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Product Image",
    type=["jpg", "jpeg", "png"]
)

# ------------------------------------------------
# Main Pipeline
# ------------------------------------------------

if uploaded_file is not None:

    # ------------------------------------------------
    # Load Uploaded Image
    # ------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        width=300
    )

    # ------------------------------------------------
    # YOLO Detection
    # ------------------------------------------------

    image_np = np.array(image)

    results = yolo_model(
        image_np
    )

    boxes = results[0].boxes

    # ------------------------------------------------
    # No Detection
    # ------------------------------------------------

    if len(boxes) == 0:

        st.error(
            "No object detected."
        )

        st.stop()

    # ------------------------------------------------
    # First Detection
    # ------------------------------------------------

    box = boxes[0]

    x1, y1, x2, y2 = (
        box.xyxy[0]
        .cpu()
        .numpy()
        .astype(int)
    )

    # ------------------------------------------------
    # Draw Detection Box
    # ------------------------------------------------

    detection_img = image_np.copy()

    cv2.rectangle(
        detection_img,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    st.subheader("YOLO Detection")

    st.image(
        detection_img,
        channels="RGB"
    )

    # ------------------------------------------------
    # Crop Product
    # ------------------------------------------------

    crop = image_np[
        y1:y2,
        x1:x2
    ]

    crop_pil = Image.fromarray(
        crop
    )

    st.subheader("Cropped Product")

    st.image(
        crop_pil,
        width=300
    )

    # ------------------------------------------------
    # Confirm Crop
    # ------------------------------------------------

    confirm = st.button(
        "Confirm Crop"
    )

    if confirm:

        st.success(
            "Crop confirmed. Running retrieval..."
        )

        # ------------------------------------------------
        # Generate Query Embedding
        # ------------------------------------------------

        inputs = processor(
            images=crop_pil,
            return_tensors="pt"
        )

        # ------------------------------------------------
        # Generate Query Embedding
        # ------------------------------------------------

        inputs = processor(
            images=crop_pil,
            return_tensors="pt"
        )

        with torch.no_grad():

            query_emb = model.get_image_features(
                **inputs
            )

        # ------------------------------------------------
        # Handle transformers output safely
        # ------------------------------------------------

        if not isinstance(query_emb, torch.Tensor):

            query_emb = query_emb.pooler_output

        # ------------------------------------------------
        # Normalize embedding
        # ------------------------------------------------

        query_emb = F.normalize(
            query_emb,
            p=2,
            dim=-1
        )

        query_emb = query_emb.cpu().numpy()
        # ------------------------------------------------
        # HNSW Retrieval
        # ------------------------------------------------

        labels, distances = index.knn_query(
            query_emb,
            k=5
        )

        st.subheader(
            "Top-5 Retrieval Results"
        )

        cols = st.columns(5)

        # ------------------------------------------------
        # Display Results
        # ------------------------------------------------

        for i in range(5):

            idx = labels[0][i]

            similarity = (
                1 - distances[0][i]
            )

            relative_path = metadata.iloc[idx][
                "image_path"
            ]

            img_path = (
                DATASET_ROOT
                / relative_path
            )

            if not img_path.exists():

                st.warning(
                    f"Missing image:\n{img_path}"
                )

                continue

            result_img = Image.open(
                img_path
            ).convert("RGB")

            with cols[i]:

                st.image(
                    result_img,
                    caption=f"Similarity: {similarity:.3f}"
                )