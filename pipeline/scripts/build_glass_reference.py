"""
Build Glass Reference Embeddings
-----------------------------------
Uses your actual "good glass" training images (from archive.zip, the same
ones PatchCore trained on) to build a CLIP image-embedding reference bank.

This makes the live-camera glass gate compare against REAL photos of your
float/solar glass, instead of relying only on generic text descriptions
like "a sheet of glass" - much stronger for confirming the camera is
actually looking at the right material.

Run once, after training:
    python build_glass_reference.py

Output:
    D:\\glass_defect\\models\\glass_reference_embeddings.npy
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# =====================================================
# CONFIG
# =====================================================

WORK_DIR = Path(r"D:\glass_defect\work")
MODELS_DIR = Path(r"D:\glass_defect\models")

# Reuse the same extracted "good" images folder that train_local.py built
GOOD_IMAGES_DIR = None  # auto-detected below, or set manually if needed

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_REFERENCE_IMAGES = 60  # enough to cover visual variety without being slow


def find_good_images_dir():
    """Reuse train_local.py's extraction — the good images end up under
    work/archive_extracted/... somewhere. Find the folder with the most
    images, same logic as train_local.py."""
    candidates = [
        WORK_DIR / "patchcore_dataset" / "train" / "good",
        WORK_DIR / "archive_extracted",
    ]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    best_dir, best_count = None, 0
    for root in candidates:
        if not root.exists():
            continue
        for dirpath, _, filenames in os.walk(root):
            count = sum(1 for f in filenames if Path(f).suffix.lower() in exts)
            if count > best_count:
                best_dir, best_count = Path(dirpath), count
    if best_dir is None:
        raise RuntimeError(
            "Could not find your good-glass images. Expected them under "
            f"{WORK_DIR}\\patchcore_dataset\\train\\good (created by "
            "train_local.py). If you moved things, set GOOD_IMAGES_DIR "
            "manually at the top of this script."
        )
    print(f"[find] Using {best_count} images from {best_dir}")
    return best_dir


def main():
    good_dir = Path(GOOD_IMAGES_DIR) if GOOD_IMAGES_DIR else find_good_images_dir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[clip] Loading {CLIP_MODEL_NAME} on {device}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [
        good_dir / f for f in os.listdir(good_dir)
        if Path(f).suffix.lower() in exts
    ]
    if not image_paths:
        raise RuntimeError(f"No images found in {good_dir}")

    if len(image_paths) > MAX_REFERENCE_IMAGES:
        import random
        random.seed(42)
        image_paths = random.sample(image_paths, MAX_REFERENCE_IMAGES)

    print(f"[clip] Embedding {len(image_paths)} reference images...")
    embeddings = []
    for i, path in enumerate(image_paths):
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[clip] Skipping {path.name}: {e}")
            continue
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
            if not torch.is_tensor(feats):
                feats = feats.pooler_output  # fallback for transformers versions
                                               # that return a model output object
            feats = feats / feats.norm(dim=-1, keepdim=True)  # normalize for cosine sim
        embeddings.append(feats.cpu().numpy()[0])
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(image_paths)}")

    embeddings = np.stack(embeddings)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "glass_reference_embeddings.npy"
    np.save(out_path, embeddings)
    print(f"[done] Saved {embeddings.shape[0]} reference embeddings -> {out_path}")


if __name__ == "__main__":
    main()
