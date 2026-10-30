"""
Glass Defect Detection - Local Training Pipeline
--------------------------------------------------
Trains the hybrid PatchCore (anomaly) + YOLOv8 (defect detection) pipeline
from your local dataset zips, adapted from the Kaggle notebook to run on
your Windows PC.

Run from PowerShell / CMD:
    python train_local.py

Outputs (ready to copy to Jetson tomorrow):
    models/patchcore.ckpt
    models/yolo_best.pt
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import shutil
import random
import zipfile
from pathlib import Path

# =====================================================
# CONFIG - edit these if your paths differ
# =====================================================

BASE_DIR = Path(r"D:\glass_defect")
DATASET_DIR = BASE_DIR / "dataset"

ARCHIVE_ZIP = DATASET_DIR / "archive.zip"          # "good" images -> PatchCore
ROBOFLOW_ZIP = DATASET_DIR / "roboflow.zip"         # YOLO annotated -> YOLOv8

WORK_DIR = BASE_DIR / "work"                        # extraction + working dir
MODELS_DIR = BASE_DIR / "models"                    # final output weights

PATCHCORE_EPOCHS = 1        # PatchCore is feature-memory based, 1 epoch is standard
PATCHCORE_CORESET_RATIO = 0.01  # lower = smaller memory bank = less GPU memory during
                                  # coreset subsampling. Default anomalib value (0.1) is
                                  # too large for 8GB laptop GPUs with a few thousand
                                  # training patches - drop further (e.g. 0.005) if you
                                  # still hit CUDA OOM.
YOLO_EPOCHS = 50
YOLO_IMGSZ = 640
YOLO_BATCH = 8               # lower to 4 if you hit CUDA OOM on a small GPU

# =====================================================
# STEP 0: Sanity checks
# =====================================================

def check_prereqs():
    missing = []
    if not ARCHIVE_ZIP.exists():
        missing.append(str(ARCHIVE_ZIP))
    if not ROBOFLOW_ZIP.exists():
        missing.append(str(ROBOFLOW_ZIP))
    if missing:
        raise FileNotFoundError(
            "Missing dataset file(s):\n" + "\n".join(missing) +
            "\n\nUpdate ARCHIVE_ZIP / ROBOFLOW_ZIP paths at the top of this "
            "script if your files are named/located differently."
        )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print("[check] Dataset files found. Working dir:", WORK_DIR)


# =====================================================
# STEP 1: Extract zips
# =====================================================

def extract_zips():
    archive_out = WORK_DIR / "archive_extracted"
    roboflow_out = WORK_DIR / "roboflow_extracted"

    for zpath, out in [(ARCHIVE_ZIP, archive_out), (ROBOFLOW_ZIP, roboflow_out)]:
        if out.exists():
            print(f"[extract] {out.name} already extracted, skipping.")
            continue
        print(f"[extract] Extracting {zpath.name} ...")
        with zipfile.ZipFile(zpath, "r") as z:
            z.extractall(out)

    return archive_out, roboflow_out


def find_image_folder(root: Path, min_images=10):
    """Walk a directory tree and find the folder with the most images.
    Handles the case where the zip has a nested folder structure."""
    best_dir, best_count = None, 0
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for dirpath, _, filenames in os.walk(root):
        count = sum(1 for f in filenames if Path(f).suffix.lower() in exts)
        if count > best_count:
            best_dir, best_count = Path(dirpath), count
    if best_dir is None or best_count < min_images:
        raise RuntimeError(
            f"Could not find a folder with >= {min_images} images under {root}. "
            f"Check the extracted contents manually: {root}"
        )
    print(f"[detect] Found {best_count} images in {best_dir}")
    return best_dir


def find_yaml(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f == "data.yaml":
                return Path(dirpath) / f
    raise RuntimeError(
        f"Could not find data.yaml under {root}. "
        f"Roboflow YOLOv8 exports should include this file — check the zip contents."
    )


# =====================================================
# STEP 2: Build PatchCore dataset (good/train, good/test split)
# =====================================================

def build_patchcore_dataset(good_images_dir: Path):
    pc_root = WORK_DIR / "patchcore_dataset"
    train_dir = pc_root / "train" / "good"
    test_dir = pc_root / "test" / "good"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in os.listdir(good_images_dir) if Path(f).suffix.lower() in exts]
    if not images:
        raise RuntimeError(f"No images found in {good_images_dir}")

    random.seed(42)
    random.shuffle(images)
    split = int(len(images) * 0.8)
    train_imgs, test_imgs = images[:split], images[split:]

    for img in train_imgs:
        shutil.copy(good_images_dir / img, train_dir / img)
    for img in test_imgs:
        shutil.copy(good_images_dir / img, test_dir / img)

    print(f"[patchcore] {len(train_imgs)} train / {len(test_imgs)} test images ready at {pc_root}")
    return pc_root


# =====================================================
# STEP 3: Train PatchCore
# =====================================================

def train_patchcore(pc_root: Path):
    from anomalib.data import Folder
    from anomalib.models import Patchcore
    from anomalib.engine import Engine

    datamodule = Folder(
        name="glass_defect",
        root=str(pc_root),
        normal_dir="train/good",
        normal_test_dir="test/good",
        num_workers=0,  # Windows: multiprocess dataloader workers can fail to load
                        # torch DLLs if the pagefile is too small; 0 avoids that
                        # entirely at the cost of slightly slower data loading.
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        num_neighbors=9,
        coreset_sampling_ratio=PATCHCORE_CORESET_RATIO,
    )

    engine = Engine(max_epochs=PATCHCORE_EPOCHS, enable_progress_bar=True)
    engine.fit(model=model, datamodule=datamodule)

    # Find the checkpoint anomalib produced
    import glob
    ckpts = glob.glob(str(WORK_DIR / "**" / "*.ckpt"), recursive=True) + \
            glob.glob("results/**/*.ckpt", recursive=True)
    if not ckpts:
        raise RuntimeError(
            "Training finished but no .ckpt found. Check the 'results/' folder "
            "manually — anomalib's output path can vary by version."
        )
    src_ckpt = ckpts[-1]
    dst_ckpt = MODELS_DIR / "patchcore.ckpt"
    shutil.copy(src_ckpt, dst_ckpt)
    print(f"[patchcore] Saved -> {dst_ckpt}")
    return dst_ckpt


# =====================================================
# STEP 4: Train YOLOv8
# =====================================================

def train_yolo(data_yaml: Path):
    from ultralytics import YOLO
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[yolo] Training on device: {'GPU' if device == 0 else 'CPU'}")
    if device == "cpu":
        print("[yolo] WARNING: CPU training will be slow. Consider reducing "
              "YOLO_EPOCHS or using yolov8n (already the smallest) if this "
              "takes too long. A few epochs is enough to sanity-check the "
              "pipeline before a longer overnight run.")

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=YOLO_EPOCHS,
        imgsz=YOLO_IMGSZ,
        batch=YOLO_BATCH,
        device=device,
        workers=2,
    )

    best_weights = Path("runs/detect/train/weights/best.pt")
    if not best_weights.exists():
        # ultralytics increments train, train2, train3... if run multiple times
        import glob
        candidates = sorted(glob.glob("runs/detect/train*/weights/best.pt"))
        if not candidates:
            raise RuntimeError("YOLO training finished but best.pt not found under runs/detect/")
        best_weights = Path(candidates[-1])

    dst = MODELS_DIR / "yolo_best.pt"
    shutil.copy(best_weights, dst)
    print(f"[yolo] Saved -> {dst}")
    return dst


# =====================================================
# MAIN
# =====================================================

def main():
    check_prereqs()
    archive_out, roboflow_out = extract_zips()

    good_images_dir = find_image_folder(archive_out)
    data_yaml = find_yaml(roboflow_out)
    print(f"[config] YOLO data.yaml -> {data_yaml}")

    pc_root = build_patchcore_dataset(good_images_dir)

    print("\n===== Training PatchCore (anomaly model) =====")
    train_patchcore(pc_root)

    print("\n===== Training YOLOv8 (defect detector) =====")
    train_yolo(data_yaml)

    print(f"\nAll done. Weights are in: {MODELS_DIR}")
    print("Copy the whole 'models' folder to the Jetson tomorrow and point")
    print("dashboard.py at it.")


if __name__ == "__main__":
    main()
