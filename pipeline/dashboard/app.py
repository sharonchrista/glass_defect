"""
Hybrid Glass Defect Detection Dashboard
-----------------------------------------
PatchCore (anomaly detection) + YOLOv8 (defect detection) + CLIP
(zero-shot labeling of anomalies) fusion, served as a Flask dashboard with
two modes:
  - Live webcam feed (continuous inference, for line monitoring)
  - Manual image upload (single-shot check)

CLIP only runs when PatchCore flags an anomaly above threshold - it labels
*what kind* of anomaly it looks like (raw material impurity, melting
inconsistency, scratch, heavy scratch, etc.) using open-vocabulary text
prompts, with no training data required for those classes.

Designed to run identically on your Windows PC (testing) and the Jetson
Orin Nano (deployment) — same code, same model files, just point --device
and --models-dir appropriately.

Run:
    python app.py --models-dir ./models --device 0

Then open:
    http://<host-ip>:5000
"""

import argparse
import io
import os
import threading
import time
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template, jsonify, request
from torchvision import transforms

app = Flask(__name__)

# =====================================================
# BASIC AUTH (required once this app is exposed publicly via tunnel)
# Set env vars before running, e.g.:
#   PowerShell:  $env:DASHBOARD_USER="admin"; $env:DASHBOARD_PASS="changeme"
#   Jetson bash: export DASHBOARD_USER=admin DASHBOARD_PASS=changeme
# If unset, auth is disabled (fine for local-only testing, NOT for tunneling).
# =====================================================

DASH_USER = os.environ.get("DASHBOARD_USER")
DASH_PASS = os.environ.get("DASHBOARD_PASS")


def check_auth(username, password):
    return username == DASH_USER and password == DASH_PASS


def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not DASH_USER or not DASH_PASS:
            return f(*args, **kwargs)  # auth disabled, local-only mode
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response(
                "Authentication required", 401,
                {"WWW-Authenticate": 'Basic realm="Glass Defect Dashboard"'}
            )
        return f(*args, **kwargs)
    return wrapper

STATE = {
    "yolo_model": None,
    "patchcore_model": None,
    "clip_model": None,
    "clip_processor": None,
    "glass_reference_embeddings": None,
    "device": None,
    "cam": None,
    "cam_lock": threading.Lock(),
    "latest_frame": None,
    "latest_result": {"prediction": "--", "reason": "Waiting for first frame..."},
    "running": False,
}

PATCHCORE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# =====================================================
# ZERO-SHOT LAYER (CLIP) - runs only when PatchCore flags an anomaly
# Catches defect types with no labeled training data: raw material
# impurities, melting inconsistencies, etc. Since PatchCore already
# doesn't know what a defect "is" - only that it deviates from good
# glass - CLIP's job is just to guess which of these labels the
# anomalous region looks most like, no training required.
# =====================================================

ZERO_SHOT_LABELS = [
    "raw material impurity",
    "melting inconsistency",
    "air bubble",
    "scratch",
    "heavy scratch",
    "edge chip",
    "uneven edge",
]

ZERO_SHOT_ANOMALY_TRIGGER = 0.7  # PatchCore score above which CLIP runs
                                   # (matches the "moderate anomaly" fusion
                                   # threshold already in use below)

# =====================================================
# GLASS GATE (CLIP) - runs FIRST, on every frame, before YOLO/PatchCore.
# Confirms the frame actually shows bare (uncoated) float/solar glass
# right after manufacturing, not a person, hand, background, tinted/coated
# glass, or unrelated object. If the gate fails, inspection is skipped
# entirely for that frame - avoids meaningless verdicts on non-glass scenes.
# =====================================================

GLASS_GATE_POSITIVE_LABELS = [
    "a flat sheet of bare uncoated float glass",
    "a flat sheet of bare uncoated solar glass",
]
GLASS_GATE_NEGATIVE_LABELS = [
    "a person or hand",
    "an empty background or room",
    "a coated or tinted glass surface",
    "an unrelated object",
]
GLASS_GATE_LABELS = GLASS_GATE_POSITIVE_LABELS + GLASS_GATE_NEGATIVE_LABELS

GLASS_GATE_CONFIDENCE_THRESHOLD = 0.45  # minimum confidence in a positive
                                          # label to proceed with inspection
                                          # (used only if no reference bank
                                          # is loaded, or as a secondary check)

GLASS_REFERENCE_SIMILARITY_THRESHOLD = 0.80  # cosine similarity to your
                                               # actual good-glass images
                                               # above which a frame counts
                                               # as glass. Tune this after
                                               # watching real numbers.


def classify_glass_gate(frame_bgr: np.ndarray):
    """Returns (is_glass: bool, label: str, confidence: float).

    Prefers comparing against your real training images (glass_reference_
    embeddings.npy, built by build_glass_reference.py) via cosine
    similarity - this is a much stronger signal than generic text
    descriptions since it's matching against your actual glass, not
    CLIP's general notion of "glass". Falls back to text-label zero-shot
    if no reference bank is loaded."""
    clip_model = STATE["clip_model"]
    clip_processor = STATE["clip_processor"]
    if clip_model is None:
        return True, "gate disabled (CLIP not loaded)", 0.0

    device = STATE["device"]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    reference_bank = STATE.get("glass_reference_embeddings")
    if reference_bank is not None:
        inputs = clip_processor(images=rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            if not torch.is_tensor(feats):
                feats = feats.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
        frame_embedding = feats.cpu().numpy()[0]

        similarities = reference_bank @ frame_embedding  # cosine sim (both normalized)
        best_sim = float(similarities.max())

        is_glass = best_sim >= GLASS_REFERENCE_SIMILARITY_THRESHOLD
        label = "matches known good glass" if is_glass else "does not match known glass"
        return is_glass, label, best_sim

    # ---- Fallback: text-label zero-shot (no reference bank loaded) ----
    inputs = clip_processor(
        text=GLASS_GATE_LABELS,
        images=rgb,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    best_idx = int(probs.argmax())
    best_label = GLASS_GATE_LABELS[best_idx]
    best_conf = float(probs[best_idx])

    is_glass = (
        best_label in GLASS_GATE_POSITIVE_LABELS
        and best_conf >= GLASS_GATE_CONFIDENCE_THRESHOLD
    )
    return is_glass, best_label, best_conf


# =====================================================
# FUSION LOGIC (same rules as the notebook)
# =====================================================

def fuse_predictions(patchcore_score: float, highest_yolo_conf: float):
    if patchcore_score > 0.95:
        return "DEFECTIVE", "Extreme anomaly detected by PatchCore"
    elif highest_yolo_conf > 0.5:
        return "DEFECTIVE", "YOLO confirmed known defect"
    elif patchcore_score > 0.7 and highest_yolo_conf > 0.25:
        return "DEFECTIVE", "Moderate anomaly supported by YOLO"
    else:
        return "CLEAN", "Insufficient defect evidence"


def zero_shot_classify(frame_bgr: np.ndarray):
    """Run CLIP zero-shot classification against ZERO_SHOT_LABELS.
    Returns (best_label, confidence) or (None, 0.0) if CLIP isn't loaded."""
    clip_model = STATE["clip_model"]
    clip_processor = STATE["clip_processor"]
    if clip_model is None:
        return None, 0.0

    device = STATE["device"]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    inputs = clip_processor(
        text=ZERO_SHOT_LABELS,
        images=rgb,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    best_idx = int(probs.argmax())
    return ZERO_SHOT_LABELS[best_idx], float(probs[best_idx])


def run_inference(frame_bgr: np.ndarray):
    yolo_model = STATE["yolo_model"]
    patchcore_model = STATE["patchcore_model"]
    device = STATE["device"]

    # ---- GLASS GATE (runs first, skips everything else if it fails) ----
    is_glass, gate_label, gate_conf = classify_glass_gate(frame_bgr)
    if not is_glass:
        overlay = frame_bgr.copy()
        cv2.putText(
            overlay, "NO GLASS DETECTED - inspection paused", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
        )
        return overlay, {
            "prediction": "NO GLASS",
            "reason": f"Gate rejected frame — looks like: {gate_label} ({gate_conf:.2f})",
            "patchcore_score": None,
            "yolo_confidence": None,
            "detected_classes": [],
            "zero_shot_label": None,
            "zero_shot_confidence": None,
            "gate_label": gate_label,
            "gate_confidence": round(gate_conf, 4),
        }

    # ---- YOLO ----
    yolo_results = yolo_model(frame_bgr, verbose=False)
    result = yolo_results[0]
    boxes = result.boxes

    highest_yolo_conf = 0.0
    detected_classes = []
    if boxes is not None and len(boxes) > 0:
        highest_yolo_conf = float(boxes.conf.max())
        class_ids = boxes.cls.cpu().numpy()
        names = result.names
        detected_classes = [names[int(c)] for c in class_ids]

    plotted = result.plot()  # BGR, annotated with YOLO boxes

    # ---- PatchCore ----
    patchcore_score = 0.0
    if patchcore_model is not None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = PATCHCORE_TRANSFORM(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            output = patchcore_model(tensor)
        patchcore_score = float(output.pred_score)

    # ---- CLIP zero-shot (only when PatchCore sees an anomaly) ----
    zero_shot_label, zero_shot_conf = None, 0.0
    if patchcore_score > ZERO_SHOT_ANOMALY_TRIGGER:
        zero_shot_label, zero_shot_conf = zero_shot_classify(frame_bgr)

    prediction, reason = fuse_predictions(patchcore_score, highest_yolo_conf)

    # If YOLO didn't already name the defect but CLIP has a guess, surface it
    if prediction == "DEFECTIVE" and not detected_classes and zero_shot_label:
        reason = f"{reason} — likely type: {zero_shot_label}"

    return plotted, {
        "prediction": prediction,
        "reason": reason,
        "patchcore_score": round(patchcore_score, 4),
        "yolo_confidence": round(highest_yolo_conf, 4),
        "detected_classes": detected_classes,
        "zero_shot_label": zero_shot_label,
        "zero_shot_confidence": round(zero_shot_conf, 4) if zero_shot_label else None,
        "gate_label": gate_label,
        "gate_confidence": round(gate_conf, 4),
    }


# =====================================================
# MODEL LOADING
# =====================================================

def load_models(models_dir: Path):
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[models] Using device: {device}")

    yolo_path = models_dir / "yolo_best.pt"
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO weights not found at {yolo_path}")
    yolo_model = YOLO(str(yolo_path))
    print(f"[models] Loaded YOLO from {yolo_path}")

    patchcore_model = None
    patchcore_path = models_dir / "patchcore.ckpt"
    if patchcore_path.exists():
        from anomalib.models import Patchcore
        patchcore_model = Patchcore.load_from_checkpoint(
            str(patchcore_path), weights_only=False
        )
        patchcore_model.eval()
        patchcore_model = patchcore_model.to(device)
        print(f"[models] Loaded PatchCore from {patchcore_path}")
    else:
        print(f"[models] WARNING: PatchCore weights not found at {patchcore_path} "
              f"- running YOLO-only mode (fusion will use score 0 for anomaly).")

    # ---- CLIP zero-shot layer ----
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_name = "openai/clip-vit-base-patch32"
        print(f"[models] Loading CLIP ({clip_name}) for zero-shot labeling...")
        clip_model = CLIPModel.from_pretrained(clip_name).to(device)
        clip_model.eval()
        clip_processor = CLIPProcessor.from_pretrained(clip_name)
        STATE["clip_model"] = clip_model
        STATE["clip_processor"] = clip_processor
        print("[models] CLIP loaded. Zero-shot labels:", ZERO_SHOT_LABELS)
    except Exception as e:
        print(f"[models] WARNING: Could not load CLIP ({e}) - zero-shot "
              f"labeling disabled, PatchCore/YOLO fusion still works normally.")

    # ---- Glass reference embeddings (from build_glass_reference.py) ----
    ref_path = models_dir / "glass_reference_embeddings.npy"
    if ref_path.exists():
        ref_embeddings = np.load(ref_path)
        STATE["glass_reference_embeddings"] = ref_embeddings
        print(f"[models] Loaded {ref_embeddings.shape[0]} glass reference "
              f"embeddings from {ref_path} - gate will use real-image "
              f"similarity matching.")
    else:
        print(f"[models] No glass reference embeddings found at {ref_path} "
              f"- gate will fall back to generic text-label matching. Run "
              f"build_glass_reference.py to build one from your training "
              f"images for a stronger gate.")

    STATE["yolo_model"] = yolo_model
    STATE["patchcore_model"] = patchcore_model
    STATE["device"] = device


# =====================================================
# WEBCAM LOOP
# =====================================================

def camera_loop(device_index: int, width: int, height: int):
    # Windows quirk: CAP_DSHOW can fail silently when opened from a background
    # thread (COM apartment initialization differs per-thread) even though the
    # exact same call succeeds from the main thread. Try a few backends with
    # short retries before giving up.
    if os.name == "nt":
        backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif hasattr(cv2, "CAP_V4L2"):
        backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends_to_try = [cv2.CAP_ANY]

    cap = None
    for backend in backends_to_try:
        for attempt in range(3):
            candidate = cv2.VideoCapture(device_index, backend)
            if candidate.isOpened():
                ok, _ = candidate.read()
                if ok:
                    cap = candidate
                    print(f"[camera] Opened with backend={backend} on attempt {attempt + 1}")
                    break
                candidate.release()
            else:
                candidate.release()
            time.sleep(0.3)
        if cap is not None:
            break

    if cap is None:
        print(f"[camera] WARNING: could not open camera index {device_index} "
              f"after trying backends {backends_to_try}. Live feed will be "
              f"unavailable — upload mode still works.")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    STATE["cam"] = cap
    STATE["running"] = True
    print("[camera] Live loop started.")

    while STATE["running"]:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue

        try:
            annotated, result = run_inference(frame)
        except Exception as e:
            print(f"[camera] Inference error: {e}")
            annotated, result = frame, STATE["latest_result"]

        with STATE["cam_lock"]:
            STATE["latest_frame"] = annotated
            STATE["latest_result"] = result

        time.sleep(0.05)  # cap ~20fps to keep CPU/GPU headroom


def generate_mjpeg():
    while True:
        with STATE["cam_lock"]:
            frame = STATE["latest_frame"]
        if frame is None:
            time.sleep(0.1)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


# =====================================================
# ROUTES
# =====================================================

@app.route("/")
@require_auth
def index():
    return render_template("index.html")


@app.route("/video_feed")
@require_auth
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/live_status")
@require_auth
def live_status():
    with STATE["cam_lock"]:
        result = dict(STATE["latest_result"])
    result["camera_available"] = STATE["cam"] is not None
    return jsonify(result)


@app.route("/predict_upload", methods=["POST"])
@require_auth
def predict_upload():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    annotated, result = run_inference(frame)

    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    import base64
    result["annotated_image"] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    return jsonify(result)


# =====================================================
# MAIN
# =====================================================

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid Glass Defect Dashboard")
    p.add_argument("--models-dir", type=str, default="./models")
    p.add_argument("--device", type=int, default=0, help="Camera index (V4L2/DirectShow)")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--no-camera", action="store_true", help="Skip live camera, upload-only mode")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models_dir = Path(args.models_dir)

    load_models(models_dir)

    if not args.no_camera:
        cam_thread = threading.Thread(
            target=camera_loop, args=(args.device, args.width, args.height), daemon=True
        )
        cam_thread.start()
    else:
        print("[camera] Skipped (--no-camera). Upload mode only.")

    try:
        app.run(host="0.0.0.0", port=args.port, threaded=True)
    finally:
        STATE["running"] = False
        if STATE["cam"] is not None:
            STATE["cam"].release()
