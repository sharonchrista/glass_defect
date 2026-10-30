# Glass Defect Detection — Local Training + Jetson Deployment

Hybrid PatchCore (anomaly detection) + YOLOv8 (defect detection) pipeline,
adapted from your Kaggle notebook to run locally and deploy to Jetson.

## TONIGHT — train on your Windows PC

1. Put this whole folder somewhere convenient, e.g. `D:\glass_defect\pipeline`.
   Your dataset should already be at:
   - `D:\glass_defect\dataset\archive.zip`
   - `D:\glass_defect\dataset\roboflow.zip`

2. Install dependencies:
   ```
   pip install -r scripts/requirements_train.txt
   ```
   If you have an NVIDIA GPU, install CUDA-enabled torch FIRST (check your
   CUDA version with `nvidia-smi`, then use the matching index URL from
   https://pytorch.org/get-started/locally/), then run the line above.
   If no GPU, plain `pip install torch torchvision` works — training will
   just be slower. You confirmed "simple YOLO" is fine for now, so CPU
   training for a sanity-check run (fewer epochs) is a reasonable fallback.

3. Open `scripts/train_local.py` and check the CONFIG section at the top —
   paths already match what you gave me, but confirm `archive.zip` really
   is your "good" glass folder and `roboflow.zip` is the annotated YOLO
   defect set. The script auto-detects the actual image/data.yaml
   locations inside each zip, so nested folder structures are handled.

4. Run it:
   ```
   cd scripts
   python train_local.py
   ```
   This will:
   - Extract both zips
   - Build the PatchCore train/test folder split from `archive.zip`
   - Train PatchCore (fast, 1 epoch — it's memory-bank based, not gradient trained)
   - Train YOLOv8n on the roboflow-format defect dataset (50 epochs default —
     lower `YOLO_EPOCHS` in the config if you're on CPU and it's too slow)
   - Save both weight files to `D:\glass_defect\models\`:
     - `patchcore.ckpt`
     - `yolo_best.pt`

5. Once training finishes, test the dashboard locally before tomorrow:
   ```
   cd ../dashboard
   pip install -r requirements_dashboard.txt
   python app.py --models-dir D:\glass_defect\models --device 0
   ```
   Open `http://localhost:5000` — try both the Live Camera tab (uses your
   laptop webcam) and the Upload Image tab (drop in a test photo). Confirm
   you get a CLEAN/DEFECTIVE verdict with scores.

   If you don't want to test the live camera tonight (e.g. no webcam handy),
   run with `--no-camera` to test upload-only mode.

## TOMORROW — deploy to Jetson Orin Nano

1. Copy the whole `glass-defect-pipeline` folder (including the trained
   `models/` folder from `D:\glass_defect\models\`) onto the Jetson — via
   USB drive, `scp`, or a shared network folder.

2. On the Jetson, install dependencies (same file, same versions matter):
   ```
   pip install -r dashboard/requirements_dashboard.txt --break-system-packages
   ```
   `anomalib` and `ultralytics` are heavier installs — expect this step to
   take a while on Jetson's ARM CPU. If `anomalib` fails to build, install
   `torch`/`torchvision` first via NVIDIA's Jetson-specific wheels (see
   the JetPack/PyTorch compatibility page) before the rest.

3. Plug in the MOKOSE webcam, confirm it's detected:
   ```
   v4l2-ctl --list-devices
   ```

4. Run the dashboard, pointing at wherever you placed the `models` folder:
   ```
   cd dashboard
   python3 app.py --models-dir ../models --device 0
   ```

5. Open `http://<jetson-ip>:5000` from any browser on the same network.
   Live Camera tab now runs continuous inference off the actual line camera.

## Notes / things to watch

- **Fusion thresholds** (`patchcore_score > 0.95`, `yolo_conf > 0.5`, etc. in
  `dashboard/app.py`'s `fuse_predictions()`) are copied straight from your
  notebook. Once you see real results on the Jetson with live glass, you'll
  likely want to tune these against your actual false-positive/negative
  rate — that's expected, not a bug.
- **If PatchCore weights are missing** the dashboard still runs in
  YOLO-only mode rather than crashing — useful if training runs long and
  you want to test YOLO alone first.
- **Model files are large** (PatchCore backbone + YOLO weights) — if
  transferring via USB drive, expect a few hundred MB to a couple GB total.
- The dashboard code is identical between Windows testing and Jetson
  deployment — only `--device` (camera index differs by OS/driver) and
  `--models-dir` should need changing.
