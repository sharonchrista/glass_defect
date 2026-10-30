# Uchita-Drishti — Machine Vision Glass Defect Detection

A hybrid machine-vision quality-inspection system that detects surface and material defects — raw-material impurities, melting inconsistencies, air bubbles, scratches, heavy scratches, edge chips and uneven edges — on uncoated float and solar glass, deployed on an NVIDIA Jetson Orin Nano for live, edge-based inspection.

Developed under CRIEYA Innovation Project **CRIEYA/Inno-Project/Oct.2025/15**, School of Computing (CSE), MIT Art, Design and Technology University, Pune.

## How it works

The pipeline combines three models in sequence:

1. **Glass gate (CLIP, zero-shot)** — verifies the camera is actually viewing uncoated float/solar glass, using embeddings of the project's own reference glass images. Rejects people, backgrounds, or coated glass in frame.
2. **PatchCore (unsupervised anomaly detection)** — trained only on defect-free glass; flags a sheet as anomalous if it deviates from the "normal" pattern, catching both known and previously unseen defect types.
3. **YOLOv8 (supervised detection)** — trained on annotated defect classes; draws bounding boxes and a confidence score around specific defects.

A rule-based fusion layer combines PatchCore's anomaly score and YOLOv8's detection confidence into a single CLEAN / DEFECTIVE verdict, with CLIP additionally providing a zero-shot defect-type label when an anomaly is flagged.

For two-camera setups, `fuse_multi_camera_results()` combines both cameras' verdicts with OR-based decision-level fusion — a sheet is flagged DEFECTIVE if either camera's pipeline detects a defect.

## Repository structure

| Path | Purpose |
| --- | --- |
| `scripts/train_local.py` | Trains PatchCore + YOLOv8 locally on a GPU workstation |
| `scripts/build_glass_reference.py` | Builds the CLIP reference embeddings for the glass gate |
| `scripts/requirements_train.txt` | Training-side Python dependencies |
| `dashboard/app.py` | Flask live-inspection dashboard (entry point) |
| `dashboard/defect_localization.py` | Pixel-to-mm homography for defect coordinates |
| `dashboard/machine_interface.py` | Swappable output layer (TCP/JSON, Modbus-TCP, file-drop) for a downstream cutting machine |
| `dashboard/multi_camera_coordinator.py` | Merges and de-duplicates detections across two cameras |
| `dashboard/line_config.py` | Physical line measurements (camera spacing, conveyor speed, encoder calibration) |
| `dashboard/templates/index.html` | Dashboard front end |
| `dashboard/requirements_dashboard.txt` | Jetson-side Python dependencies |
| `docs/` | Project documentation (`GITHUB_README.md`, `GITHUB_SETUP.md`, `PUBLIC_DEPLOY.md`) |
| `jetson-lan-switching-guide.txt` | Switching the Jetson between a direct static-IP link and the campus LAN |

Model weights (`*.pt`, `*.ckpt`), the dataset, and `glass_reference_embeddings.npy` are intentionally excluded via `.gitignore` — they're too large for GitHub and must be supplied separately.

## Setup

**1. Training (GPU workstation):**
```bash
pip install -r scripts/requirements_train.txt
cd scripts
python train_local.py
```

**2. Jetson Orin Nano (inference):**
```bash
pip3 install -r dashboard/requirements_dashboard.txt --break-system-packages
```
If you hit ABI conflicts between JetPack's system packages and pip-installed ones:
```bash
pip3 install --user --force-reinstall --break-system-packages matplotlib numpy pandas
```

**3. Run the dashboard:**
```bash
cd dashboard
python3 app.py --models-dir ../models --device 0 --device2 2
```
Then open `http://<jetson-ip>:5000` in a browser on the same network. Omit `--device2` for single-camera mode, or add `--no-clip` to skip CLIP entirely (e.g. on a network with no internet access — the glass gate then auto-falls back to "always pass").

## Hardware

- NVIDIA Jetson Orin Nano Developer Kit (NVMe-only boot — no SD card slot)
- First-time OS flash via a bootable-USB installer and the J14 serial header (pin3=RX, pin4=TX, pin11=GND, 115200 baud)
- MOKOSE HD Webcam USB industrial camera(s)

## Status

Core detection pipeline and the live single/dual-camera dashboard are working and deployed on-device. Defect-localisation-to-cutting-machine integration is designed (homography calibration, encoder-based conveyor tracking, swappable machine-interface output) but awaiting the physical conveyor/encoder hardware for calibration and field trials.

## Team

- **Principal Investigator:** Dr. Mansi Bhonsle, Associate Professor, Dept. of CSE-CC (incoming PI; outgoing PI: Dr. Sharon Christa)
- **Team Lead / AI Engineer:** Nitya Dixit
- **Edge AI Engineer:** Ritika Sharma

## License

Internal CRIEYA-funded research project, MIT Art, Design and Technology University. Contact the CRIEYA office (crieya@mituniversity.edu.in) for usage queries.
