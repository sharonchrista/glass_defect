# Hybrid Glass Defect Detection

PatchCore (anomaly detection) + YOLOv8 (defect detection) fusion pipeline
for real-time glass defect inspection, deployed on a Jetson Orin Nano.

**Live dashboard:** https://dashboard.yourdomain.com *(update once your
Cloudflare Tunnel is running — see `PUBLIC_DEPLOY.md`)*

**Project page:** https://sharonchrista.github.io/glass-defect-pipeline/

## What this does

Two models run per frame and their outputs are fused:
- **PatchCore** flags frames that deviate from a learned "good glass" baseline
  (catches novel/unlabeled defect types — bubbles, sand inclusions, etc.)
- **YOLOv8** detects and localizes known defect classes (chips, scratches,
  uneven edges) from a labeled dataset
- A rule-based fusion layer combines both scores into a final CLEAN /
  DEFECTIVE verdict (see `dashboard/app.py`, `fuse_predictions()`)

Served as a Flask dashboard with two modes: live camera feed (continuous
inference on a line camera) and manual image upload.

## Repository structure

```
scripts/      training pipeline (extracts dataset, trains both models)
dashboard/    Flask app — live feed + upload, runs identically on
              dev PC and Jetson
docs/         static GitHub Pages landing page (this repo's project page)
```

## Running this yourself

See `README.md` in the repo root for local training + Jetson deployment
steps, and `PUBLIC_DEPLOY.md` for exposing the dashboard publicly via
Cloudflare Tunnel.

Model weights and the training dataset are **not** included in this
repository (see `.gitignore`) — train locally following the instructions,
or reach out for access to pretrained weights.

## Stack

Python · PyTorch · Anomalib (PatchCore) · Ultralytics (YOLOv8) · Flask ·
OpenCV · Jetson Orin Nano · Cloudflare Tunnel

---
Built by Dr. Sharon Christa, MIT-ADT University.
