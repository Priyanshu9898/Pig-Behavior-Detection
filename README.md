# 🐷 Pig Behavior Detection System

<p align="center">
  <a href="https://github.com/Priyanshu9898/Pig-Behavior-Detection/stargazers"><img src="https://img.shields.io/github/stars/Priyanshu9898/Pig-Behavior-Detection?style=for-the-badge" alt="Stars"></a>
  <a href="https://github.com/Priyanshu9898/Pig-Behavior-Detection/network/members"><img src="https://img.shields.io/github/forks/Priyanshu9898/Pig-Behavior-Detection?style=for-the-badge" alt="Forks"></a>
  <a href="https://github.com/Priyanshu9898/Pig-Behavior-Detection/issues"><img src="https://img.shields.io/github/issues/Priyanshu9898/Pig-Behavior-Detection?style=for-the-badge" alt="Issues"></a>
  <a href="https://github.com/Priyanshu9898/Pig-Behavior-Detection/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Priyanshu9898/Pig-Behavior-Detection?style=for-the-badge" alt="License"></a>
</p>


Detect *eating, drinking, lying, moving,* and *standing* behaviors of pigs from video‑frame sequences recorded in commercial pens.

---

## 🛠 Tech Stack

| Layer | Technology | Badge |
|-------|------------|-------|
| Language | Python 3.10+ | ![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white) |
| CV Model | Ultralytics YOLOv8 | ![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-orange) |
| DL Framework | PyTorch 2.2 (+ CUDA 11.x) | ![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white) |
| Libraries | OpenCV 4.9, NumPy, TQDM | ![OpenCV](https://img.shields.io/badge/OpenCV-4.9-blueviolet?logo=opencv&logoColor=white) |


## 🎯 Goals
| # | Objective |
|---|-----------|
| 1 | Build an end‑to‑end pipeline that ingests an MP4 sequence and produces per‑frame behavior labels in the format expected by **`output.json`**. |
| 2 | Achieve accuracy comparable to the metrics reported in Table 2 of the reference paper—while **training is optional** for the grader. |
| 3 | Provide a clean, reproducible workflow that can be executed on a laptop, an on‑prem GPU, or Google Colab. |
| 4 | Summarize evaluation results in **`accuracy.txt`** exactly as required for submission. |

---

## 📚 Dataset  
* **Source:** 12 annotated sequences in `annoted.tar` from <https://homepages.inf.ed.ac.uk/rbf/PIGDATA/>  
* **Train / Val split:**  
  * **Train:** first 7 sub‑folders - ['000002','000009','000016','000028','000036','000033','000010']
  * **Val:** last 5 sub‑folders - ['000113','000005','000208','000060','000078']
* **Ignore** any other un‑annotated sequences on the site.

> **Tip:** Extract `annoted.tar` so the folder hierarchy looks like:
> ```
> data/
> ├── 01/
> │   ├── color.mp4
> │   ├── depth_scale.npy
> │   ├── background.png
> │   └── output.json        # ground truth for this clip
> └── ...
> ```

---

## 🚀 Project Setup

> **Prerequisites**  ‑ Python ≥ 3.10 • Git • (Optional) NVIDIA GPU w/ CUDA 11+

### 1 · Clone the repository

```bash
git clone https://github.com/Priyanshu9898/Pig-Behavior-Detection.git
cd Pig-Behavior-Detection
```

### 2 · Create & activate a virtual environment

| OS            | Command                                                      |
|---------------|--------------------------------------------------------------|
| macOS / Linux | `python -m venv env && source env/bin/activate`              |
| Windows PS    | `python -m venv env; .\env\Scripts\Activate.ps1`            |

### 3 · Install dependencies

```bash
pip install -r requirements.txt             # runtime
```

### 4 · Download & unpack the dataset

```bash
mkdir -p data
# Manually place annoted.tar inside data/ then
tar -xf data/annoted.tar -C data/
```

## 📝 Notebook: `yolo_training.ipynb`

`yolo_training.ipynb` is for training yolo model.
1. **Environment Setup** – Installs the required libraries and checks CUDA availability.
2. **Data Prep** – Extracts `annoted.tar`, applies the train/val split, and converts labels to YOLOv8 format.
3. **Model Config** – Loads the YOLOv8‑small backbone.
4. **Training Loop** – Trains for 50 epochs.
5. **Checkpoint Export** – Saves the best model to `weights/model.pt`.
6. **Quick Validation** – Runs inference on the validation set and prints `accuracy.txt`.

## 📝 Notebook: `Complete_Pipeline.ipynb`
During inference, the code reads each video frame, applies YOLO every frame to detect pigs, tracks each detection over time with a lightweight MOSSE tracker, and—on the fly—classifies posture and uses simple motion and orientation heuristics to decide “moving,” “eating,” or “drinking” when appropriate. Finally, it assembles each pig’s per-frame labels into the required JSON format, writes out an accuracy report comparing predictions to the ground‐truth output.json files, and computes AP, TP, FP, and miss rates for each clip and overall.e.

## 🔍 Detailed Pipeline Overview

### 1. Frame Cropping & Manifest Creation
- Reads each clip’s **`output.json`** to map annotated (every‑third) frames → raw frames.
- Opens the color video, crops each pig’s bounding box per frame, and saves crops into `data_crops/{train,val}/{behaviour}/`.
- Builds a **CSV manifest** (`manifest.csv`) listing every crop path and its label.

### 2. Posture Model Training *(ResNet‑50)*
- Loads the manifest and filters to **lying** vs **standing** examples.
- Splits into train/val folds **by clip ID** to avoid leakage.
- Computes class‑balanced **sampler** weights *and* **loss** weights.
- Applies heavy data augmentation: random crop, flip, rotation, color‑jitter, random erasing.
- Fine‑tunes a **pre‑trained ResNet‑50** (2‑class head) with mixed precision, cosine LR scheduler, and early stopping.
- Logs train/val loss & accuracy each epoch and saves the **best checkpoint**.

### 3. Object Detector Training *(YOLOv8)*
- Converts ground‑truth boxes to YOLO TXT under `yolo/images/{train,val}` & `yolo/labels/{train,val}`.
- Writes `data.yaml` (1 class: *pig*) and trains a YOLOv8 detector.
- Exports best `.pt` weights and optionally **ONNX** for deployment.

### 4. Inference Pipeline (per clip)
**a. Pen Mask Reconstruction** – thresholds `background.png` to obtain the usable‑pen binary mask.

**b. Detection & Tracking** – runs YOLO on each resized frame; retains detections whose centroids fall **inside** the mask and maintains **MOSSE trackers** to bridge gaps.

**c. On‑the‑fly Behavior Classification**  
• **Moving:** compares centroid displacement over 2 s (via `depth_scale`) against a cm threshold.  
• **Eating / Drinking:** uses image‑moment orientation to check alignment toward feeder vs waterer coordinates.  
• **Lying / Standing:** batches crops through the ResNet posture model for posture prediction.

**d. JSON Export** – collates each tracklet’s per‑frame behavior changes into the required **`output.json`** format.

### 5. Evaluation & Reporting
- Compares each predicted JSON against ground truth per clip.
- Computes **AP**, *true‑positive %*, *false‑positive %*, and *missed‑detection %* for each sequence and overall.
- Writes an **`accuracy.txt`** summary matching Table 2 of the paper.

---

## 🤝 Contributing & License

PRs and issues are welcome! Code is released under the MIT License – see `LICENSE`.  

---
