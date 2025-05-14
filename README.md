# 🐷 Pig Behavior Detection System

Detect *eating, drinking, lying, moving,* and *standing* behaviors of pigs from video‑frame sequences recorded in commercial pens.

---

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

## 🏗️ Project Structure
