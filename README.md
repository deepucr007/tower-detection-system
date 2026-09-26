# 📡 AI Tower Detection System

An enterprise-grade computer vision system and interactive web dashboard engineered for telecommunications tower inspection, structural hazard prevention, and aerial imagery analysis.

---

## 🏆 Model Performance Summary (Real Validation Data)

| Metric | Score | Evaluation Note |
| :--- | :--- | :--- |
| **Precision** | **94.64%** | Highly specific detections; virtually eliminates false alarms. |
| **Recall** | **100.00%** | **Zero missed towers** on held-out validation; critical for safety audits. |
| **F1-Score** | **97.25%** | Near-optimal balance between precision and recall. |
| **mAP @ 0.50** | **98.02%** | High bounding box localization accuracy across tower classes. |
| **mAP @ 0.50:0.95** | **72.79%** | Robust boundary localization under strict IoU thresholds. |
| **Inference Latency** | **~25-35 ms** | Real-time edge-deployable inference powered by YOLOv11. |

---

## 🚀 Interactive Web Dashboard

The dashboard strictly implements the required workflow:

```
[1. Image Upload] ➔ [2. Upload Progress] ➔ [3. Execute Trigger] ➔ [4. Quality Check] ➔ [5. Object Detection] ➔ [6. Completion Notification] ➔ [7. Output Display]
```

### Dashboard Features
1. **Multi-Format Upload with Visual Progress**: Supports drag-and-drop JPG, JPEG, and PNG images with percentage progress bar.
2. **Quick Test Suite**: 1-click test buttons for Valid Monopole, Valid Supporting Tower, and rejected edge cases (Blurry, Dark, Overexposed).
3. **Execution Gate**: Processing begins only upon clicking `⚡ EXECUTE DETECTION`.
4. **Automated Quality Filtering Guardrail**:
   - **Laplacian Blur Variance Filter** ($>80.0$)
   - **Underexposure Luminance Floor** ($\ge 45.0$)
   - **Overexposure Highlight Ceiling** ($\le 210.0$)
   - *Halt Guarantee*: Unsuitable photographs are explicitly rejected and barred from model inference.
5. **Completion Notification**: Real-time notification banner reporting total execution time, latency, and detection counts.
6. **Dedicated Output Section**: Side-by-side comparison between original input image and high-contrast annotated detection output.
7. **Per-Class Average Confidence**: Dynamic analytics cards calculating and displaying the average confidence score for each detected class.
8. **Export Suite**: 1-click download for detection coordinates (CSV) and annotated high-resolution images (PNG).

---

## 📁 Repository Structure

```
tower-detection-system/
├── app.py                          # Web Dashboard (Single focused flow)
├── quality.py                      # Pre-Flight Image Quality Filter
├── detector.py                     # YOLO Model Wrapper & Latency Tracker
├── config.py                       # Configuration & Default Thresholds
├── evaluate_model.py               # Model Evaluation Script
├── plot_results.py                 # Graph Generation & Asset Organizer
├── test_pipeline.py                # Standalone End-to-End CLI Pipeline Test
├── test_quality.py                 # Standalone Image Quality Filter Test
├── train.py                        # YOLO Model Training Script
├── requirements.txt                # Python Dependencies
├── run_dashboard.bat               # 1-Click Launch Script
├── README.md                       # Documentation
│
├── models/
│   └── best.pt                     # Best Model Checkpoint (mAP50: 98.02%)
│
├── sample_images/                  # Curated Demo Suite
│   ├── good_monopole.jpg           # Valid monopole tower
│   ├── good_supporting.jpg         # Valid supporting tower
│   ├── test_blur.jpg               # Blurry sample (Fails Quality Check)
│   ├── test_dark.jpg               # Underexposed sample (Fails Quality Check)
│   └── test_bright.jpg             # Overexposed sample (Fails Quality Check)
│
├── graphs/
│   ├── training/                   # Training Loss & Convergence Curves
│   └── evaluation/                 # Confusion Matrix & Validation Predictions
│
├── evaluation/
│   └── evaluation_report.json      # Quantitative JSON metrics
│
└── dataset/                        # Annotated Training & Validation Dataset
    ├── data.yaml
    ├── images/
    └── labels/
```

---

## ⚡ How to Run

### 1. Launching the Dashboard
- **Option A (1-Click)**: Double-click `run_dashboard.bat`.
- **Option B (Terminal)**:
  ```powershell
  .\venv\Scripts\python.exe -m streamlit run app.py
  ```
- **Browser URL**: `http://127.0.0.1:8501` or `http://localhost:8501`

### 2. Standalone CLI Pipeline Tests
```powershell
.\venv\Scripts\python.exe test_pipeline.py sample_images/good_monopole.jpg
.\venv\Scripts\python.exe test_pipeline.py sample_images/test_blur.jpg
```
