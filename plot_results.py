import pandas as pd
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

# Setup directories
GRAPHS_DIR = Path("graphs")
TRAIN_GRAPHS_DIR = GRAPHS_DIR / "training"
EVAL_GRAPHS_DIR = GRAPHS_DIR / "evaluation"

GRAPHS_DIR.mkdir(exist_ok=True)
TRAIN_GRAPHS_DIR.mkdir(exist_ok=True)
EVAL_GRAPHS_DIR.mkdir(exist_ok=True)

# Set high-quality styling
plt.style.use("dark_background")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 1.2

# Find results.csv
result_files = list(Path(".").rglob("results.csv"))
if not result_files:
    print("ERROR: results.csv was not found.")
    exit(1)

results_path = result_files[0]
print(f"Found results.csv: {results_path}")
df = pd.read_csv(results_path)
print(f"Loaded {len(df)} epochs.")

# Strip whitespace from column names if any
df.columns = [c.strip() for c in df.columns]

# -----------------------------
# 1. Loss curves (Train vs Val)
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0b1120")
ax1.set_facecolor("#0f172a")
ax2.set_facecolor("#0f172a")

ax1.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", color="#38bdf8", linewidth=2.2, marker="o", markersize=4)
ax1.plot(df["epoch"], df["train/cls_loss"], label="Train Cls Loss", color="#818cf8", linewidth=2.2, marker="s", markersize=4)
ax1.plot(df["epoch"], df["train/dfl_loss"], label="Train DFL Loss", color="#34d399", linewidth=2.2, marker="^", markersize=4)
ax1.set_title("Training Losses by Epoch", fontsize=13, fontweight="bold", pad=12, color="#f8fafc")
ax1.set_xlabel("Epoch", fontsize=11, color="#94a3b8")
ax1.set_ylabel("Loss", fontsize=11, color="#94a3b8")
ax1.legend(facecolor="#1e293b", edgecolor="#334155")
ax1.grid(True, linestyle="--", alpha=0.3, color="#475569")

if "val/box_loss" in df.columns and "val/cls_loss" in df.columns:
    ax2.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss", color="#f43f5e", linewidth=2.2, marker="o", markersize=4)
    ax2.plot(df["epoch"], df["val/cls_loss"], label="Val Cls Loss", color="#fb923c", linewidth=2.2, marker="s", markersize=4)
    ax2.plot(df["epoch"], df["val/dfl_loss"], label="Val DFL Loss", color="#e879f9", linewidth=2.2, marker="^", markersize=4)
    ax2.set_title("Validation Losses by Epoch", fontsize=13, fontweight="bold", pad=12, color="#f8fafc")
    ax2.set_xlabel("Epoch", fontsize=11, color="#94a3b8")
    ax2.set_ylabel("Loss", fontsize=11, color="#94a3b8")
    ax2.legend(facecolor="#1e293b", edgecolor="#334155")
    ax2.grid(True, linestyle="--", alpha=0.3, color="#475569")

plt.tight_layout()
fig.savefig(TRAIN_GRAPHS_DIR / "loss_curves.png", dpi=300, facecolor=fig.get_facecolor())
fig.savefig(GRAPHS_DIR / "loss_curves.png", dpi=300, facecolor=fig.get_facecolor())
plt.close(fig)

# -----------------------------
# 2. Precision & Recall curves
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#0b1120")
ax.set_facecolor("#0f172a")

ax.plot(df["epoch"], df["metrics/precision(B)"], label="Precision (B)", color="#06b6d4", linewidth=2.5, marker="o", markersize=5)
ax.plot(df["epoch"], df["metrics/recall(B)"], label="Recall (B)", color="#10b981", linewidth=2.5, marker="s", markersize=5)

ax.set_title("Precision and Recall Convergence", fontsize=14, fontweight="bold", pad=14, color="#f8fafc")
ax.set_xlabel("Epoch", fontsize=11, color="#94a3b8")
ax.set_ylabel("Metric Score", fontsize=11, color="#94a3b8")
ax.set_ylim(-0.05, 1.05)
ax.legend(facecolor="#1e293b", edgecolor="#334155", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.3, color="#475569")

plt.tight_layout()
fig.savefig(TRAIN_GRAPHS_DIR / "precision_recall.png", dpi=300, facecolor=fig.get_facecolor())
fig.savefig(GRAPHS_DIR / "precision_recall.png", dpi=300, facecolor=fig.get_facecolor())
plt.close(fig)

# -----------------------------
# 3. mAP curves
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#0b1120")
ax.set_facecolor("#0f172a")

ax.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP @ 0.50 (B)", color="#6366f1", linewidth=2.5, marker="o", markersize=5)
ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP @ 0.50:0.95 (B)", color="#a855f7", linewidth=2.5, marker="^", markersize=5)

ax.set_title("Mean Average Precision (mAP) Progress", fontsize=14, fontweight="bold", pad=14, color="#f8fafc")
ax.set_xlabel("Epoch", fontsize=11, color="#94a3b8")
ax.set_ylabel("mAP Score", fontsize=11, color="#94a3b8")
ax.set_ylim(-0.05, 1.05)
ax.legend(facecolor="#1e293b", edgecolor="#334155", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.3, color="#475569")

plt.tight_layout()
fig.savefig(TRAIN_GRAPHS_DIR / "map_curves.png", dpi=300, facecolor=fig.get_facecolor())
fig.savefig(GRAPHS_DIR / "map_curves.png", dpi=300, facecolor=fig.get_facecolor())
plt.close(fig)

# -----------------------------
# 4. Multi-Panel Training Dashboard
# -----------------------------
fig, ((p1, p2), (p3, p4)) = plt.subplots(2, 2, figsize=(15, 10), facecolor="#0b1120")
for p in [p1, p2, p3, p4]:
    p.set_facecolor("#0f172a")
    p.grid(True, linestyle="--", alpha=0.25, color="#475569")

p1.plot(df["epoch"], df["train/box_loss"], label="Train Box", color="#38bdf8", linewidth=2)
p1.plot(df["epoch"], df["train/cls_loss"], label="Train Cls", color="#818cf8", linewidth=2)
p1.plot(df["epoch"], df["train/dfl_loss"], label="Train DFL", color="#34d399", linewidth=2)
p1.set_title("Training Loss Components", fontweight="bold", color="#f8fafc")
p1.set_ylabel("Loss", color="#94a3b8")
p1.legend(facecolor="#1e293b", edgecolor="#334155")

p2.plot(df["epoch"], df["metrics/precision(B)"], label="Precision", color="#06b6d4", linewidth=2)
p2.plot(df["epoch"], df["metrics/recall(B)"], label="Recall", color="#10b981", linewidth=2)
p2.set_title("Precision & Recall Evolution", fontweight="bold", color="#f8fafc")
p2.set_ylabel("Score", color="#94a3b8")
p2.set_ylim(-0.05, 1.05)
p2.legend(facecolor="#1e293b", edgecolor="#334155")

p3.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50", color="#6366f1", linewidth=2.2)
p3.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@50-95", color="#ec4899", linewidth=2.2)
p3.set_title("Validation mAP Evolution", fontweight="bold", color="#f8fafc")
p3.set_ylabel("mAP", color="#94a3b8")
p3.set_xlabel("Epoch", color="#94a3b8")
p3.set_ylim(-0.05, 1.05)
p3.legend(facecolor="#1e293b", edgecolor="#334155")

if "lr/pg0" in df.columns:
    p4.plot(df["epoch"], df["lr/pg0"], label="Learning Rate (pg0)", color="#f59e0b", linewidth=2)
    p4.set_title("Learning Rate Schedule", fontweight="bold", color="#f8fafc")
    p4.set_ylabel("LR", color="#94a3b8")
    p4.set_xlabel("Epoch", color="#94a3b8")
    p4.legend(facecolor="#1e293b", edgecolor="#334155")

plt.suptitle("AI Tower Detection Model - Training Performance Dashboard", fontsize=16, fontweight="bold", color="#f1f5f9", y=0.99)
plt.tight_layout()
fig.savefig(TRAIN_GRAPHS_DIR / "training_summary.png", dpi=300, facecolor=fig.get_facecolor())
plt.close(fig)

# -----------------------------
# 5. Copy evaluation & training assets
# -----------------------------
# Look for latest evaluation run
eval_dirs = sorted(list(Path("runs/detect/evaluation").glob("tower_model*")), key=lambda p: p.stat().st_mtime, reverse=True)
if eval_dirs:
    latest_eval = eval_dirs[0]
    print(f"Copying evaluation artifacts from: {latest_eval}")
    eval_files = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "val_batch0_pred.jpg",
        "val_batch1_pred.jpg"
    ]
    for ef in eval_files:
        src = latest_eval / ef
        if src.exists():
            shutil.copy2(src, EVAL_GRAPHS_DIR / ef)
            # also copy confusion matrix to top graphs dir
            if "confusion_matrix" in ef:
                shutil.copy2(src, GRAPHS_DIR / ef)
            print(f"  Copied {ef} to graphs/evaluation/")

# Copy training artifacts (labels, batches)
runs_train = Path("runs/detect/runs/tower_detection")
if runs_train.exists():
    for tf in ["labels.jpg", "train_batch0.jpg", "train_batch1.jpg", "train_batch2.jpg"]:
        src = runs_train / tf
        if src.exists():
            shutil.copy2(src, TRAIN_GRAPHS_DIR / tf)
            print(f"  Copied {tf} to graphs/training/")

print("\nSuccessfully organized all hackathon graphs in 'graphs/training' and 'graphs/evaluation'!")