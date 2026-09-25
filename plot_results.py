import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Create output folder
OUTPUT_DIR = Path("graphs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Find results.csv
result_files = list(Path(".").rglob("results.csv"))

if not result_files:
    print("ERROR: results.csv was not found.")
    exit()

results_path = result_files[0]

print(f"Found results.csv: {results_path}")

# Read CSV
df = pd.read_csv(results_path)

print(f"Loaded {len(df)} epochs.")
print("Generating graphs...")

# -----------------------------
# 1. Loss curves
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
plt.plot(df["epoch"], df["train/cls_loss"], label="Train Class Loss")
plt.plot(df["epoch"], df["train/dfl_loss"], label="Train DFL Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curves")
plt.legend()
plt.grid(True)

plt.savefig(OUTPUT_DIR / "loss_curves.png", dpi=300)
plt.close()

# -----------------------------
# 2. Precision and Recall
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(
    df["epoch"],
    df["metrics/precision(B)"],
    label="Precision"
)

plt.plot(
    df["epoch"],
    df["metrics/recall(B)"],
    label="Recall"
)

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Precision and Recall")
plt.legend()
plt.grid(True)

plt.savefig(OUTPUT_DIR / "precision_recall.png", dpi=300)
plt.close()

# -----------------------------
# 3. mAP curves
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(
    df["epoch"],
    df["metrics/mAP50(B)"],
    label="mAP@50"
)

plt.plot(
    df["epoch"],
    df["metrics/mAP50-95(B)"],
    label="mAP@50-95"
)

plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.title("mAP Curves")
plt.legend()
plt.grid(True)

plt.savefig(OUTPUT_DIR / "map_curves.png", dpi=300)
plt.close()

print("Graphs generated successfully!")
print(f"Saved in: {OUTPUT_DIR.resolve()}")