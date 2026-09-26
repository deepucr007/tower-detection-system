import os
import shutil
import time
from pathlib import Path

# Skip polars CPU check on Sandy Bridge CPU
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from ultralytics import YOLO

def main():
    print("=" * 60)
    print("FAST FINE-TUNING ON CPU: FREEZE BACKBONE + HEAD ADAPTATION")
    print("=" * 60)
    
    t0 = time.time()
    model = YOLO("models/best.pt")
    
    print("\nStarting fast fine-tuning on new tower samples...")
    results = model.train(
        data="dataset/data.yaml",
        epochs=3,
        imgsz=416,
        batch=16,
        freeze=10,  # Freeze backbone to speed up CPU training 4x
        project="runs/detect/runs",
        name="tower_fast_finetune",
        optimizer="AdamW",
        lr0=0.001,
        verbose=True
    )
    
    print(f"\nFine-tuning completed in {round(time.time() - t0, 1)} seconds.")
    
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy2(str(best_weights), "models/best.pt")
        print(f"Updated models/best.pt with new weights from {best_weights}!")
    else:
        print("Warning: new weights file not found, retaining existing models/best.pt.")

if __name__ == "__main__":
    main()
