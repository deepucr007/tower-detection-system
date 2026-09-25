# AI Tower Detection Project

Classes:
- 0: monopole tower
- 1: supporting tower

The organiser classes `non recognizable` and `partial tower` were excluded.

Generated dataset:
- Train images: 160
- Validation images: 40

## Next steps

Open a terminal in this folder and run:
```powershell
python -m pip install -r requirements.txt
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
python train.py
```

After training, copy the best model checkpoint to:
`models/best.pt`

Then run the Streamlit application after `app.py` is created.
