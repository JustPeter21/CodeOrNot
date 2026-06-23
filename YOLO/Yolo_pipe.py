from pathlib import Path
from ultralytics import YOLO

yaml_path = Path("./dataset") / "data.yaml"

model = YOLO("yolo11n.pt")

results = model.train(
    data=str(yaml_path),
    epochs=30,
    imgsz=640,
    batch=12,
    device=0,
    workers=4,
    patience=20,
    save=True,
    project="dm_detector",
    name="yolo11n_dm",
    amp=False,
    exist_ok=True
)

metrics = model.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")