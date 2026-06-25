from pathlib import Path
from ultralytics import YOLO
print(2)

project_dir = Path(__file__).resolve().parent.parent.parent.parent
PATH = project_dir / "Dataset" / "DS_Y8n-obb"
yaml_path = PATH / "data.yaml"
print(yaml_path)

model = YOLO("yolov8n-obb.pt")
print(1)

results = model.train(
    data=str(yaml_path),
    epochs=100,
    imgsz=640,
    batch=12,
    device=0,
    workers=4,
    patience=20,
    save=True,
    project="dm_detector",
    name="yolo8n_dm",
    amp=False,
    exist_ok=True,
    task='obb',
    degrees=0.0,
)

metrics = model.val()

print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP75: {metrics.box.map75:.4f}")