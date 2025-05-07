import subprocess
from pathlib import Path

try:
    import ultralytics
except ImportError:
    print("📦 Installing 'ultralytics' package...")
    subprocess.check_call(["pip", "install", "ultralytics"])

project_root = Path(__file__).resolve().parent.parent
dataset_yaml = project_root / "yolo_dataset" / "dataset.yaml"

if not dataset_yaml.exists():
    raise FileNotFoundError(f"❌ dataset.yaml not found at: {dataset_yaml}")

train_cmd = [
    "yolo", "train",
    "model=yolov8n.pt",
    f"data={dataset_yaml.as_posix()}",
    "epochs=50",
    "imgsz=640",
    "batch=16"
]
print(f"🚀 Starting YOLO training with dataset: {dataset_yaml}")
subprocess.run(train_cmd)