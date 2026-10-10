"""Train a custom YOLO image classifier from a folder-structured product dataset.

Expected dataset:
dataset/
  train/
    Product_A/
      image1.jpg
    Product_B/
      image2.jpg
  val/
    Product_A/
    Product_B/

Use product names as folder names. Keep train/val images separate.
"""
from pathlib import Path
from ultralytics import YOLO

DATASET = Path("dataset")
if not (DATASET / "train").exists() or not (DATASET / "val").exists():
    raise SystemExit(
        "Dataset not found. Create dataset/train/<product_name>/ and "
        "dataset/val/<product_name>/ folders first. See README.md."
    )

model = YOLO("yolo11n-cls.pt")
model.train(data=str(DATASET.resolve()), epochs=40, imgsz=224, batch=16)
print("Training complete. Find the best checkpoint under runs/classify/train/weights/best.pt")
print("Copy best.pt to models/product_classifier.pt for the app to use it.")
