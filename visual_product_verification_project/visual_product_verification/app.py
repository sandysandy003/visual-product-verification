import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QHBoxLayout, QVBoxLayout, QFormLayout, QComboBox,
    QMessageBox, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView
)

BASE = Path(__file__).resolve().parent
CATALOG_FILE = BASE / "catalog" / "products.csv"
REFERENCE_DIR = BASE / "catalog" / "reference_images"
MODEL_FILE = BASE / "models" / "product_classifier.pt"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_catalog(path=CATALOG_FILE):
    """Return a barcode -> product dictionary from the local CSV catalog."""
    products = {}
    if not Path(path).exists():
        return products
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            barcode = row.get("barcode", "").strip()
            if barcode:
                products[barcode] = {
                    "barcode": barcode,
                    "product_name": row.get("product_name", "").strip(),
                    "brand": row.get("brand", "").strip(),
                    "category": row.get("category", "").strip(),
                    "reference_image": row.get("reference_image", "").strip(),
                }
    return products


def orb_similarity(image_a, image_b):
    """A lightweight visual baseline: compare local ORB features between two images."""
    if image_a is None or image_b is None:
        return 0.0
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY) if image_a.ndim == 3 else image_a
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY) if image_b.ndim == 3 else image_b
    orb = cv2.ORB_create(nfeatures=1200)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des_a, des_b), key=lambda m: m.distance)
    good = [m for m in matches if m.distance < 55]
    denominator = max(1, min(len(kp_a), len(kp_b)))
    return min(1.0, len(good) / denominator * 3.0)


def save_audit(row, path=None):
    path = Path(path) if path else OUTPUT_DIR / "verification_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = [
        "timestamp", "barcode_entered", "catalog_product", "visual_prediction",
        "visual_score", "status", "image_file"
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})
    return path


class ProductVerifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Product Verification | Computer Vision")
        self.resize(1180, 820)
        self.catalog = load_catalog()
        self.image = None
        self.image_path = ""
        self.camera = None
        self.last_prediction = ("No visual match", 0.0)
        self.last_saved_path = ""

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        title = QLabel("VISUAL PRODUCT VERIFICATION")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 23px; font-weight: 700; padding: 12px;")
        root.addWidget(title)

        self.preview = QLabel("Load a product image or capture one from your webcam")
        self.preview.setMinimumSize(700, 380)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#20252b;color:#f2f2f2;border:1px solid #555;border-radius:8px;")
        root.addWidget(self.preview, 1)

        image_controls = QHBoxLayout()
        self.open_btn = QPushButton("Open Product Image")
        self.camera_btn = QPushButton("Capture Webcam Image")
        self.open_btn.clicked.connect(self.open_image)
        self.camera_btn.clicked.connect(self.capture_webcam)
        image_controls.addWidget(self.open_btn)
        image_controls.addWidget(self.camera_btn)
        root.addLayout(image_controls)

        panel = QGroupBox("Checkout verification")
        form = QFormLayout(panel)
        self.barcode_input = QLineEdit()
        self.barcode_input.setPlaceholderText("Enter scanned barcode (or type a demo barcode from products.csv)")
        self.verify_btn = QPushButton("Verify Product")
        self.verify_btn.clicked.connect(self.verify_product)
        barcode_row = QHBoxLayout()
        barcode_row.addWidget(self.barcode_input, 1)
        barcode_row.addWidget(self.verify_btn)
        form.addRow("Scanned barcode:", barcode_row)

        self.visual_label = QLabel("Visual prediction: not analyzed")
        self.catalog_label = QLabel("Catalog product: not looked up")
        self.status_label = QLabel("Status: Ready")
        self.status_label.setWordWrap(True)
        form.addRow("Visual result:", self.visual_label)
        form.addRow("Catalog result:", self.catalog_label)
        form.addRow("Decision:", self.status_label)
        root.addWidget(panel)

        self.history = QTableWidget(0, 5)
        self.history.setHorizontalHeaderLabels(["Time", "Barcode", "Catalog product", "Visual result", "Status"])
        self.history.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history.setMaximumHeight(170)
        root.addWidget(self.history)

        footer = QLabel(
            "Prototype note: ORB matching is a baseline, not a calibrated probability. "
            "For robust SKU-level recognition, train the optional custom classifier on your own catalog images."
        )
        footer.setWordWrap(True)
        footer.setStyleSheet("color:#555;padding:4px;")
        root.addWidget(footer)

    def show_image(self, image):
        if image is None:
            return
        self.image = image.copy()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview.setPixmap(pixmap)
        self.last_prediction = self.predict_visual_product(image)
        self.visual_label.setText(
            f"{self.last_prediction[0]} (visual score: {self.last_prediction[1]:.3f})"
        )

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select product image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            QMessageBox.warning(self, "Image error", "Could not read the selected image.")
            return
        self.image_path = path
        self.show_image(image)

    def capture_webcam(self):
        cap = cv2.VideoCapture(0)
        try:
            if not cap.isOpened():
                QMessageBox.warning(self, "Camera error", "Could not open webcam. Check camera permissions.")
                return
            ok, frame = cap.read()
            if not ok:
                QMessageBox.warning(self, "Camera error", "Camera opened but no frame was received.")
                return
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = OUTPUT_DIR / f"capture_{stamp}.jpg"
            cv2.imwrite(str(path), frame)
            self.image_path = str(path)
            self.show_image(frame)
        finally:
            cap.release()

    def predict_visual_product(self, image):
        # Optional custom deep-learning classifier. Train it using train_classifier.py.
        if MODEL_FILE.exists():
            try:
                from ultralytics import YOLO
                model = YOLO(str(MODEL_FILE))
                result = model.predict(image, imgsz=224, verbose=False)[0]
                if result.probs is not None:
                    idx = int(result.probs.top1)
                    name = str(result.names[idx])
                    score = float(result.probs.top1conf)
                    return name, score
            except Exception as exc:
                print(f"Custom classifier unavailable; using ORB baseline: {exc}")

        # Baseline mode compares the captured image with registered reference images.
        best_name, best_score = "Unknown product", 0.0
        for product in self.catalog.values():
            ref_name = product.get("reference_image", "").strip()
            if not ref_name:
                continue
            ref_path = REFERENCE_DIR / ref_name
            if not ref_path.exists():
                continue
            ref = cv2.imread(str(ref_path))
            score = orb_similarity(image, ref)
            if score > best_score:
                best_score = score
                best_name = product["product_name"] or product["barcode"]
        return best_name, best_score

    def verify_product(self):
        if self.image is None:
            QMessageBox.information(self, "Image required", "Open or capture a product image first.")
            return
        barcode = self.barcode_input.text().strip()
        if not barcode:
            QMessageBox.information(self, "Barcode required", "Enter a barcode before verifying.")
            return

        product = self.catalog.get(barcode)
        catalog_name = product["product_name"] if product else "Barcode not found in local catalog"
        visual_name, visual_score = self.last_prediction
        # ORB scores are similarity heuristics, not probabilities. Use conservative manual-review logic.
        if not product:
            status = "REVIEW: barcode unknown"
        elif visual_name == "Unknown product" or visual_score < 0.03:
            status = "REVIEW: visual match uncertain"
        elif visual_name.casefold() == catalog_name.casefold():
            status = "MATCH: names agree; confirm operationally"
        else:
            status = "REVIEW: possible mismatch"

        self.catalog_label.setText(f"{catalog_name} | Brand: {product['brand'] if product else '-'}")
        self.status_label.setText(status)

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "barcode_entered": barcode,
            "catalog_product": catalog_name,
            "visual_prediction": visual_name,
            "visual_score": round(float(visual_score), 5),
            "status": status,
            "image_file": self.image_path,
        }
        log_path = save_audit(row)
        r = self.history.rowCount()
        self.history.insertRow(r)
        values = [row["timestamp"], barcode, catalog_name, f"{visual_name} ({visual_score:.3f})", status]
        for c, value in enumerate(values):
            self.history.setItem(r, c, QTableWidgetItem(str(value)))
        self.status_label.setText(f"{status} | Audit saved: {log_path}")

    def closeEvent(self, event):
        if self.camera is not None:
            self.camera.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = ProductVerifier()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
