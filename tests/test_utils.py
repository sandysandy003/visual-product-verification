import csv
from pathlib import Path
from app import load_catalog, save_audit, orb_similarity


def test_load_catalog_reads_demo_products():
    catalog = load_catalog()
    assert "8900000000011" in catalog
    assert catalog["8900000000011"]["product_name"] == "Demo Coffee Packet"


def test_save_audit_creates_csv(tmp_path):
    target = tmp_path / "audit.csv"
    row = {
        "timestamp": "2026-01-01T10:00:00",
        "barcode_entered": "123",
        "catalog_product": "Example",
        "visual_prediction": "Example",
        "visual_score": 0.7,
        "status": "MATCH",
        "image_file": "sample.jpg",
    }
    save_audit(row, target)
    with open(target, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["barcode_entered"] == "123"


def test_orb_similarity_returns_zero_for_missing_images():
    assert orb_similarity(None, None) == 0.0
