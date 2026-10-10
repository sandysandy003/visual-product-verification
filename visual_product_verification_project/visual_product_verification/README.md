# Visual Product Verification for Automated Checkout

Desktop prototype for comparing a visually recognized product with a scanned barcode and logging potential mismatches.

## Features

- PySide6 desktop GUI
- Load product images or capture a webcam frame
- Local barcode-to-product catalog
- Baseline visual matching with OpenCV ORB reference-image features
- Optional custom YOLO image-classification model
- Match/review status and CSV audit log
- Unit tests

## Important limitation

The included ORB method is a visual similarity baseline, not a calibrated confidence probability. Retail packaging recognition needs controlled image capture, multiple reference images, a product-specific dataset, and validation. Do not use the prototype as the sole authority for charging customers or alleging fraud.

## Install (Windows PowerShell)

Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

## Run the desktop application

1. Run `python app.py`.
2. Open a product image or capture a single webcam image.
3. Enter a barcode from `catalog/products.csv`.
4. Click Verify Product.
5. Check the status and `outputs/verification_log.csv`.

The three rows in the CSV are demo catalog entries only. Replace them with products you are authorized to use. The `reference_image` field must be the filename of an image placed in `catalog/reference_images/`.

## Build a useful reference catalog

For each SKU:
1. Capture several images from realistic checkout angles and lighting.
2. Use varied backgrounds, rotations, distances, and partial occlusions.
3. Keep a separate set of images for validation.
4. Add one chosen reference filename to `products.csv` for baseline ORB matching.
5. For deep-learning training, arrange the images into the folder structure below.

## Custom deep-learning classifier

Create a dataset with at least two images per class for a smoke test; for meaningful evaluation, gather substantially more varied images per SKU.

```text
dataset/
├── train/
│   ├── Demo_Coffee_Packet/
│   ├── Demo_Tea_Packet/
│   └── Demo_Biscuit_Packet/
└── val/
    ├── Demo_Coffee_Packet/
    ├── Demo_Tea_Packet/
    └── Demo_Biscuit_Packet/
```

Folder names become class names. Avoid leakage: near-duplicate images of the same capture session should not be split across train and validation.

Train:

```powershell
python train_classifier.py
```

After training, copy the best checkpoint to:

```text
models/product_classifier.pt
```

The app will use that classifier when the file exists; otherwise it uses the ORB baseline. The trained model's class names should match the catalog's `product_name` values (case-insensitive) for direct comparison.

## Barcode lookup extension

The included application uses a local catalog to keep checkout verification deterministic. An optional extension is to look up supported food barcodes through the Open Food Facts API. API catalog coverage can be incomplete, so keep a local override and treat missing entries as unknown rather than assuming a product is invalid.

Official API documentation: https://openfoodfacts.github.io/documentation/docs/Product-Opener/api/tutorial-off-api/

## Test

```powershell
pytest -q
```

These are utility-level tests. Also manually test correct matches, wrong barcodes, unknown barcodes, unclear images, and different lighting.

## GitHub upload

Create an empty GitHub repository, then run these commands from the project folder:

```powershell
git init
git add .
git commit -m "Build visual product verification desktop app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/visual-product-verification.git
git push -u origin main
```

Replace `YOUR_USERNAME` and the remote URL with your own GitHub repository details. Do not commit private product photos, customer data, model weights, or output logs.

## Suggested industry-level extensions

- Multi-image reference bank per SKU
- Barcode decoding with a supported scanner or a tested barcode library
- Embedding-based visual retrieval (e.g., a pretrained vision encoder) and calibrated thresholds
- Product database backed by PostgreSQL
- FastAPI inference service and Docker packaging
- Confusion matrix, top-k accuracy, per-SKU precision/recall, latency benchmarking
- Human review workflow and audit trail

## Licensing

Review Ultralytics licensing terms before commercial distribution: https://www.ultralytics.com/license
