# Data

## clean_train_labels.csv

This file contains the noise-filtered label file for the PatchCamelyon dataset.

**Source:** [Kaggle — Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection)

**How it was created:**
1. All 220,025 training images were scanned using PIL to detect corrupt, blank, or unreadable patches
2. 29 noisy/unreadable images were identified and excluded
3. The resulting clean label file contains **219,996 valid (id, label) pairs**

**Columns:**
- `id` — image filename (without .tif extension)
- `label` — 0 (non-cancerous) or 1 (cancerous)

**Class distribution (full 219,996 samples):**
- Non-cancerous (0): ~130,908 (~59.5%)
- Cancerous (1): ~89,088 (~40.5%)

## Raw Images

The raw 96×96 TIFF image patches are **not included** in this repo due to size (~7GB).

Download from Kaggle:
```bash
kaggle competitions download -c histopathologic-cancer-detection -p ./data/raw --unzip
```

Expected structure after download:
```
data/raw/train/    ← ~220K .tif image patches
data/raw/test/     ← test patches
```
