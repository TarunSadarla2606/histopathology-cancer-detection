# Histopathologic Cancer Detection using Deep Learning

> *Detecting metastatic cancer in pathology image patches using a custom CNN — with an empirical study of how training data volume affects model performance.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/c/histopathologic-cancer-detection)
[![Course](https://img.shields.io/badge/UNT-SDAI%20Fall%202024-00853E)]()

**Group 2, Project 1 — Software Development for AI (Fall 2024), University of North Texas**

**Live Demo & Final Presentation (with deployment video):** [Google Drive](https://drive.google.com/drive/folders/1DAKgyJzvRk91gaYX1U1vtqYsDt35Ty07?usp=drive_link)

---

## Overview

Metastatic cancer — where primary tumour cells invade the lymph nodes — is a leading cause of cancer-related mortality. Digital pathology imaging produces whole-slide images (WSIs) of lymph node tissue, which pathologists manually examine to identify cancerous regions. This process is both **time-consuming** (each slide can take 10+ minutes) and **prone to error** under fatigue.

This project develops an automated AI pipeline for binary classification of **96×96 pixel pathology image patches** as cancerous or non-cancerous, using the [PatchCamelyon (PCam) dataset](https://www.kaggle.com/c/histopathologic-cancer-detection).

**My contribution** centres on a systematic empirical study: does more training data always improve model performance, and if so, how much data is *enough*? I designed a stratified 4-subset experiment (20/40/60/100% of the dataset) to answer this directly.

---

## My Contributions (Model Developer — Approach 1)

### 1. Noise Filtering Pipeline

Before any training, I scanned all 220,025 raw image patches using PIL image verification to identify corrupt, blank, or unreadable files. This found **29 noisy images** that would have silently degraded training quality. The cleaned label file (`data/clean_train_labels.csv`) with **219,996 valid samples** was used for all subsequent experiments.

```
Total raw samples:    220,025
Noisy images removed:      29  (0.013%)
Clean dataset:        219,996
```

### 2. Stratified Subset Design

To empirically study how training data volume affects model performance, I created four stratified subsets using scikit-learn's `train_test_split` with `stratify=label`. Each subset maintains the original ~60% non-cancerous / ~40% cancerous class ratio:

| Subset | % of Data | Total Samples | Train    | Validation | Test   |
|--------|-----------|---------------|----------|------------|--------|
| S1     | 20%       | 43,999        | 37,399   | 3,300      | 3,300  |
| S2     | 40%       | 87,998        | 74,798   | 6,600      | 6,600  |
| S3     | 60%       | 131,997       | 112,197  | 9,900      | 9,900  |
| S4     | 100%      | 219,996       | 186,796  | 16,600     | 16,600 |

Class balance per subset (approximately maintained):
- Non-cancerous (0): ~59.5% of each subset
- Cancerous (1): ~40.5% of each subset

### 3. Custom 5-Layer CNN Architecture

```
Input: 96×96 RGB → CenterCrop(32×32) → ToTensor()

Block 1:  Conv(3→32,   k=3, p=1) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.25)
Block 2:  Conv(32→64,  k=3, p=1) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.25)
Block 3:  Conv(64→128, k=3, p=1) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.25)
Block 4:  Conv(128→256,k=3, p=1) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.25)
Block 5:  Conv(256→512,k=3, p=1) + BatchNorm + ReLU + AdaptiveAvgPool(1) + Dropout(0.25)
Head:     Flatten → Linear(512→256) + ReLU → Linear(256→1)
Output:   Raw logit → BCEWithLogitsLoss
```

**Training configuration:**
- Loss: `BCEWithLogitsLoss` (numerically stable for imbalanced binary classification)
- Optimizer: Adam (lr=0.001, β₁=0.9, β₂=0.999, ε=1e-7)
- Epochs: 30 | Batch size: 64

### 4. Evaluation: ROC-AUC, PR-AUC, Sensitivity, Specificity

For each of the four subsets, I evaluated using clinically relevant metrics: ROC-AUC (overall discriminability), PR-AUC (performance under class imbalance), and the sensitivity/specificity trade-off critical in cancer screening.

### 5. VGG16 Exploration (Partial)

I began a VGG16 fine-tuning approach as comparison against the from-scratch CNN. The architecture (`notebooks/vgg16_exploration.ipynb`) modifies VGG16 for 32×32 input and fine-tunes with the first 10 layers frozen. This was not completed before the submission deadline.

---

## Results

### Quantitative Results — Custom CNN

| Subset | Train Size | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1    | CM (TN/FP/FN/TP) |
|--------|-----------|---------|--------|-------------|-------------|-------|------------------|
| S1 (20%) | 37,399  | **0.8108** | **0.6941** | 0.758 | 0.864 | 0.658 | 2281/359/160/500 |
| S2 (40%) | 74,798  | 0.7888  | 0.7088 | 0.647 | 0.931 | 0.672 | 4914/366/466/854 |
| S3 (60%) | 112,197 | —*      | —      | 0.693 | 0.929 | 0.669 | 7357/563/608/1372 |
| S4 (100%)| 186,796 | **0.8228** | **0.7029** | 0.797 | 0.849 | 0.662 | 11201/1998/670/2629 |

*S3 was loaded from saved weights; ROC-AUC not recomputed at test time.

### Key Findings

**Full dataset (S4) achieves the best overall discrimination** — ROC-AUC peaks at 0.8228, confirming that more data improves the model's ability to separate cancerous from non-cancerous patches at any threshold.

**The sensitivity-specificity trade-off shifts with data volume** — S2 shows unusually high specificity (0.931) with lower sensitivity (0.647), suggesting the model is conservative with the 40% subset. S4 rebalances this to sensitivity=0.797 / specificity=0.849 — more appropriate for cancer screening where false negatives are clinically costly.

**PR-AUC plateaus across subsets** — PR-AUC values (0.69–0.71) are notably lower than ROC-AUC, reflecting the class imbalance (~40% cancerous). This gap highlights the importance of reporting PR-AUC alongside ROC-AUC for imbalanced medical datasets.

**Non-monotonic improvement** — ROC-AUC dips at S2 and S3 before recovering at S4, suggesting the 32×32 centre-crop may limit the model's ability to leverage additional data. Larger patch sizes or multi-scale inputs would likely show stronger monotonic improvement.

### Result Figures

| Figure | Description |
|--------|-------------|
| [`fig_roc_pr_auc.png`](results/figures/fig_roc_pr_auc.png) | ROC-AUC and PR-AUC across all 4 subsets |
| [`fig_metrics_all_subsets.png`](results/figures/fig_metrics_all_subsets.png) | Sensitivity, specificity, precision, F1, accuracy — all subsets |
| [`fig_confusion_matrices.png`](results/figures/fig_confusion_matrices.png) | Confusion matrices (2×2 grid) for all 4 subsets |
| [`fig_data_volume_study.png`](results/figures/fig_data_volume_study.png) | Training size vs. all metrics — the core data volume study |
| [`fig_dataset_composition.png`](results/figures/fig_dataset_composition.png) | Subset composition and noise filtering summary |

---

## Repository Structure

```
histopathology-cancer-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                              ← Standalone Python modules
│   ├── dataset.py     — noise filtering, subset creation, PCamDataset, DataLoaders
│   ├── model.py       — CustomCNN (5-layer) and VGG16Cancer architectures
│   ├── train.py       — full training loop with Adam, BCEWithLogitsLoss, argparse CLI
│   └── evaluate.py    — ROC-AUC, PR-AUC, confusion matrix, ROC/PR curve plots
│
├── notebooks/
│   ├── cancercnn_early.ipynb    ← Early prototype (original labels, batch=32)
│   ├── cnn_v1.ipynb             ← CNN v1 (clean labels, 3 subsets)
│   ├── cnn_v2_final.ipynb       ← Final — all 4 subsets, full evaluation metrics
│   └── vgg16_exploration.ipynb  ← VGG16 fine-tuning exploration (partial)
│
├── data/
│   ├── clean_train_labels.csv   ← Noise-filtered label file (219,996 samples)
│   └── README.md                ← Dataset download instructions
│
├── architecture/
│   ├── model_architecture.drawio ← CNN architecture diagram (draw.io)
│   ├── sw_architecture.drawio    ← Software architecture diagram (draw.io)
│   ├── sw_architecture.png       ← SW architecture (PNG)
│   └── sw_architecture.pdf       ← SW architecture (PDF)
│
├── results/
│   ├── experiment_results.csv   ← All metrics for 4 subsets
│   └── figures/
│       ├── fig_roc_pr_auc.png
│       ├── fig_metrics_all_subsets.png
│       ├── fig_confusion_matrices.png
│       ├── fig_data_volume_study.png
│       └── fig_dataset_composition.png
│
└── docs/
    ├── graded_report.pdf          ← Annotated graded report (HW5)
    ├── proposal.pdf               ← Group 2 project proposal (HW2)
    ├── hw1_initial_analysis.pdf   ← Individual initial analysis (HW1)
    ├── hw3_individual_report.pdf  ← Individual progress report (HW3)
    ├── midterm_presentation.pptx  ← Midterm slides (HW3)
    └── system_documentation.pptx ← System documentation deck
```

---

## Notebook Guide

The three CNN notebooks are preserved intentionally — they show the project's evolution:

| Notebook | Purpose | Key difference |
|----------|---------|----------------|
| `cancercnn_early.ipynb` | First prototype | Original noisy labels, batch=32, no clean subset pipeline |
| `cnn_v1.ipynb` | Refined version | Clean labels, batch=64, 3-subset pipeline |
| `cnn_v2_final.ipynb` | **Main notebook** | All 4 subsets, full metrics (ROC-AUC, PR-AUC, confusion matrices), S3 loaded from saved weights |

Comparing the three notebooks directly illustrates why noise filtering mattered and how the subset pipeline was refined.

---

## Quickstart

```bash
git clone https://github.com/TarunSadarla2606/histopathology-cancer-detection.git
cd histopathology-cancer-detection
pip install -r requirements.txt
```

**Using the Python modules:**
```python
from src.model import CustomCNN
from src.dataset import build_loaders

model = CustomCNN()
train_loader, val_loader, test_loader = build_loaders(
    'train_4.csv', 'val_4.csv', 'test_4.csv',
    data_dir='/path/to/pcam/train/'
)
```

**Training from CLI:**
```bash
python src/train.py --subset 4 \
                    --data_dir /path/to/pcam/train/ \
                    --label_dir /kaggle/working \
                    --epochs 30 \
                    --save_path weights_s4.pth
```

**Evaluation:**
```bash
python src/evaluate.py --weights weights_s4.pth \
                        --test_csv test_4.csv \
                        --data_dir /path/to/pcam/train/ \
                        --output_dir results/
```

**Kaggle paths:**
```
/kaggle/input/histopathologic-cancer-detection/train/  ← raw .tif patches
/kaggle/input/hcd-clean/clean_train_labels.csv         ← cleaned labels
```

---

## Team Contributions

| Member | Role | Contributions |
|--------|------|---------------|
| **Tarun Sadarla** | Model Developer (Approach 1) | Noise filtering, stratified 4-subset design, custom CNN (all subsets), ROC-AUC/PR-AUC evaluation, VGG16 exploration |
| Group teammates | Approach 2 | Swin Transformer implementation and evaluation |

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
tqdm>=4.65.0
```

---

*CSCE 5931 — Software Development for AI, University of North Texas, Fall 2024*
