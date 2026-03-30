"""
dataset.py
----------
Dataset loading and stratified subset creation for the PatchCamelyon (PCam)
histopathologic cancer detection task.

PCam Dataset (Kaggle):
    220,025 labelled 96×96 RGB image patches from whole-slide images.
    Binary labels: 0 = non-cancerous, 1 = cancerous (metastatic).

After noise filtering: 219,996 valid samples (29 noisy images removed).

Four stratified subsets created to study the effect of data volume:
    Subset 1 (20%):  43,999 samples  → Train 37,399 | Val 3,300 | Test 3,300
    Subset 2 (40%):  87,998 samples  → Train 74,798 | Val 6,600 | Test 6,600
    Subset 3 (60%): 131,997 samples  → Train 112,197 | Val 9,900 | Test 9,900
    Subset 4 (100%): 219,996 samples → Train 186,796 | Val 16,600 | Test 16,600
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# ── Constants ──────────────────────────────────────────────────────
TOTAL_CLEAN_SAMPLES = 219_996
SUBSET_PERCENTAGES  = [0.20, 0.40, 0.60, 1.00]
VAL_TEST_SPLIT      = 0.15   # 15% val, 15% test of each subset
TRANSFORM_DEFAULT   = transforms.Compose([
    transforms.CenterCrop(32),   # 96×96 → 32×32 centre crop
    transforms.ToTensor(),
])


# ── Noise filtering ────────────────────────────────────────────────

def filter_noisy_images(data_dir: str, labels_path: str,
                         output_path: str = 'clean_train_labels.csv') -> pd.DataFrame:
    """
    Scan all images in data_dir for corrupt/blank patches and save a
    cleaned label file. Identified and removed 29 noisy images from the
    original 220,025.

    Args:
        data_dir:     Path to directory containing .tif image patches.
        labels_path:  Path to original train_labels.csv.
        output_path:  Where to save clean_train_labels.csv.

    Returns:
        DataFrame with columns ['id', 'label'] for clean images only.
    """
    labels_df = pd.read_csv(labels_path)
    valid_ids, noisy_count = [], 0

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df),
                       desc='Scanning for noisy images'):
        img_path = os.path.join(data_dir, f"{row['id']}.tif")
        try:
            with Image.open(img_path) as img:
                img.verify()   # raises if corrupt
            valid_ids.append(row['id'])
        except Exception:
            noisy_count += 1

    clean_df = labels_df[labels_df['id'].isin(valid_ids)].reset_index(drop=True)
    clean_df.to_csv(output_path, index=False)
    print(f"Noise filtering: {len(labels_df):,} raw → {len(clean_df):,} clean "
          f"({noisy_count} noisy removed)")
    return clean_df


# ── Subset creation ────────────────────────────────────────────────

def create_subsets(labels_path: str, output_dir: str = '.') -> dict:
    """
    Create four stratified subsets (20/40/60/100%) from the clean label file.
    Each subset maintains the original ~60/40 non-cancerous/cancerous ratio.

    Args:
        labels_path: Path to clean_train_labels.csv.
        output_dir:  Directory to save subset CSV files.

    Returns:
        dict mapping subset number (1–4) to the subset DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    labels_df = pd.read_csv(labels_path)
    total = len(labels_df)
    subsets = {}

    for i, pct in enumerate(SUBSET_PERCENTAGES, start=1):
        n = int(total * pct)
        subset_df, _ = train_test_split(
            labels_df, train_size=n, stratify=labels_df['label'], random_state=42
        )
        subset_df = subset_df.reset_index(drop=True)
        subset_df.to_csv(os.path.join(output_dir, f'subset_{i}.csv'), index=False)
        subsets[i] = subset_df
        nc = (subset_df['label'] == 0).sum()
        c  = (subset_df['label'] == 1).sum()
        print(f"Subset {i} ({int(pct*100)}%): {n:,} samples — NC: {nc:,} | C: {c:,}")

    return subsets


def split_subset(subset_df: pd.DataFrame, subset_name: str,
                  output_dir: str = '.') -> tuple:
    """
    Split a subset DataFrame into train / val / test (70 / 15 / 15).
    Saves three CSV files: train_N.csv, val_N.csv, test_N.csv.

    Returns:
        (train_df, val_df, test_df)
    """
    train_df, temp_df = train_test_split(
        subset_df, test_size=0.30, stratify=subset_df['label'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42
    )
    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        df.reset_index(drop=True).to_csv(
            os.path.join(output_dir, f'{split_name}_{subset_name}.csv'), index=False
        )
    print(f"  Subset {subset_name}: train={len(train_df):,} | "
          f"val={len(val_df):,} | test={len(test_df):,}")
    return train_df, val_df, test_df


# ── PyTorch Dataset ────────────────────────────────────────────────

class PCamDataset(Dataset):
    """
    PyTorch Dataset for PatchCamelyon image patches.

    Args:
        csv_file:  Path to CSV with columns ['id', 'label'].
        data_dir:  Directory containing .tif image patches.
        transform: torchvision transforms to apply. Defaults to CenterCrop(32) + ToTensor().
    """

    def __init__(self, csv_file: str, data_dir: str, transform=None):
        self.data     = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform if transform else TRANSFORM_DEFAULT

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row      = self.data.iloc[idx]
        img_path = os.path.join(self.data_dir, f"{row['id']}.tif")
        image    = Image.open(img_path).convert('RGB')
        label    = torch.tensor(float(row['label']), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label


def build_loaders(train_csv: str, val_csv: str, test_csv: str,
                  data_dir: str, batch_size: int = 64,
                  num_workers: int = 2) -> tuple:
    """
    Build DataLoaders for a given train/val/test CSV split.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    kw = dict(data_dir=data_dir, transform=TRANSFORM_DEFAULT)
    train_ds = PCamDataset(train_csv, **kw)
    val_ds   = PCamDataset(val_csv,   **kw)
    test_ds  = PCamDataset(test_csv,  **kw)

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **loader_kw),
        DataLoader(val_ds,   shuffle=False, **loader_kw),
        DataLoader(test_ds,  shuffle=False, **loader_kw),
    )


if __name__ == "__main__":
    print("PCam Dataset — subset statistics")
    print(f"{'Subset':<10} {'%':<6} {'Total':<10} {'Train':<10} {'Val':<8} {'Test':<8}")
    splits = [(1,20,43999,37399,3300,3300),(2,40,87998,74798,6600,6600),
              (3,60,131997,112197,9900,9900),(4,100,219996,186796,16600,16600)]
    for s,pct,tot,tr,v,te in splits:
        print(f"S{s:<9} {pct:<6} {tot:<10,} {tr:<10,} {v:<8,} {te:<8,}")
