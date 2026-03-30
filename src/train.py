"""
train.py
--------
Training loop for the custom CNN on PatchCamelyon subsets.

Usage:
    python train.py --subset 4 \
                    --data_dir /kaggle/input/histopathologic-cancer-detection/train \
                    --label_dir /kaggle/working \
                    --epochs 30 \
                    --save_path weights_s4.pth

Canonical hyperparameters:
    Loss:      BCEWithLogitsLoss
    Optimizer: Adam (lr=0.001, β₁=0.9, β₂=0.999, ε=1e-7)
    Epochs:    30 | Batch: 64
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

from model import CustomCNN
from dataset import build_loaders


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []
    for images, labels in tqdm(loader, desc='  Train', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    return total_loss / len(loader), accuracy_score(all_labels, preds)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='   Val', leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    auc   = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader), accuracy_score(all_labels, preds), auc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Subset: {args.subset}")

    train_loader, val_loader, _ = build_loaders(
        f"{args.label_dir}/train_{args.subset}.csv",
        f"{args.label_dir}/val_{args.subset}.csv",
        f"{args.label_dir}/test_{args.subset}.csv",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    print(f"Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,}")

    model     = CustomCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.999), eps=1e-7)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc           = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_auc   = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch [{epoch:02d}/{args.epochs}] "
              f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val   loss={va_loss:.4f} acc={va_acc:.4f} AUC={va_auc:.4f}")
        if va_auc > best_auc:
            best_auc = va_auc
            if args.save_path:
                torch.save(model.state_dict(), args.save_path)
                print(f"  ✓ Saved best model (AUC={best_auc:.4f}) → {args.save_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset',     type=int,   default=4,   choices=[1,2,3,4])
    parser.add_argument('--data_dir',   required=True)
    parser.add_argument('--label_dir',  default='/kaggle/working')
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--save_path',  default=None)
    train(parser.parse_args())
