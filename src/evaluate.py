"""
evaluate.py
-----------
Full evaluation pipeline: ROC-AUC, PR-AUC, confusion matrix,
sensitivity, specificity, and visualisations.

Canonical results (Custom CNN):
    Subset 1 (20%): ROC-AUC=0.8108 | PR-AUC=0.6941 | CM=[[2281,359],[160,500]]
    Subset 2 (40%): ROC-AUC=0.7888 | PR-AUC=0.7088 | CM=[[4914,366],[466,854]]
    Subset 3 (60%): loaded from saved weights | CM=[[7357,563],[608,1372]]
    Subset 4 (100%): ROC-AUC=0.8228 | PR-AUC=0.7029 | CM=[[11201,1998],[670,2629]]

Usage:
    python evaluate.py --weights weights_s4.pth \
                       --test_csv /kaggle/working/test_4.csv \
                       --data_dir /kaggle/input/histopathologic-cancer-detection/train \
                       --output_dir results/
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve,
                             confusion_matrix, accuracy_score)
from tqdm import tqdm

from model import CustomCNN
from dataset import PCamDataset
from torch.utils.data import DataLoader


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating', leave=False):
            logits = model(images.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= threshold).astype(int)
    return probs, labels, preds


def print_metrics(probs, labels, preds):
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    print("\n" + "="*50)
    print("  EVALUATION RESULTS")
    print("="*50)
    print(f"  Accuracy    : {accuracy_score(labels, preds):.4f}")
    print(f"  ROC-AUC     : {roc_auc_score(labels, probs):.4f}")
    print(f"  PR-AUC      : {average_precision_score(labels, probs):.4f}")
    print(f"  Sensitivity : {tp/(tp+fn):.4f}  (Recall)")
    print(f"  Specificity : {tn/(tn+fp):.4f}")
    print(f"  Precision   : {tp/(tp+fp):.4f}")
    print(f"  F1-Score    : {2*tp/(2*tp+fp+fn):.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TP={tp:,}")
    print("="*50)


def plot_roc_curve(probs, labels, save_path=None):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curve — Custom CNN on PCam')
    ax.legend(loc='lower right')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr_curve(probs, labels, save_path=None):
    prec, rec, _ = precision_recall_curve(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color='darkorange', lw=2, label=f'PR (AUC = {pr_auc:.4f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve — Custom CNN on PCam')
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(preds, labels, save_path=None):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Pred NC','Pred C'],
                yticklabels=['Actual NC','Actual C'])
    tn,fp,fn,tp = cm.ravel()
    ax.set_title(f'Confusion Matrix\nSens={tp/(tp+fn):.3f}  Spec={tn/(tn+fp):.3f}',
                 fontweight='bold')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model = CustomCNN().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    test_ds     = PCamDataset(args.test_csv, args.data_dir)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=2, pin_memory=True)
    print(f"Test set: {len(test_ds):,} samples")

    probs, labels, preds = evaluate(model, test_loader, device)
    print_metrics(probs, labels, preds)

    plot_roc_curve(probs, labels,
                   save_path=os.path.join(args.output_dir, 'roc_curve.png'))
    plot_pr_curve(probs, labels,
                  save_path=os.path.join(args.output_dir, 'pr_curve.png'))
    plot_confusion_matrix(preds, labels,
                          save_path=os.path.join(args.output_dir, 'confusion_matrix.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',    required=True)
    parser.add_argument('--test_csv',   required=True)
    parser.add_argument('--data_dir',   required=True)
    parser.add_argument('--output_dir', default='results/')
    main(parser.parse_args())
