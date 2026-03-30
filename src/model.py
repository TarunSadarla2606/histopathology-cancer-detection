"""
model.py
--------
Custom 5-layer CNN for binary cancer detection on PatchCamelyon patches.

Architecture:
    Input: (B, 3, 32, 32)   [after CenterCrop(32) from 96×96 patches]

    Block 1:  Conv(3→32,   k=3, p=1) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.25)
    Block 2:  Conv(32→64,  k=3, p=1) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.25)
    Block 3:  Conv(64→128, k=3, p=1) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.25)
    Block 4:  Conv(128→256,k=3, p=1) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.25)
    Block 5:  Conv(256→512,k=3, p=1) + BatchNorm + ReLU + AdaptiveAvgPool(1) + Dropout(0.25)

    Head:     Flatten → Linear(512, 256) + ReLU → Linear(256, 1)
              Output: raw logit (use BCEWithLogitsLoss during training)

Training configuration:
    Loss:      BCEWithLogitsLoss
    Optimizer: Adam (lr=0.001, β₁=0.9, β₂=0.999, ε=1e-7)
    Epochs:    30
    Batch:     64

Results (full dataset, Subset 4):
    ROC-AUC = 0.8228 | PR-AUC = 0.7029
    Sensitivity = 0.797 | Specificity = 0.849
    Confusion matrix: [[11201, 1998], [670, 2629]]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    """
    Custom 5-layer CNN for binary cancer patch classification.

    Designed for 32×32 centre-cropped patches from the PatchCamelyon dataset.
    Uses BatchNorm after each conv layer for training stability, and Dropout2d
    for regularisation.

    Output is a raw logit (scalar per sample). Use BCEWithLogitsLoss for training
    and sigmoid for inference.
    """

    def __init__(self, dropout: float = 0.25):
        super().__init__()

        def conv_block(in_ch, out_ch, drop=dropout):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(drop),
            )

        self.block1 = conv_block(3,   32)    # 32→16
        self.block2 = conv_block(32,  64)    # 16→8
        self.block3 = conv_block(64,  128)   # 8→4
        self.block4 = conv_block(128, 256)   # 4→2
        # Block 5 uses AdaptiveAvgPool instead of MaxPool to handle variable input
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),         # → (B, 512, 1, 1)
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),               # raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return self.classifier(x).squeeze(1)  # (B,)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return cancer probability (sigmoid of logit)."""
        return torch.sigmoid(self.forward(x))


class VGG16Cancer(nn.Module):
    """
    VGG16 fine-tuned for binary cancer patch classification.

    Modified for 32×32 input:
        - First conv layer replaced: Conv(3, 64, k=3, p=1) instead of k=3, p=1
        - First 10 layers frozen (feature extraction mode)
        - Classifier head replaced with binary output

    Note: This was an exploratory approach — not completed before the submission
    deadline. The architecture is included here for reference.
    """

    def __init__(self, freeze_layers: int = 10):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights='IMAGENET1K_V1')

        # Adapt first layer for 32×32 input
        vgg.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # Freeze early layers
        for i, param in enumerate(vgg.features.parameters()):
            if i < freeze_layers:
                param.requires_grad = False

        self.features    = vgg.features
        self.avgpool     = vgg.avgpool
        self.classifier  = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x).squeeze(1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(4, 3, 32, 32).to(device)
    model = CustomCNN().to(device)
    out   = model(dummy)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CustomCNN | output: {out.shape} | params: {n_params:,}")
