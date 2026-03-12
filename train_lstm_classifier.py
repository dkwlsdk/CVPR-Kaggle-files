"""
④ train_lstm_classifier.py
SUPERVISED_TENSORS/X_train.pt, y_train.pt, t_train.pt를 사용하여
LSTM 기반 사고 분류기를 학습한다.

모델 구조:
  LSTM Encoder (hidden=128, layers=2)
  ├─ Binary head  → BCEWithLogitsLoss  (사고 여부 0/1)
  └─ Multi head   → CrossEntropyLoss   (6클래스: 정상+5종 사고 유형)
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 경로 설정 ───────────────────────────────────────────────
TENSOR_DIR  = r"C:\Users\echin\Desktop\CVPR Kaggle\SUPERVISED_TENSORS"
MODEL_DIR   = r"C:\Users\echin\Desktop\CVPR Kaggle\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 하이퍼파라미터 ──────────────────────────────────────────
HIDDEN_DIM   = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
EPOCHS       = 20
BATCH_SIZE   = 256
LR           = 1e-3
VAL_RATIO    = 0.2
NUM_CLASSES  = 6   # 0=정상, 1=head-on, 2=rear-end, 3=sideswipe, 4=t-bone, 5=single


# ── 모델 정의 ───────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """
    Binary head  : 사고 여부 예측 (Sigmoid, BCE)
    Multi head   : 사고 유형 예측 (Softmax, CrossEntropy)
    """
    def __init__(self, n_features: int, hidden_dim: int, num_layers: int,
                 num_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(hidden_dim, 1)          # 사고 여부
        self.multi_head  = nn.Linear(hidden_dim, num_classes) # 사고 유형

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, (hidden, _) = self.lstm(x)
        feat = self.dropout(hidden[-1])          # (batch, hidden_dim)
        binary_logit = self.binary_head(feat)    # (batch, 1)
        multi_logit  = self.multi_head(feat)     # (batch, num_classes)
        return binary_logit.squeeze(1), multi_logit


def train():
    print("텐서 로드 중...")
    X = torch.load(os.path.join(TENSOR_DIR, "X_train.pt")).float()
    y = torch.load(os.path.join(TENSOR_DIR, "y_train.pt")).float()
    t = torch.load(os.path.join(TENSOR_DIR, "t_train.pt")).long()
    print(f"X: {X.shape}, y: {y.shape}, t: {t.shape}")

    # ── 클래스 불균형 가중치 계산 ──
    pos_count = y.sum().item()
    neg_count = len(y) - pos_count
    pos_weight = torch.tensor([neg_count / (pos_count + 1e-6)])
    print(f"정상: {int(neg_count)}, 사고: {int(pos_count)}, pos_weight: {pos_weight.item():.2f}")

    # ── Train / Val 분할 ──
    dataset = TensorDataset(X, y, t)
    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 디바이스: {device}")

    n_features = X.shape[2]
    model = LSTMClassifier(n_features, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(device)

    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    ce_criterion  = nn.CrossEntropyLoss()
    optimizer     = optim.Adam(model.parameters(), lr=LR)
    scheduler     = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── 학습 ──
        model.train()
        total_loss = 0.0
        for bx, by, bt in train_loader:
            bx, by, bt = bx.to(device), by.to(device), bt.to(device)
            optimizer.zero_grad()
            bin_logit, multi_logit = model(bx)
            loss_bin   = bce_criterion(bin_logit, by)
            loss_multi = ce_criterion(multi_logit, bt)
            loss = loss_bin + 0.5 * loss_multi   # 두 손실 합산 (비율 조정 가능)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ── 검증 ──
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by, bt in val_loader:
                bx, by, bt = bx.to(device), by.to(device), bt.to(device)
                bin_logit, multi_logit = model(bx)
                val_loss += (bce_criterion(bin_logit, by) +
                             0.5 * ce_criterion(multi_logit, bt)).item()
                preds = (torch.sigmoid(bin_logit) > 0.5).float()
                correct += (preds == by).sum().item()
                total   += len(by)

        avg_val_loss = val_loss / len(val_loader)
        val_acc      = correct / total * 100

        scheduler.step(avg_val_loss)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:2d}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.1f}%")

        # ── 베스트 모델 저장 ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "lstm_sim_classifier.pth"))
            print(f"  ✓ 베스트 모델 저장 (val_loss={best_val_loss:.4f})")

    # ── Loss / Accuracy 그래프 ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["val_acc"], color="green", label="Val Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "sim_classifier_training.png"))
    print(f"\n학습 완료. 모델 및 그래프 저장: {MODEL_DIR}")


if __name__ == "__main__":
    train()
