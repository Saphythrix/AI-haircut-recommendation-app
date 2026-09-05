import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Robust path setup to ensure imports and file loading work from any working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

for path in [project_root, current_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from stage2_face_shape.compute_features import FEATURE_NAMES
except ImportError:
    try:
        from compute_features import FEATURE_NAMES
    except ImportError:
        FEATURE_NAMES = None

try:
    from stage2_face_shape.model import FaceShapeMLP
except ImportError:
    from model import FaceShapeMLP


CLASS_NAMES = ["Heart", "Oblong", "Oval", "Round", "Square"]
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# ---------- 1. Load the CSV built in Part F ----------
csv_path = os.path.join(project_root, "data", "ratios", "train_features.csv")
if not os.path.exists(csv_path):
    csv_path = "../data/ratios/train_features.csv"

df = pd.read_csv(csv_path)

# Automatically detect feature columns (excluding 'label')
if FEATURE_NAMES and all(col in df.columns for col in FEATURE_NAMES):
    feature_columns = FEATURE_NAMES
else:
    feature_columns = [col for col in df.columns if col != "label"]

print(f"Loaded dataset with {len(feature_columns)} features: {feature_columns}")

X = df[feature_columns].values.astype(np.float32)
y = df["label"].map(LABEL_TO_INDEX).values.astype(np.int64)

# ---------- 2. Normalize features ----------
feature_mean = X.mean(axis=0)
feature_std = X.std(axis=0)
X = (X - feature_mean) / feature_std

# Save normalization constants for inference
checkpoint_dir = os.path.join(current_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

np.save(os.path.join(checkpoint_dir, "feature_mean.npy"), feature_mean)
np.save(os.path.join(checkpoint_dir, "feature_std.npy"), feature_std)

# ---------- 3. Train/validation split ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_val_tensor = torch.tensor(X_val)
y_val_tensor = torch.tensor(y_val)

# ---------- 4. Build model, loss, optimizer, scheduler ----------
model = FaceShapeMLP(input_size=len(feature_columns), hidden_size=32, num_classes=5)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Reduce LR if val_loss plateaus for 15 consecutive epochs
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=15,
    min_lr=1e-5,
)

# ---------- 5. Training loop with metric tracking ----------
NUM_EPOCHS = 150

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "lr": []
}

print(f"Starting training for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    model.train()

    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = loss_function(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Track training accuracy
    train_pred_classes = torch.argmax(predictions, dim=1)
    train_acc = (train_pred_classes == y_train_tensor).float().mean().item()

    # Evaluation on validation set
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_tensor)
        val_loss = loss_function(val_predictions, y_val_tensor)
        val_pred_classes = torch.argmax(val_predictions, dim=1)
        val_acc = (val_pred_classes == y_val_tensor).float().mean().item()

    # Step the learning rate scheduler based on validation loss
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    # Record metrics for every epoch
    history["epoch"].append(epoch + 1)
    history["train_loss"].append(loss.item())
    history["val_loss"].append(val_loss.item())
    history["train_acc"].append(train_acc * 100)
    history["val_acc"].append(val_acc * 100)
    history["lr"].append(current_lr)

    # Print status every 10 epochs and on final epoch
    if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | "
              f"LR: {current_lr:.6f}")

# ---------- 6. Save the trained model ----------
model_save_path = os.path.join(checkpoint_dir, "face_shape_mlp.pth")
torch.save(model.state_dict(), model_save_path)
print(f"\n[✓] Model saved to {model_save_path}")

# ---------- 7. Final Evaluation & Classification Report ----------
model.eval()
with torch.no_grad():
    final_val_preds = model(X_val_tensor)
    y_pred = torch.argmax(final_val_preds, dim=1).cpu().numpy()
    y_true = y_val_tensor.cpu().numpy()

print("\n" + "=" * 55)
print("              CLASSIFICATION REPORT")
print("=" * 55)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ---------- 8. Matplotlib Visualizations ----------
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle("FaceShapeMLP Training Performance & Validation Metrics", fontsize=16, fontweight="bold")

# Plot 1: Loss Curve
axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
axes[0].set_title("Loss Curves (Cross-Entropy)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Epoch", fontsize=11)
axes[0].set_ylabel("Loss", fontsize=11)
axes[0].legend(loc="upper right", frameon=True)
axes[0].grid(True, linestyle=":", alpha=0.6)

# Plot 2: Accuracy Curve
axes[1].plot(history["epoch"], history["train_acc"], label="Train Accuracy", color="#2ca02c", linewidth=2)
axes[1].plot(history["epoch"], history["val_acc"], label="Val Accuracy", color="#d62728", linewidth=2, linestyle="--")
axes[1].set_title("Accuracy Curves (%)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylabel("Accuracy (%)", fontsize=11)
axes[1].legend(loc="lower right", frameon=True)
axes[1].grid(True, linestyle=":", alpha=0.6)

# Plot 3: Learning Rate Schedule Curve
axes[2].plot(history["epoch"], history["lr"], label="Learning Rate", color="#9467bd", linewidth=2)
axes[2].set_title("Learning Rate Decay", fontsize=13, fontweight="bold")
axes[2].set_xlabel("Epoch", fontsize=11)
axes[2].set_ylabel("Learning Rate", fontsize=11)
axes[2].set_yscale("log")
axes[2].legend(loc="upper right", frameon=True)
axes[2].grid(True, linestyle=":", alpha=0.6)

# Plot 4: Confusion Matrix Heatmap
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[3], cbar=False)
axes[3].set_title("Validation Confusion Matrix", fontsize=13, fontweight="bold")
axes[3].set_xlabel("Predicted Class", fontsize=11)
axes[3].set_ylabel("Ground Truth Class", fontsize=11)

plt.tight_layout()

# Save plot artifact
plot_save_path = os.path.join(checkpoint_dir, "training_metrics.png")
plt.savefig(plot_save_path, dpi=300)
print(f"[✓] Metrics plot saved to {plot_save_path}")

# Show interactive figure window
plt.show()