import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

# Robust path setup to ensure imports and files resolve from any working directory
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

# ---------- 1. Load Test Features & Normalization ----------
checkpoint_dir = os.path.join(current_dir, "checkpoints")
feature_mean = np.load(os.path.join(checkpoint_dir, "feature_mean.npy"))
feature_std = np.load(os.path.join(checkpoint_dir, "feature_std.npy"))
num_features = len(feature_mean)

csv_path = os.path.join(project_root, "data", "ratios", "test_features.csv")
if not os.path.exists(csv_path):
    csv_path = "../data/ratios/test_features.csv"

df = pd.read_csv(csv_path)

if FEATURE_NAMES and all(col in df.columns for col in FEATURE_NAMES) and len(FEATURE_NAMES) == num_features:
    feature_columns = FEATURE_NAMES
else:
    feature_columns = [col for col in df.columns if col != "label"][:num_features]

X = df[feature_columns].values.astype(np.float32)
y_true = df["label"].map(LABEL_TO_INDEX).values.astype(np.int64)

# ---------- 2. Apply Saved Training Normalization ----------
X = (X - feature_mean) / feature_std

# ---------- 3. Load Trained Model ----------
model = FaceShapeMLP(input_size=num_features, hidden_size=32, num_classes=5)
weights_path = os.path.join(checkpoint_dir, "face_shape_mlp.pth")
model.load_state_dict(torch.load(weights_path, weights_only=True))
model.eval()

# ---------- 4. Inference & Evaluation ----------
with torch.no_grad():
    predictions = model(torch.tensor(X))
    predicted_classes = torch.argmax(predictions, dim=1).cpu().numpy()

accuracy = (predicted_classes == y_true).mean()
print(f"Test accuracy: {accuracy:.2%}\n")

print("Confusion matrix (rows = true label, columns = predicted label):")
print(confusion_matrix(y_true, predicted_classes))

print("\nDetailed per-class report:")
print(classification_report(y_true, predicted_classes, target_names=CLASS_NAMES, digits=4))