import os
import sys
import numpy as np
import torch

# Robust path setup to ensure imports and assets resolve from any working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
stage1_dir = os.path.join(project_root, "stage1_landmarks")

for path in [project_root, current_dir, stage1_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from stage1_landmarks.detect_landmarks import detect_landmarks
except ImportError:
    from detect_landmarks import detect_landmarks

try:
    from stage2_face_shape.compute_ratios import compute_ratios
except ImportError:
    from compute_ratios import compute_ratios

try:
    from stage2_face_shape.model import FaceShapeMLP
except ImportError:
    from model import FaceShapeMLP


CLASS_NAMES = ["Heart", "Oblong", "Oval", "Round", "Square"]
CHECKPOINT_DIR = os.path.join(current_dir, "checkpoints")


def predict_face_shape(image_path, verbose=True):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return None

    landmark_result = detect_landmarks(image_path)
    if landmark_result is None or landmark_result[0] is None:
        if verbose:
            print(f"No face detected in image: {image_path}")
        return None

    landmark_list, _ = landmark_result
    ratios = compute_ratios(landmark_list)
    if ratios is None:
        if verbose:
            print("Could not compute ratios for this image.")
        return None

    # Load normalization constants
    mean_file = os.path.join(CHECKPOINT_DIR, "feature_mean.npy")
    std_file = os.path.join(CHECKPOINT_DIR, "feature_std.npy")
    model_file = os.path.join(CHECKPOINT_DIR, "face_shape_mlp.pth")

    if not (os.path.exists(mean_file) and os.path.exists(std_file) and os.path.exists(model_file)):
        raise FileNotFoundError(
            f"Checkpoint files missing in '{CHECKPOINT_DIR}'. Please run train.py first."
        )

    feature_mean = np.load(mean_file)
    feature_std = np.load(std_file)

    # Normalize features and prepare tensor batch
    features = (np.array(ratios, dtype=np.float32) - feature_mean) / feature_std
    features_tensor = torch.tensor(features).unsqueeze(0)  # Shape: (1, 4)

    # Load model architecture matching trained weights
    model = FaceShapeMLP(input_size=4, hidden_size=32, num_classes=5)
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()

    with torch.no_grad():
        output = model(features_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_index].item()

    predicted_shape = CLASS_NAMES[predicted_index]
    if verbose:
        print(f"\nPredicted face shape: {predicted_shape} (confidence: {confidence:.1%})")
        print("Class Probabilities:")
        for idx, class_name in enumerate(CLASS_NAMES):
            prob = probabilities[0][idx].item()
            bar = "█" * int(prob * 30)
            print(f"  {class_name:7s}: {prob:6.2%} | {bar}")

    return {
        "face_shape": predicted_shape,
        "confidence": confidence,
        "probabilities": {
            name: probabilities[0][i].item() for i, name in enumerate(CLASS_NAMES)
        },
        "ratios": ratios,
    }


if __name__ == "__main__":
    # Default test photo from stage1_landmarks/test_images
    test_image = os.path.join(project_root, "stage1_landmarks", "test_images", "DSC_2112[1] copy.jpg.jpeg")
    if not os.path.exists(test_image):
        test_image = os.path.join(project_root, "stage1_landmarks", "test_images", "Myphoto.jpeg")

    predict_face_shape(test_image)