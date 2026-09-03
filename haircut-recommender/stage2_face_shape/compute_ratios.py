import os
import sys
import math

# ---------------------------------------------------------------------------
# Robust path setup to ensure imports work from any working directory
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

for path in [project_root, current_dir, os.path.join(project_root, "stage1_landmarks")]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from stage1_landmarks.detect_landmarks import detect_landmarks
except ImportError:
    # pyrefly: ignore [missing-import]
    from detect_landmarks import detect_landmarks

# ---------------------------------------------------------------------------
# Key Landmark Indices (from Stage 1 Part G)
# ---------------------------------------------------------------------------
FOREHEAD_TOP = 10
CHIN = 152
CHEEK_LEFT = 227
CHEEK_RIGHT = 447
JAW_LEFT = 234
JAW_RIGHT = 454
FOREHEAD_LEFT = 71
FOREHEAD_RIGHT = 301


def euclidean_distance(point1, point2):
    """
    Computes the 2D Euclidean distance between two landmark points [x, y, ...].
    Distance formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)
    """
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    return math.sqrt(dx * dx + dy * dy)


def compute_ratios(landmarks):
    """
    Calculates facial measurements and returns the 4 key geometric ratios as a list of floats:
    1. length_to_width_ratio       = face_length / cheekbone_width
    2. jaw_to_cheekbone_ratio      = jaw_width / cheekbone_width
    3. forehead_to_cheekbone_ratio = forehead_width / cheekbone_width
    4. jaw_to_forehead_ratio       = jaw_width / forehead_width

    Args:
        landmarks (list): 468 MediaPipe landmark coordinates [[x, y, z], ...]
    
    Returns:
        list[float] or None: [length_to_width, jaw_to_cheekbone, forehead_to_cheekbone, jaw_to_forehead]
    """
    if landmarks is None or len(landmarks) < 468:
        return None

    # 1. Compute Key Facial Distances
    face_length = euclidean_distance(landmarks[FOREHEAD_TOP], landmarks[CHIN])
    cheekbone_width = euclidean_distance(landmarks[CHEEK_LEFT], landmarks[CHEEK_RIGHT])
    jaw_width = euclidean_distance(landmarks[JAW_LEFT], landmarks[JAW_RIGHT])
    forehead_width = euclidean_distance(landmarks[FOREHEAD_LEFT], landmarks[FOREHEAD_RIGHT])

    # Avoid division by zero
    if min(face_length, cheekbone_width, jaw_width, forehead_width) <= 0:
        return None

    # 2. Compute the 4 Ratios
    ratios = [
        face_length / cheekbone_width,        # length_to_width_ratio
        jaw_width / cheekbone_width,          # jaw_to_cheekbone_ratio
        forehead_width / cheekbone_width,      # forehead_to_cheekbone_ratio
        jaw_width / forehead_width            # jaw_to_forehead_ratio
    ]

    return ratios


# ---------------------------------------------------------------------------
# E.3 Test Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    RATIO_NAMES = [
        "length_to_width_ratio",
        "jaw_to_cheekbone_ratio",
        "forehead_to_cheekbone_ratio",
        "jaw_to_forehead_ratio"
    ]

    test_image_path = os.path.join(
        project_root, "stage1_landmarks", "test_images", "DSC_2112[1] copy.jpg.jpeg"
    )

    if not os.path.exists(test_image_path):
        test_image_path = os.path.join(
            project_root, "stage1_landmarks", "test_images", "Myphoto.jpeg"
        )
    if not os.path.exists(test_image_path):
        test_image_path = os.path.join(
            project_root, "data", "raw", "webcam_capture.jpg"
        )

    print(f"\n[1/3] Loading image from: {test_image_path}")
    landmarks, _ = detect_landmarks(test_image_path)

    if landmarks is None:
        print("[-] Error: No face detected in the test image.")
    else:
        print(f"[2/3] Successfully extracted {len(landmarks)} landmarks.")
        ratios = compute_ratios(landmarks)

        print("\n[3/3] ================= Computed Facial Ratios =================")
        for name, value in zip(RATIO_NAMES, ratios):
            print(f"  • {name:<28}: {value:.4f}")
        print("=" * 60)

        # Checkpoint sanity check
        all_in_range = all(0.5 <= val <= 1.8 for val in ratios)
        if all_in_range:
            print("\n[✓] CHECKPOINT PASSED: All ratios fall in the expected ~0.5 - 1.5 range.")
        else:
            print("\n[!] WARNING: Some ratios are outside the expected range. Check landmark indices.")
