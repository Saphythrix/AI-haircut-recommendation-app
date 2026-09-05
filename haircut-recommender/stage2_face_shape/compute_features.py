import os
import sys
import math

# ---------------------------------------------------------------------------
# Robust path setup (same as your original)
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

for path in [project_root, current_dir, os.path.join(project_root, "stage1_landmarks")]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from stage1_landmarks.detect_landmarks import detect_landmarks
except ImportError:
    from detect_landmarks import detect_landmarks

# ---------------------------------------------------------------------------
# Key Landmark Indices
# ---------------------------------------------------------------------------
FOREHEAD_TOP = 10
CHIN = 152
CHEEK_LEFT = 227
CHEEK_RIGHT = 447
JAW_LEFT = 234
JAW_RIGHT = 454
FOREHEAD_LEFT = 71
FOREHEAD_RIGHT = 301

# New landmarks for vertical face-thirds (classic face-shape heuristic)
EYEBROW_MID = 9    # glabella, between eyebrows
NOSE_BASE = 2      # subnasale, base of nose


def euclidean_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def angle_at_vertex(a, b, c):
    """
    Angle (in degrees) at vertex b, formed by rays b->a and b->c.
    A SMALL angle means a sharp corner (e.g. square jaw, pointed chin).
    A LARGE angle means a soft/rounded corner.
    """
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return None
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


FEATURE_NAMES = [
    "length_to_width_ratio",
    "jaw_to_cheekbone_ratio",
    "forehead_to_cheekbone_ratio",
    "jaw_to_forehead_ratio",
    "jaw_angle_left",
    "jaw_angle_right",
    "chin_angle",
    "upper_face_ratio",
    "midface_ratio",
    "lower_face_ratio",
]


def compute_features(landmarks):
    """
    Returns a list of 10 features:
      [0-3]  original width-based ratios
      [4-5]  jaw corner angles (left/right) -> jaw sharpness
      [6]    chin angle -> chin pointiness
      [7-9]  vertical face-third proportions (upper/mid/lower), sum ~= 1.0
    """
    if landmarks is None or len(landmarks) < 468:
        return None

    # --- Core distances ---
    face_length = euclidean_distance(landmarks[FOREHEAD_TOP], landmarks[CHIN])
    cheekbone_width = euclidean_distance(landmarks[CHEEK_LEFT], landmarks[CHEEK_RIGHT])
    jaw_width = euclidean_distance(landmarks[JAW_LEFT], landmarks[JAW_RIGHT])
    forehead_width = euclidean_distance(landmarks[FOREHEAD_LEFT], landmarks[FOREHEAD_RIGHT])

    if min(face_length, cheekbone_width, jaw_width, forehead_width) <= 0:
        return None

    # --- Original 4 ratios ---
    length_to_width_ratio = face_length / cheekbone_width
    jaw_to_cheekbone_ratio = jaw_width / cheekbone_width
    forehead_to_cheekbone_ratio = forehead_width / cheekbone_width
    jaw_to_forehead_ratio = jaw_width / forehead_width

    # --- Angularity features ---
    # Jaw corner angle: how sharply the contour turns at the jawline.
    # Square jaws -> smaller angle. Round/soft jaws -> larger angle.
    jaw_angle_left = angle_at_vertex(landmarks[CHEEK_LEFT], landmarks[JAW_LEFT], landmarks[CHIN])
    jaw_angle_right = angle_at_vertex(landmarks[CHEEK_RIGHT], landmarks[JAW_RIGHT], landmarks[CHIN])

    # Chin angle: how pointed the chin is.
    # Heart/Oval -> smaller (pointier) angle. Square/Round -> larger (flatter) angle.
    chin_angle = angle_at_vertex(landmarks[JAW_LEFT], landmarks[CHIN], landmarks[JAW_RIGHT])

    if None in (jaw_angle_left, jaw_angle_right, chin_angle):
        return None

    # --- Vertical face-thirds ---
    # In a classic "balanced oval", these three are roughly equal.
    # A dominant lower_face_ratio suggests Square/Round; dominant upper suggests Oblong, etc.
    upper_face = euclidean_distance(landmarks[FOREHEAD_TOP], landmarks[EYEBROW_MID])
    midface = euclidean_distance(landmarks[EYEBROW_MID], landmarks[NOSE_BASE])
    lower_face = euclidean_distance(landmarks[NOSE_BASE], landmarks[CHIN])
    total = upper_face + midface + lower_face

    if total <= 0:
        return None

    upper_face_ratio = upper_face / total
    midface_ratio = midface / total
    lower_face_ratio = lower_face / total

    return [
        length_to_width_ratio,
        jaw_to_cheekbone_ratio,
        forehead_to_cheekbone_ratio,
        jaw_to_forehead_ratio,
        jaw_angle_left,
        jaw_angle_right,
        chin_angle,
        upper_face_ratio,
        midface_ratio,
        lower_face_ratio,
    ]


# Backward compatibility alias
compute_ratios = compute_features


if __name__ == "__main__":
    test_image_path = os.path.join(
        project_root, "stage1_landmarks", "test_images", "DSC_2112[1] copy.jpg.jpeg"
    )
    if not os.path.exists(test_image_path):
        test_image_path = os.path.join(project_root, "stage1_landmarks", "test_images", "Myphoto.jpeg")
    if not os.path.exists(test_image_path):
        test_image_path = os.path.join(project_root, "data", "raw", "webcam_capture.jpg")

    print(f"\n[1/3] Loading image from: {test_image_path}")
    landmarks, _ = detect_landmarks(test_image_path)

    if landmarks is None:
        print("[-] Error: No face detected in the test image.")
    else:
        print(f"[2/3] Successfully extracted {len(landmarks)} landmarks.")
        features = compute_features(landmarks)

        print("\n[3/3] ================= Computed Facial Features =================")
        for name, value in zip(FEATURE_NAMES, features):
            print(f"  • {name:<28}: {value:.4f}")
        print("=" * 65)
