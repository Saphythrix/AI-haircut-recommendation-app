import os
import cv2
import mediapipe as mp

# Suppress internal C++ logging warnings
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

mp_face_mesh = mp.solutions.face_mesh
_DEFAULT_DETECTOR = None


def create_face_mesh_detector():
    return mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )


def get_default_detector():
    """Returns a shared detector instance so we don't re-initialize on every image."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = create_face_mesh_detector()
    return _DEFAULT_DETECTOR


def load_image(image_path):
    ori_img = cv2.imread(image_path)
    if ori_img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img_rgb = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    return ori_img, img_rgb


def detect_landmarks(image_path, detector=None, verbose=False):
    """
    Detects 468 facial landmarks on the given image.
    Re-uses a single detector instance for maximum performance.
    """
    if detector is None:
        detector = get_default_detector()

    ori_image, rgb_image = load_image(image_path)
    result = detector.process(rgb_image)

    if not result.multi_face_landmarks:
        if verbose:
            print("NO FACE DETECTED")
        return None, None

    face_landmark = result.multi_face_landmarks[0]
    landmark_list = [[lm.x, lm.y, lm.z] for lm in face_landmark.landmark]

    return landmark_list, ori_image


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    image_path = os.path.join(base_dir, "test_images", "DSC_2112[1] copy.jpg.jpeg")
    
    landmarks, ori_img = detect_landmarks(image_path, verbose=True)
    if landmarks:
        print(f"Successfully detected {len(landmarks)} landmarks.")
        print("First 3 landmarks:", landmarks[:3])