import cv2
import os
try:
    from stage1_landmarks.detect_landmarks import detect_landmarks
except ImportError:
    from detect_landmarks import detect_landmarks

KEY_LANDMARKS = {
    152: "Chin",
    234: "L Jaw",
    454: "R Jaw",
    227: "L Cheek",
    447: "R Cheek",
    10:  "Forehead",
    71:  "L Forehead",
    301: "R Forehead",
    33:  "L Eye",
    263: "R Eye",
    1:   "Nose"
}

def draw_landmarks(image,landmark_list,key_landmarks=KEY_LANDMARKS):
    height, width, _ = image.shape
    annotated = image.copy()
    # 1. Draw all 468 landmarks in dim gray as background context
    for lm in landmark_list:
        px = int(lm[0] * width)
        py = int(lm[1] * height)
        cv2.circle(annotated, (px, py), 1, (160, 160, 160), -1)
    # 2. Highlight key indices in bright red with text annotations
    for idx, label in key_landmarks.items():
        if idx < len(landmark_list):
            lm = landmark_list[idx]
            px = int(lm[0] * width)
            py = int(lm[1] * height)
            # Draw highlighted dot
            cv2.circle(annotated, (px, py), 4, (0, 0, 255), -1)
            # Draw label with index number
            text = f"{idx}:{label}"
            cv2.putText(
                annotated,
                text,
                (px + 6, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )
    return annotated

if __name__ == "__main__":
    # 1. Image path
    base_dir = os.path.dirname(__file__)
    image_path = os.path.join(base_dir, "test_images", "DSC_2112[1] copy.jpg.jpeg")
    # 2. Output path
    output_dir = os.path.abspath(os.path.join(base_dir, "..", "data", "landmarks"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "verified_indices.jpg")
    print(f"Detecting landmarks for verification on: {image_path}")
    landmark_list, ori_img = detect_landmarks(image_path)
    if landmark_list is not None:
        annotated_image = draw_landmarks(ori_img, landmark_list)
        cv2.imwrite(output_path, annotated_image)
        print(f"Saved verified indices visualization to: {output_path}")
    else:
        print("No face detected.")
    