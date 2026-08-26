import cv2
import os

try:
    from stage1_landmarks.detect_landmarks import detect_landmarks
except ImportError:
    from detect_landmarks import detect_landmarks
    
def draw_landmarks(image,landmark_list):
    height,width,_=image.shape
    image_with_landmarks=image.copy()

    for lm in landmark_list:
        x_pix=int(lm[0]*width)
        y_pix=int(lm[1]*height)
        cv2.circle(image_with_landmarks,(x_pix,y_pix),2,(0,255,0),-1)

    return image_with_landmarks

if __name__ == "__main__":
    # 1. Resolve path to your test image
    base_dir = os.path.dirname(__file__)
    image_path = os.path.join(base_dir, "test_images", "DSC_2112[1] copy.jpg.jpeg")
    
    # 2. Define output save path
    output_dir = os.path.abspath(os.path.join(base_dir, "..", "data", "landmarks"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_output.jpg")
    print(f"Running detection on: {image_path}")
    landmark_list, ori_img = detect_landmarks(image_path)
    # 3. If face detected, draw points and save
    if landmark_list is not None:
        print(f"Found {len(landmark_list)} landmarks. Drawing landmarks...")
        annotated_image = draw_landmarks(ori_img, landmark_list)
        cv2.imwrite(output_path, annotated_image)
        print(f"Saved annotated image to: {output_path}")
    else:
        print("No face detected in the image.")

