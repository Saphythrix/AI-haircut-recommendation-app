import os
import cv2
def capture_from_webcam(save_path=None):
    """
    Opens webcam feed.
    Press 's' to capture and save the photo.
    Press 'q' to quit without saving.
    """
    if save_path is None:
        base_dir = os.path.dirname(__file__)
        output_dir = os.path.abspath(os.path.join(base_dir, "..", "data", "raw"))
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "webcam_test.jpg")
    else:
        # Ensure target directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return None
    print("Webcam active: Press 's' to save a photo, or 'q' to quit.")
    saved_file = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break
        cv2.imshow("Webcam - press 's' to save, 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite(save_path, frame)
            print(f"Photo successfully saved to: {save_path}")
            saved_file = save_path
            break
        elif key == ord('q'):
            print("Capture cancelled by user.")
            break
    # Clean up resources (outside the while loop)
    cap.release()
    cv2.destroyAllWindows()
    return saved_file
if __name__ == "__main__":
    capture_from_webcam()