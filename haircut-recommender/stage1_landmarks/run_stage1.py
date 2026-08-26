import os
import cv2
try:
    from stage1_landmarks.detect_landmarks import detect_landmarks
    from stage1_landmarks.visualize_landmarks import draw_landmarks
    from stage1_landmarks.webcam_capture import capture_from_webcam
except ImportError:
    from detect_landmarks import detect_landmarks
    from visualize_landmarks import draw_landmarks
    from webcam_capture import capture_from_webcam

def run_pipeline():
    base_dir=os.path.dirname(__file__)
    data_dir=os.path.abspath(os.path.join(base_dir,"..","data"))
    
    
    choice=input("Choose how do you want to input your image (Type 1/2):\n1.Upload a image\n2.Take a selfie\n")
    if choice=="1":
        default_img=os.path.join(base_dir,"test_images", "DSC_2112[1] copy.jpg.jpeg")
        user_input=input("Enter you image path or just press enter to use the default image:")
        image_path = user_input if user_input else default_img
        if not os.path.exists(image_path):
            print(f"\nError: File not found at '{image_path}'")
            return
    
    elif choice=="2":
        raw_output_dir=os.path.join(data_dir,"raw")
        os.makedirs(raw_output_dir,exist_ok=True)
        raw_save_path = os.path.join(raw_output_dir, "webcam_capture.jpg")

        print("\nOpening webcam... (Press 's' to capture photo, 'q' to cancel)")
        image_path = capture_from_webcam(raw_save_path)

        if not image_path or not os.path.exists(image_path):
            print("\nWebcam capture was cancelled or failed.")
            return
    else:
        print(f"\nInvalid choice '{choice}'. Please enter 1 or 2.")
        return

    print(f"\nRunning landmark detection on: {image_path}")
    landmark_list, original_image = detect_landmarks(image_path)
    if landmark_list is None:
        print("Detection failed: No face found in the image.")
        return
    # Step 5 & 6: Draw and Save Visualized Output
    print(f"Detected {len(landmark_list)} landmarks. Drawing overlay...")
    annotated_image = draw_landmarks(original_image, landmark_list)
    landmarks_dir = os.path.join(data_dir, "landmarks")
    os.makedirs(landmarks_dir, exist_ok=True)
    
    # Save output with a clear name
    output_filename = "stage1_output.jpg"
    output_path = os.path.join(landmarks_dir, output_filename)
    cv2.imwrite(output_path, annotated_image)
    # Step 7: Print Summary
    print("\n" + "=" * 60)
    print("                  STAGE 1 COMPLETE!")
    print("=" * 60)
    print(f"  • Landmarks Extracted : {len(landmark_list)} points")
    print(f"  • Source Image        : {image_path}")
    print(f"  • Saved Result        : {output_path}")
    print("=" * 60)
if __name__ == "__main__":
    run_pipeline() 
        