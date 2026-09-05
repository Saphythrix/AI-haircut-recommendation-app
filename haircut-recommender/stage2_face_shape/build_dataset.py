import os
import csv
import sys

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
    from detect_landmarks import detect_landmarks

try:
    from stage2_face_shape.compute_features import compute_features, FEATURE_NAMES
except ImportError:
    from compute_features import compute_features, FEATURE_NAMES

CLASS_NAMES = ["Heart", "Oblong", "Oval", "Round", "Square"]


def build_feature_csv(dataset_folder, output_csv_path):
    """
    dataset_folder: path to either the training_set or testing_set folder,
                    which contains one subfolder per class name.
    output_csv_path: where to write the resulting CSV.
    """
    rows_written = 0
    skipped = 0

    if not os.path.exists(dataset_folder):
        print(f"Error: Dataset folder not found: {dataset_folder}")
        return

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # Header row — now 10 feature columns instead of 4
        writer.writerow(FEATURE_NAMES + ["label"])

        for class_name in CLASS_NAMES:
            class_folder = os.path.join(dataset_folder, class_name)
            if not os.path.isdir(class_folder):
                print(f"Warning: folder not found for class '{class_name}', skipping.")
                continue

            image_files = [
                f for f in os.listdir(class_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            total_class_images = len(image_files)
            print(f"\nProcessing {total_class_images} images for class '{class_name}'...")

            class_written = 0
            class_skipped = 0

            for idx, filename in enumerate(image_files, 1):
                image_path = os.path.join(class_folder, filename)

                landmark_list, _ = detect_landmarks(image_path)
                if landmark_list is None:
                    # No face detected in this image — skip it rather than crash
                    skipped += 1
                    class_skipped += 1
                    continue

                features = compute_features(landmark_list)
                if features is None:
                    skipped += 1
                    class_skipped += 1
                    continue

                writer.writerow(features + [class_name])
                rows_written += 1
                class_written += 1

                # Live progress update every 50 images or at completion
                if idx % 50 == 0 or idx == total_class_images:
                    pct = (idx / total_class_images) * 100
                    print(
                        f"  -> [{class_name}] {idx}/{total_class_images} ({pct:.1f}%) | "
                        f"Extracted: {class_written}, Skipped: {class_skipped}",
                        end="\r" if idx != total_class_images else "\n"
                    )

    print(f"\nCompleted '{os.path.basename(output_csv_path)}': Wrote {rows_written} total rows (Skipped {skipped} with no clear face).")


if __name__ == "__main__":
    # Adjust paths to match dataset location in data/FaceShape Dataset
    training_folder = os.path.join(project_root, "data", "FaceShape Dataset", "training_set")
    testing_folder = os.path.join(project_root, "data", "FaceShape Dataset", "testing_set")

    ratios_dir = os.path.join(project_root, "data", "ratios")
    os.makedirs(ratios_dir, exist_ok=True)

    train_csv_path = os.path.join(ratios_dir, "train_features.csv")
    test_csv_path = os.path.join(ratios_dir, "test_features.csv")

    print("=" * 65)
    print("STEP 1/2: Processing Training Set...")
    print("=" * 65)
    build_feature_csv(training_folder, train_csv_path)

    print("\n" + "=" * 65)
    print("STEP 2/2: Processing Testing Set...")
    print("=" * 65)
    build_feature_csv(testing_folder, test_csv_path)