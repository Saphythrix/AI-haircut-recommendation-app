# Stage 1 Manual: Face Landmark Detection
### The Haircut Recommender Project — Module 1 of 6

---

## How to Use This Manual

This is not a copy-paste tutorial. It's built so you write almost all of the actual logic yourself, while I hand you the boring boilerplate (imports, file paths, folder setup) so you can focus your energy on the parts that teach you something.

Every code block is labeled one of two ways:

- **`# GIVEN`** — copy this exactly. It's setup/plumbing, not the concept being taught.
- **`# YOUR TASK`** — I'll describe in plain English (and sometimes pseudocode) what the code needs to do. You write the actual Python. Don't move to the next block until yours runs and produces the expected output.

If you get stuck on a `YOUR TASK` block, that's normal — struggle for at least 15–20 minutes before looking anything up. That struggle is where the learning happens.

By the end of this manual you will have, **written mostly by your own hand**:
1. A script that detects 468 facial landmarks in any photo
2. A script that captures a photo from your webcam
3. A script that draws the landmarks on an image so you can visually verify they're correct
4. A working understanding of *why* each piece exists, not just that it works

---

## Part A — Theory (Read This Before Writing Any Code)

### A.1 What problem is Stage 1 actually solving?

Your final app needs to know, geometrically, what shape a person's face is — how wide the jaw is relative to the cheekbones, how long the face is relative to its width, and so on. A raw photo is just a grid of pixel values; it has no concept of "jaw" or "cheekbone." Stage 1's entire job is to convert a raw photo into a set of **precise, labeled coordinates** on the face, so that every later stage (shape classification, hair segmentation, generation) has something structured to work with.

Think of it as building the skeleton that every other stage will hang off of. If your landmarks are noisy or wrong, every downstream stage inherits that error — which is exactly why we isolate and validate this stage completely before touching anything else.

### A.2 What is MediaPipe, actually?

MediaPipe is an open-source perception framework built by Google. It ships a collection of **pretrained deep learning models** wrapped in an easy-to-call Python API, for tasks like face detection, hand tracking, pose estimation, and — what you'll use — **Face Mesh**.

Here's the important part conceptually: you are not writing or training a neural network in Stage 1. You are calling one that Google already trained on millions of face images. This is a completely normal and professional thing to do — nobody re-trains a face detector from scratch when a well-validated one already exists. The skill you're building here is **correctly integrating and validating a pretrained model**, which is its own real engineering competency.

### A.3 What's actually happening inside Face Mesh (the deep learning part)

Since you already know CNNs, here's the architecture in terms you'll recognize:

1. **Face detection sub-model (BlazeFace):** A lightweight CNN scans the full image and outputs a bounding box around each face it finds, plus a confidence score. This is a classic **object detection** task — you've likely seen this pattern before (CNN backbone → bounding box regression + objectness score).

2. **Landmark sub-model (the Mesh network):** The detected face region is cropped and fed into a second CNN. But this one does **not** end in a softmax classification head like the classifiers you're used to. Instead, its final layer is a **regression head** that outputs 468 × 3 = 1,404 continuous numbers — the (x, y, z) coordinates of every landmark, all at once, in a single forward pass.

This is the key conceptual shift for you: **classification outputs a probability distribution over discrete classes; this model outputs continuous spatial coordinates.** Same CNN machinery (convolutions, pooling/downsampling, dense layers), completely different output head and loss function (this was trained with something like mean-squared-error between predicted and ground-truth landmark positions, not cross-entropy).

3. **Temporal smoothing (for video/webcam):** When you feed it a video stream, MediaPipe also does light smoothing across frames so landmarks don't jitter wildly frame-to-frame. You don't need to implement this — it's built in.

### A.4 Why 468 points? Why not just find "the jaw" directly?

The model doesn't have a concept of "jaw" — it just knows point index 172 (for example) reliably lands on the left jaw contour, because that's how it was trained on annotated data. The **semantic meaning** of each point is something *we* impose afterward, by knowing the index map. This is why Part D of this manual has you explicitly identify and verify which indices correspond to which facial features — that mapping is knowledge you must hold, the model doesn't expose it directly.

### A.5 Coordinate systems — this trips up almost everyone the first time

MediaPipe returns landmark coordinates **normalized** between 0 and 1, relative to image width and height — not pixel positions.

- `landmark.x = 0.5` means "halfway across the image width," regardless of whether the image is 640px or 4000px wide.
- `landmark.y = 0.5` means "halfway down the image height."
- `landmark.z` is a **relative depth**, roughly scaled to the same units as x — smaller (more negative) z means closer to the camera than the face center, and it uses the midpoint between the eyes as a rough zero-reference. It is *not* real-world distance in centimeters. For Stage 1 and 2, you'll mostly ignore z and focus on x, y.

To draw these on an actual image or use them for pixel-based math, you must convert:

```
pixel_x = landmark.x * image_width
pixel_y = landmark.y * image_height
```

You will write this conversion yourself in Part D — it's a one-line calculation, but you need to *understand why* it's necessary, not just paste it.

### A.6 Why build this as an isolated, standalone module?

You're not wiring this into Stage 2 yet. You're going to prove, with your own eyes, that this module works correctly on multiple images before it becomes a dependency for anything else. This is standard practice in real engineering: **validate each unit before integrating it.** Skipping this step is the #1 reason beginner pipelines fail mysteriously three stages downstream — the bug was actually in stage one, just never checked in isolation.

---

## Part B — Environment Setup (Windows)

### B.1 Install Python

MediaPipe's official PyPI package currently supports Python 3.9 through 3.12 on Windows. If you already have Python installed, check your version first.

**GIVEN** — run this in PowerShell or Command Prompt to check:
```powershell
python --version
```

If you don't have Python, or have something outside 3.9–3.12, install **Python 3.11** from [python.org/downloads](https://www.python.org/downloads/) (3.11 is a safe, stable middle ground for compatibility with both MediaPipe and OpenCV).

**Important during install:** on the first installer screen, check the box **"Add python.exe to PATH"** before clicking Install. This is the single most common reason `python` isn't recognized in the terminal afterward.

### B.2 Create the project folder and a virtual environment

A virtual environment keeps this project's packages isolated from anything else on your machine — standard practice, not optional.

**GIVEN** — in PowerShell, navigate to where you want the project and run:
```powershell
mkdir haircut-recommender
cd haircut-recommender
python -m venv venv
```

Now activate it:
```powershell
venv\Scripts\activate
```

**Windows-specific gotcha:** if you get an error like *"running scripts is disabled on this system"*, PowerShell's execution policy is blocking it. Fix it by running this once (as your normal user, not admin needed for this scope):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then re-run the activate command. Once activated, you'll see `(venv)` prefixed in your terminal prompt.

### B.3 Install the required packages

**GIVEN:**
```powershell
pip install mediapipe opencv-python numpy
```

### B.4 Verify the install

**GIVEN** — create a file called `check_setup.py` in your project root with this content, and run it:
```python
import cv2
import mediapipe as mp
import numpy as np

print("OpenCV version:", cv2.__version__)
print("MediaPipe version:", mp.__version__)
print("NumPy version:", np.__version__)
```
Run with:
```powershell
python check_setup.py
```
You should see three version numbers print with no errors. If you get a DLL load error, it usually means you're missing the Visual C++ Redistributable — search "Microsoft Visual C++ Redistributable download" and install the x64 version, then retry.

**Checkpoint:** Do not proceed until this runs cleanly.

---

## Part C — Project Structure for This Stage

Inside your `haircut-recommender` folder, build this structure:

```
haircut-recommender/
├── venv/                          (already created)
├── data/
│   ├── raw/                       ← photos you test with go here
│   └── landmarks/                 ← saved landmark output goes here
├── stage1_landmarks/
│   ├── detect_landmarks.py        ← core detection logic
│   ├── visualize_landmarks.py     ← draws landmarks on an image
│   ├── webcam_capture.py          ← captures a photo from your webcam
│   ├── run_stage1.py              ← ties the above together
│   └── test_images/               ← a few sample face photos
└── check_setup.py
```

**YOUR TASK:** create these folders and empty files yourself using File Explorer or:
```powershell
mkdir data\raw, data\landmarks, stage1_landmarks\test_images
New-Item stage1_landmarks\detect_landmarks.py, stage1_landmarks\visualize_landmarks.py, stage1_landmarks\webcam_capture.py, stage1_landmarks\run_stage1.py
```

Drop 2–3 clear, front-facing photos of faces (yours, or any free stock photos) into `stage1_landmarks/test_images/` before continuing.

---

## Part D — Building `detect_landmarks.py`

This file's job: given an image file path, return the list of 468 landmarks.

### D.1 Imports and setup

**GIVEN:**
```python
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
```

### D.2 Initialize the Face Mesh model

**YOUR TASK.** MediaPipe's `FaceMesh` class needs to be instantiated with a few configuration options before you can use it. Look up `mp_face_mesh.FaceMesh(...)` and figure out what these three parameters mean, then set them appropriately for processing a **single static image** (not a video stream):

- `static_image_mode` — should this be `True` or `False` when working with individual photos rather than a continuous video feed? Think about what "static" implies.
- `max_num_faces` — how many faces should it look for? For this project, one is enough.
- `min_detection_confidence` — a threshold between 0 and 1. Look up what this controls and pick a reasonable value (research typical defaults, e.g. somewhere around 0.5).

Write this as a function:
```python
def create_face_mesh_detector():
    # your instantiation of mp_face_mesh.FaceMesh(...) goes here
    # return the object
    pass
```

**Why this matters conceptually:** you're configuring the inference behavior of a pretrained model — this is the same kind of configuration you'll do later with diffusion models (inference steps, guidance scale, etc.), so get comfortable reading a model's parameter docs and reasoning about what each one does before blindly setting values.

### D.3 Load an image and convert color format

**YOUR TASK.** OpenCV loads images in **BGR** color order by default, but MediaPipe expects **RGB**. Write a function that:
1. Takes an `image_path` string
2. Loads the image with `cv2.imread`
3. Converts it from BGR to RGB (look up the correct `cv2.cvtColor` flag — there's a specific constant for BGR→RGB)
4. Returns both the original image (for later drawing) and the RGB version (for feeding to the model)

```python
def load_image(image_path):
    # YOUR TASK: implement as described above
    pass
```

**Why this matters:** this is a classic beginner bug source — feeding a model an image in the wrong color order won't crash your program, it'll just silently produce wrong or degraded results. Always check a library's expected input format instead of assuming.

### D.4 Run detection and extract landmarks

**YOUR TASK.** This is the core of the module. Write a function `detect_landmarks(image_path)` that:

1. Calls your `create_face_mesh_detector()` from D.2
2. Calls your `load_image()` from D.3
3. Passes the RGB image to the detector's `.process()` method — look up this method's usage in MediaPipe's Face Mesh documentation
4. The result object has an attribute holding detected faces (look up what it's called — hint: it starts with `multi_face_`). Check whether it's `None` (no face found) — handle that case by returning `None` or an empty list, don't let it crash
5. If a face was found, take the first face's landmarks (since we set `max_num_faces=1`) and loop through them, building a **plain Python list** of `(x, y, z)` tuples — do not return MediaPipe's internal object directly; extract the raw numbers so this function's output doesn't force every other file to depend on MediaPipe's internal types

Pseudocode skeleton:
```python
def detect_landmarks(image_path):
    detector = create_face_mesh_detector()
    original_image, rgb_image = load_image(image_path)

    results = # call .process() on the detector with rgb_image

    if # no face found condition:
        return None

    face_landmarks = # get the first face's landmark list

    landmark_list = []
    for landmark in face_landmarks.landmark:
        # append (landmark.x, landmark.y, landmark.z) to landmark_list
        pass

    return landmark_list, original_image
```

### D.5 Test it

**YOUR TASK.** At the bottom of `detect_landmarks.py`, add a small test block that only runs when the file is executed directly (look up the `if __name__ == "__main__":` pattern if you haven't used it — it's standard Python for "only run this when the file is run directly, not when imported elsewhere").

Inside it: call `detect_landmarks()` on one of your test images, and print:
- How many landmarks were returned (should be exactly 468)
- The first 3 landmarks' raw values

Run it:
```powershell
python stage1_landmarks\detect_landmarks.py
```

**Checkpoint — do not proceed until you see exactly 468 landmarks printed with sensible-looking (x, y, z) values between roughly -0.5 and 1.5 (z can go slightly outside 0–1).**

---

## Part E — Building `visualize_landmarks.py`

You need to *see* the landmarks to trust them. Numbers alone don't tell you if something's subtly wrong (e.g., landmarks shifted, mirrored, or scaled incorrectly).

### E.1 Given: the drawing primitive

Here's how to draw **one** circle on an image with OpenCV, so you can see the pattern:
```python
# GIVEN — example of drawing a single point
cv2.circle(image, (pixel_x, pixel_y), radius=2, color=(0, 255, 0), thickness=-1)
```
`(pixel_x, pixel_y)` must be **integers** — `cv2.circle` will error or misbehave on floats, so you'll need to cast them.

### E.2 Your task: draw all 468 points

Write a function `draw_landmarks(image, landmark_list)` that:
1. Gets the image's height and width (look up the `.shape` attribute of an OpenCV image — note it returns `(height, width, channels)`, height first)
2. Loops through every `(x, y, z)` tuple in `landmark_list`
3. Converts each normalized `(x, y)` to pixel coordinates using the formula from Part A.5 — remember to cast to `int`
4. Draws a small circle at that pixel position on the image
5. Returns the modified image

```python
def draw_landmarks(image, landmark_list):
    height, width, _ = image.shape
    # YOUR TASK: loop and draw as described
    return image
```

### E.3 Save and inspect the output

**YOUR TASK.** In a test block, call `detect_landmarks()` on a test image, pass the results into `draw_landmarks()`, and save the output using `cv2.imwrite()` to somewhere like `data/landmarks/test_output.jpg`. Open the saved file and look at it.

**Checkpoint — what you should see:** a dense mesh of ~468 small dots tracing the contours of the eyes, eyebrows, nose, lips, and face outline, all correctly aligned to the actual facial features in the photo. If the dots are offset, mirrored, or clustered in the wrong place, go back and check your coordinate conversion in E.2 and your BGR/RGB handling in D.3.

---

## Part F — Building `webcam_capture.py`

### F.1 Given: opening the webcam and the capture loop

Webcam handling in OpenCV involves a specific loop pattern that's mostly boilerplate mechanics rather than a deep learning concept, so this part is largely given — but read it carefully, since you'll reuse this pattern constantly in later stages (e.g., a live "preview" mode for the final app).

```python
# GIVEN
import cv2

def capture_from_webcam(save_path):
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("Could not open webcam.")
        return None

    print("Press 's' to save a photo, or 'q' to quit without saving.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Webcam - press 's' to save, 'q' to quit", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # YOUR TASK: save the current frame to save_path using cv2.imwrite,
            # then break out of the loop
            pass
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return save_path
```

**YOUR TASK:** fill in the `'s'` key handler as described in the comment. Then add a test block at the bottom that calls `capture_from_webcam("data/raw/webcam_test.jpg")`.

Run it and confirm: a window opens showing your live camera feed, pressing `s` saves a frame to the specified path and closes the window, and pressing `q` closes without saving.

**Windows note:** if the webcam window doesn't respond to keypresses, click on the video window itself first to give it focus before pressing a key — this is an OpenCV quirk on Windows, not a bug in your code.

---

## Part G — Identifying Key Landmark Indices

You now have all 468 points, but you don't yet know *which index numbers* correspond to the jaw, cheekbones, forehead, and chin — you'll need these in Stage 2 for the ratio math.

### G.1 The reference

MediaPipe's Face Mesh uses a fixed, documented topology — the same index always maps to the same relative facial position across every face, because the model was trained to be consistent this way. Here are commonly used reference indices for the landmarks you'll need in Stage 2:

| Feature | Approx. landmark index |
|---|---|
| Chin (bottom of face) | 152 |
| Left jaw edge | 234 |
| Right jaw edge | 454 |
| Left cheekbone | 227 |
| Right cheekbone | 447 |
| Forehead center (top) | 10 |
| Left forehead corner | 71 |
| Right forehead corner | 301 |
| Left eye outer corner | 33 |
| Right eye outer corner | 263 |
| Nose tip | 1 |

Treat this table as a starting hypothesis, not gospel — you're going to verify it yourself in the next step rather than trusting it blindly, which is good practice whenever you use someone else's reference data.

### G.2 Your task: verify these indices visually

Write a small script (can be a new test block or a new file `verify_indices.py`) that:
1. Runs `detect_landmarks()` on a clear front-facing test photo
2. Draws **all** landmarks in a small, dim color (e.g., gray) as background context
3. Then draws **only** the indices from the table above in a bright, distinct color (e.g., red), each labeled with its index number using `cv2.putText`
4. Saves the result

**Checkpoint:** open the saved image and visually confirm each red dot actually sits where the table claims — chin dot on the chin, jaw dots on the jaw edges, etc. If any are off, that's useful information for Stage 2 and worth noting now rather than discovering it later when your face-shape ratios come out wrong.

---

## Part H — Assembling `run_stage1.py`

**YOUR TASK.** This script is the single entry point for Stage 1, tying everything together. It should:

1. Accept a command-line flag or a simple prompt asking: "Use an existing image, or capture from webcam?"
2. If webcam: call `capture_from_webcam()` to get an image, saving it into `data/raw/`
3. If existing image: accept a file path (you can hardcode a path to one of your test images for now, or use Python's `input()` to ask for one)
4. Call `detect_landmarks()` on whichever image was obtained
5. Call `draw_landmarks()` on the result
6. Save the visualized output into `data/landmarks/`
7. Print a summary: number of landmarks found, and the save location

This file should contain almost no *new* logic — it's orchestration of the pieces you already built. If you find yourself writing substantial new logic here, that's a sign it belongs in one of the other files instead.

---

## Part I — Troubleshooting (Windows-Specific)

| Symptom | Likely cause | Fix |
|---|---|---|
| `pip install mediapipe` fails, no matching distribution | Python version outside 3.9–3.12, or 32-bit Python | Reinstall 64-bit Python 3.10 or 3.11 |
| DLL load failed / import error on `cv2` or `mediapipe` | Missing Visual C++ Redistributable | Install the Microsoft VC++ x64 redistributable |
| `venv\Scripts\activate` refuses to run | PowerShell execution policy | Run the `Set-ExecutionPolicy` command from Part B.2 |
| Webcam window opens but freezes | Camera in use by another app (Zoom, Teams, Windows Camera app) | Close other apps using the camera |
| Landmarks look mirrored left-right | Some webcams/photos are pre-flipped | Note this now — you'll need consistent orientation in Stage 2's left/right jaw comparisons |
| `results.multi_face_landmarks` is always `None` | Face not well-lit, too small in frame, or at an extreme angle | Test with a clear, front-facing, well-lit photo first before debugging further |

---

## Part J — Glossary

- **CNN (Convolutional Neural Network):** the backbone architecture used inside MediaPipe's models — you already know this.
- **Regression head:** the final layer(s) of a network that output continuous numeric values (like coordinates) instead of class probabilities.
- **Bounding box:** the rectangle marking where a detected object (here, a face) sits in an image.
- **Normalized coordinates:** values scaled to a fixed range (here, 0–1) so they're independent of the original image's pixel dimensions.
- **Landmark / keypoint:** a specific, semantically meaningful point on a detected object (e.g., "nose tip").
- **Mesh topology:** the fixed structure/ordering of the 468 points — index 1 always means the same relative facial position across every face processed.
- **Inference:** running a trained model forward to get a prediction (as opposed to training it).
- **`static_image_mode`:** a MediaPipe setting controlling whether the model treats each frame independently (photos) or exploits temporal continuity (video).

---

## Part K — Definition of Done for Stage 1

Before moving to Stage 2, confirm all of the following:

- [ ] `check_setup.py` runs with no errors
- [ ] `detect_landmarks()` reliably returns exactly 468 landmarks on at least 2 different test photos
- [ ] `visualize_landmarks.py` produces an image where the dot mesh visibly and correctly traces the face in the photo
- [ ] `webcam_capture.py` successfully captures and saves a photo from your live camera
- [ ] You've visually verified the key indices from Part G actually land where the table claims, on your own face
- [ ] `run_stage1.py` runs end-to-end from either an image file or webcam capture, with no manual code edits needed between runs
- [ ] You can explain, without notes, the difference between normalized and pixel coordinates, and why the model outputs regression values instead of classes

Once every box is checked, you're ready for **Stage 2: Face Shape Classification**, where these landmarks get converted into geometric ratios and fed into a small MLP you'll build and train yourself.
