import cv2
import numpy as np
import face_recognition
import os
import time

# ===== Verify GPU Support =====
import dlib
if not dlib.DLIB_USE_CUDA:
    print("⚠️ WARNING: dlib was not compiled with CUDA support!")
    print("👉 Install dlib with CUDA: https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf ")
else:
    print("✅ GPU ACCELERATION ENABLED: dlib using CUDA")

# ===== Settings =====
MATCH_THRESHOLD = 0.6
PROCESS_EVERY_N_FRAMES = 1      # Use 2 if GPU is overloaded
DISPLAY_WIDTH = 800
MIN_FACE_SIZE_PX = 60

# ===== Paths =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "image")


def load_known_faces():
    """Load known faces using CNN (GPU) for best encoding quality"""
    known_encodings = []
    known_names = []

    print(f"\n🔍 Loading reference images from '{IMAGE_DIR}'...\n")

    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Directory not found: {IMAGE_DIR}")
        return [], []

    for filename in os.listdir(IMAGE_DIR):
        ext = filename.lower()
        if not (ext.endswith('.jpg') or ext.endswith('.jpeg') or ext.endswith('.png')):
            continue

        path = os.path.join(IMAGE_DIR, filename)
        try:
            img = face_recognition.load_image_file(path)

            # 🔥 Use CNN (runs on GPU if available)
            face_locs = face_recognition.face_locations(img, model="cnn")
            if len(face_locs) == 0:
                print(f"⚠️ No face detected in {filename}")
                continue

            encoding = face_recognition.face_encodings(img, face_locs)[0]
            name = os.path.splitext(filename)[0]

            known_encodings.append(encoding)
            known_names.append(name)
            print(f"✅ {name} loaded from {filename}")

        except Exception as e:
            print(f" Error loading {filename}: {str(e)}")

    if not known_encodings:
        print(" No valid faces loaded! Add .jpg/.png files to 'image/' folder.")
        return [], []

    print(f"\n Loaded {len(known_encodings)} person(s): {', '.join(set(known_names))}\n")
    return known_encodings, known_names


def live_multi_face_recognition():
    # Load known faces (GPU-encoded)
    known_encodings, known_names = load_known_faces()
    if not known_encodings:
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

    frame_count = 0
    prev_time = time.time()
    fps = 0  # ✅ Initialize fps here to avoid UnboundLocalError

    print("🎥 Starting GPU-accelerated face recognition...")
    print("🔐 Press ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display_frame = frame.copy()

        frame_count += 1
        current_time = time.time()

        # ✅ Calculate FPS every second
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - prev_time)
            prev_time = current_time
            frame_count = 0

        # Only process every N frames to reduce GPU load
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            # Just display without processing
            cv2.putText(display_frame, "Live View", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("🔐 Face Recognition [GPU Mode]", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            continue

        # 🔥 GPU-Accelerated Face Detection (CNN)
        try:
            face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        except Exception as e:
            print(f"❌ CNN face detection failed: {e}")
            face_locations = []

        recognized_count = 0

        # Process each detected face
        for top, right, bottom, left in face_locations:
            face_height = bottom - top
            if face_height < MIN_FACE_SIZE_PX:
                continue

            try:
                # Extract encoding using full-resolution frame
                face_encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])[0]

                # Compare against all known faces
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_idx = np.argmin(distances)
                best_distance = distances[best_idx]

                # Match decision
                if best_distance < MATCH_THRESHOLD:
                    name = known_names[best_idx]
                    color = (0, 255, 0)  # Green
                    label = f"{name}"
                    recognized_count += 1
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red
                    label = "Unknown"

            except Exception as e:
                label = "?"
                color = (255, 165, 0)  # Orange
                print(f"⚠️ Encoding failed: {e}")

            # Draw bounding box and label
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(display_frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show stats (✅ now safe to use 'fps')
        cv2.putText(display_frame, f"FPS: {int(fps)} | Faces: {len(face_locations)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("🔐 Face Recognition [GPU Mode]", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_multi_face_recognition()