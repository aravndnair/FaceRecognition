# 🎯 Real-Time Face Recognition with CNN and CUDA (Webcam-Based)

This project is a **real-time face recognition system** that uses your **webcam** to automatically detect and verify faces. It relies fully on a **CNN-based face detection model** for higher accuracy and uses **CUDA and cuDNN** to accelerate performance on NVIDIA GPUs.

---

## 🚀 Features

- 🧠 **CNN-only face detection and recognition** (no HOG fallback)
- 🎥 Fully **automated live verification** from webcam — no key press needed
- ⚡ Accelerated using **CUDA + cuDNN**
- ✅ Matches real-time webcam input against a **reference image**
- 🖼️ Displays results with face rectangles and match info in real-time

---

## 🧠 Tech Stack

- `Python`
- `face_recognition`
- `OpenCV`
- `NumPy`
- `CUDA` + `cuDNN`
- `dlib` (compiled with CUDA for CNN support)

---

## 🛠️ Requirements

- **Python 3.7+**
- **NVIDIA GPU** with CUDA support
- Installed:
  - `face_recognition`
  - `opencv-python`
  - `numpy`
  - `dlib` (with GPU/CUDA support)
- **CUDA Toolkit** and **cuDNN** installed properly

> ⚠️ `dlib` must be compiled with CUDA for CNN to work. Without it, the program will fail to detect faces.

---

## 📸 How It Works

1. A **reference face** is loaded and encoded when the program starts.
2. The webcam continuously captures frames.
3. Each frame is passed through a **CNN-based face detection model**.
4. If a face is detected:
   - It is encoded.
   - Compared with the reference face using `face_distance`.
   - A match decision is made based on a threshold.
5. The match result, method used, and face bounding boxes are shown on screen.

---

## 🧪 Usage

1. Place a reference image in the `image/` folder (e.g., `carlos.jpg`).
2. Update the `REFERENCE_IMAGE` path in `main.py` if needed.
3. Run the script:

