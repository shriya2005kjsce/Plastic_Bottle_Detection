# 🧴 Plastic Bottle Detection

Offline real-time object detection system using **YOLOv8** to identify plastic bottles in images or video streams. Trained using a custom Roboflow dataset.

---

## 📦 Download Project Dataset
[🔗 Roboflow Dataset (Version 2)](https://app.roboflow.com/plastic-fqcm2/plastic_bottle-nwhv0/2)

---

## 🔁 Complete Roboflow → YOLOv8 Workflow

### 📤 1. Upload Images to Roboflow
- Visit [roboflow.com](https://roboflow.com) and log in.
- Create a new project (object detection) and choose a name and format (e.g., VOC, YOLO).
- Upload your `.jpg`, `.png`, and annotations (`.xml`, `.txt`, `.json`).

### ✏️ 2. Annotate Images
- Use Roboflow’s annotation tool to draw bounding boxes.
- Label each object (e.g., `plastic_bottle`, `plastic_box`).
- Optionally invite team members to help.

### ➕ 3. Add to Dataset
- After annotation, click **“Add images to dataset”**.
- Split into Train/Test/Validation (recommended: 70/20/10).
- Click **“Save & Continue”**.

### ⚙️ 4. Generate Dataset Version
- Apply preprocessing (resize to 640x640, augmentations like flip, brightness, etc.).
- Click **“Generate”** to create a dataset version.

---

## 🧠 5. Train the Model

### 🔵 Option 1: Train on Roboflow (Cloud)
- Click **“Train Model”** on Roboflow.
- Wait ~10–30 mins for training to finish.
- Download the best checkpoint (`best.pt`).

### 🔶 Option 2: Train Locally Using YOLOv8
- Download the dataset in **YOLOv8 PyTorch** format.
- You’ll receive:
  - `train`, `valid`, `test` folders with images and labels
  - `data.yaml` file for YOLO

### 🏋️ 6. Train YOLOv8 Locally

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
