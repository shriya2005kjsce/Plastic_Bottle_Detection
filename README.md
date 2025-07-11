# Plastic_Bottle_Detection
Offline Real-time object detection system using YOLOv8 to identify plastic bottles in images or video streams
🔁 COMPLETE ROBOFLOW TO YOLO WORKFLOW
________________________________________
📤 1. Upload Images to Roboflow
•	Go to https://roboflow.com and sign in or create an account.
•	Create a new project → choose type (object detection), name it (e.g., plastic_bottle_detect), and select the annotation format (e.g., VOC, YOLO).
•	Upload your images (either raw or annotated).
•	Supported formats: images with .jpg, .png, and annotations like .xml, .txt, .json.
________________________________________
✏️ 2. Annotate Images (if not already annotated)
•	Roboflow opens the annotation tool where you draw bounding boxes around objects.
•	Label each object (e.g., plastic_bottle, plastic_box).
•	You can also invite teammates to help annotate.
________________________________________
➕ 3. Add to Dataset
•	Once images are annotated, click “Add images to dataset”.
•	Choose “Train, Test, and Validation Split” (Roboflow suggests 70/20/10).
•	Click “Save & Continue”.
________________________________________
⚙️ 4. Generate Dataset Version
•	Roboflow allows preprocessing like:
o	Resize (e.g., 640x640)
o	Augmentation (rotation, flip, brightness, noise, etc.)
•	Click “Generate” to create a new version of your dataset.
________________________________________
🧠 5. Train the Model
You have two options:
🔵 Option 1: Train on Roboflow
•	Click "Train Model".
•	Roboflow will use Roboflow Universe (cloud GPU).
•	Wait until training finishes (usually 10–30 minutes).
•	Download the best checkpoint directly.
🔶 Option 2: Train Locally Using YOLOv8
•	Click "Download Dataset" → choose "YOLOv8 PyTorch" format.
This will download:
•	train, valid, test folders with images and .txt labels.
•	data.yaml for YOLO.
🏋️ 6. Train YOLOv8 Model Locally
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
After training:
•	Weights saved in runs/detect/train/weights/best.pt
FinalThen Detecting .py Code
Step	Action
1	Upload images to Roboflow
2	Annotate objects
3	Add images to dataset
4	Generate version (resize, augment)
5	Train model (on Roboflow or locally)
6	Download dataset as YOLOv8 format
7	Train with ultralytics YOLO
8	Use .pt file in Python script

