import random
import cv2
import numpy as np
from ultralytics import YOLO

# Class names for plastic bottle detection
class_list = ['plastic_box']  # Update this based on your dataset

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load your trained model - UPDATE THIS PATH after training
#model = YOLO(r"C:\Users\Shriya\AppData\Local\Programs\Python\Python310\plastic_bottle_runs\plastic_bottle_detect3\weights\best.pt")
model = YOLO(r"C:\Users\Shriya\Desktop\Extracurricular\GitHub\Plastic_Bottle_Detection\plastic_bottle_runs\plastic_bottle_detect3\weights\best.pt")

# Alternative: Use a pre-trained model if you haven't trained yet
# model = YOLO('yolov8n.pt')  # This will detect general objects, not specifically plastic bottles

# Video capture from camera (0 for default camera, 1 for external camera)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to quit, 's' to save current frame")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame. Exiting...")
        break
    
    # Predict on the frame
    detect_params = model.predict(source=[frame], conf=0.20, save=False)
    
    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    
    # Draw detections on frame
    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            
            # Get detection info
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]
            
            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )
            
            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            label = f"{class_list[int(clsID)]} {round(conf, 2)}"
            cv2.putText(
                frame,
                label,
                (int(bb[0]), int(bb[1]) - 10),
                font,
                0.8,
                (255, 255, 255),
                2,
            )
            
            # Print detection info
            print(f"Detected: {class_list[int(clsID)]} with confidence {round(conf, 2)}")
    
    # Display the frame
    cv2.imshow("Plastic Bottle Detection", frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save current frame
        cv2.imwrite(f"detection_frame_{random.randint(1000, 9999)}.jpg", frame)
        print("Frame saved!")

# Release everything
cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed.")
