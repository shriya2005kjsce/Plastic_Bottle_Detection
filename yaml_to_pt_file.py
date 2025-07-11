from ultralytics import YOLO
import os

# Set the path to your downloaded dataset
dataset_path = r"C:\Users\Shriya\Desktop\Extracurricular\GitHub\Plastic_Bottle_Detection\Plastic_Bottle.v2-roboflow-instant-2--eval-.yolov8 (2)"

# Load a pre-trained YOLO model (you can use yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
model = YOLO('yolov8n.pt')  # This will download automatically if not present

# Train the model
results = model.train(
    data=os.path.join(dataset_path, r"C:\Users\Shriya\Desktop\Extracurricular\GitHub\Plastic_Bottle_Detection\Plastic_Bottle.v2-roboflow-instant-2--eval-.yolov8 (2)\data.yaml"),  # path to dataset YAML file
    epochs=50,  # number of training epochs
    imgsz=640,  # image size
    batch=16,   # batch size (reduce if you get memory errors)
    device='cpu',  # use 'cuda' if you have GPU, 'cpu' for CPU only
    project='plastic_bottle_runs',  # project name
    name='plastic_bottle_detect',   # experiment name
    save=True,  # save checkpoints
    plots=True  # save training plots
)

# The trained model will be saved as 'best.pt' in the runs folder
print("Training completed!")
print(f"Best model saved at: {results.save_dir}/weights/best.pt")
