from ultralytics import YOLO

# Load your newly trained model
model = YOLO("runs/detect/train2/weights/best.pt")

# Run detection on a new image or video
results = model.predict(
    source="D:/Jeremy/UoN/Year 4/Final Year Project Coding/FYP_BaselineNN/Test Video.mp4",
    save=True, 
    show=False,  # This line enables the live pop-up window
    conf=0.5
)