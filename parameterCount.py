from ultralytics import YOLO

# Load your model (replace with your 'best_v5.pt')
model = YOLO("best_v5.pt")

# This prints a table with Layer index, Name, and Parameter count
model.info(detailed=True)