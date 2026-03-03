import os
import shutil
from ultralytics import YOLO

# 1. Paths - Update these to your exact folders
model_path = r'D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_BaselineNN\runs\detect\train5\weights\best.pt'
val_images_path = r'D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_BaselineNN\datasets\FVP-Baseline.v5-v5.yolov8\valid\images'
val_labels_path = r'D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_BaselineNN\datasets\FVP-Baseline.v5-v5.yolov8\valid\labels'
output_dir = r'D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_BaselineNN\runs\detect\true_validation_errors'

model = YOLO(model_path)
results = model.predict(source=val_images_path, conf=0.25, save=False)

os.makedirs(output_dir, exist_ok=True)

for result in results:
    img_name = os.path.basename(result.path)
    label_name = img_name.rsplit('.', 1)[0] + '.txt'
    label_file = os.path.join(val_labels_path, label_name)
    
    # Check if a ground truth label actually exists
    has_label = os.path.exists(label_file) and os.path.getsize(label_file) > 0
    detected_something = len(result.boxes) > 0
    
    is_mistake = False
    
    # MISTAKE 1: False Negative (Label exists, but model missed it)
    if has_label and not detected_something:
        is_mistake = True
        error_type = "MISS"
        
    # MISTAKE 2: False Positive (No label, but model hallucinated a part)
    elif not has_label and detected_something:
        is_mistake = True
        error_type = "GHOST"
        
    # MISTAKE 3: Low Confidence (Model found it but is "weak")
    elif detected_something and any(box.conf < 0.45 for box in result.boxes):
        is_mistake = True
        error_type = "WEAK"

    if is_mistake:
        result.save(filename=os.path.join(output_dir, f"{error_type}_{img_name}"))

print(f"Done! Errors saved in: {output_dir}")