import cv2
from ultralytics import YOLO
import os

# 1. Configuration
model = YOLO('runs/detect/train2/weights/best.pt')
video_path = 'Part 2.mp4'
TRUE_PART_NAME = 'Part 2' # The part actually on the line

# 2. Setup Folders
output_dir = 'cross_class_errors'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Run inference
    results = model(frame, conf=0.4) # Slightly higher confidence to ignore weak bg noise
    
    for result in results:
        # We ignore len(result.boxes) == 0 because we don't care about bg/missing detections here
        
        for box in result.boxes:
            pred_class = model.names[int(box.cls[0])]
            
            # LOGIC: Only save if a part is detected AND it is the WRONG part
            # We also ensure the prediction isn't 'background' (if 'background' is a named class)
            if pred_class != TRUE_PART_NAME and pred_class != 'background':
                
                # Draw the incorrect label on the frame for easier manual inspection
                label_text = f"ERR: Predicted {pred_class} (True: {TRUE_PART_NAME})"
                cv2.putText(frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                save_path = f"{output_dir}/frame_{frame_idx}_MISLABEL_{pred_class}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"Captured Cross-Class Error: {pred_class} at frame {frame_idx}")

    frame_idx += 1

cap.release()
print("Done. Any images in 'cross_class_errors' are confirmed mislabels.")