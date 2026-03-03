import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "best_v5.pt"      
VIDEO_PATH = "Test Video.mp4"  
OUTPUT_PATH = "counted_output.avi"

# Line Coordinates (x1, y1) to (x2, y2)
# Change this based on your video resolution and line orientation
# E.g., for a horizontal line near the bottom: (100, 500) to (1800, 500)
LINE_START = (600, 0)
LINE_END = (600, 1080)

model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

# ==========================================
# 2. INITIALIZATION
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

# Dictionary to store {tracker_id: {class_name, previous_center_point}}
track_history = {}
# Final tally per class
counts = {name: 0 for name in CLASS_NAMES.values()}

def check_intersection(p1, p2, p3, p4):
    """
    Checks if line segment (p1, p2) intersects with line segment (p3, p4).
    Uses standard cross-product logic.
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

print("Starting real-time tracking and counting...")

# ==========================================
# 3. VIDEO PROCESSING LOOP
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Run YOLO with integrated tracking
    # persist=True keeps tracking IDs between frames
    results = model.track(frame, persist=True, show=False, conf=0.5)[0]
    
    # Draw the counting line on the frame
    cv2.line(frame, LINE_START, LINE_END, (0, 255, 255), 3)

    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.int().cpu().numpy()
        class_ids = results.boxes.cls.int().cpu().numpy()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2) # Get the Y center too
            current_center = (center_x, center_y)

            # Draw bounding box and center point (optional, but helpful for visualization)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, current_center, 4, (0, 0, 255), -1)

            # Get the class name
            class_name = CLASS_NAMES[class_id]

            # --- COUNTING LOGIC ---
            if track_id in track_history:
                previous_center = track_history[track_id]['prev_center']
                
                # Check if the line segment between the previous center and current center
                # intersects with our counting line
                if check_intersection(LINE_START, LINE_END, previous_center, current_center):
                    # Hasn't been counted yet?
                    if not track_history[track_id].get('counted', False):
                        counts[class_name] += 1
                        track_history[track_id]['counted'] = True
                        
                        # Change line color briefly to show a count happened
                        cv2.line(frame, LINE_START, LINE_END, (0, 0, 255), 3)

            # Update the track history for this ID
            track_history[track_id] = {
                'class_name': class_name,
                'prev_center': current_center,
                'counted': track_history.get(track_id, {}).get('counted', False)
            }

    # Display the counts on the frame
    y_offset = 30
    for class_name, count in counts.items():
        if count > 0: # Only show classes that have been counted
            text = f"{class_name}: {count}"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_offset += 35

    # Write the frame to the output video
    video_writer.write(frame)

    # --- DISPLAY THE VIDEO ---
    cv2.imshow("YOLOv8 Counting", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# 4. CLEANUP
# ==========================================
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("Processing complete.")
print("Final Counts:")
for name, count in counts.items():
    if count > 0:
        print(f"{name}: {count}")