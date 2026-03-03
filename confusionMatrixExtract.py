import os
import cv2
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION PATHS
# ==========================================
MODEL_PATH = 'best_v5.pt'
IMAGES_PATH = r'D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_BaselineNN\datasets\FVP-Baseline.v5-v5.yolov8\valid\images'
LABELS_PATH = r'D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_BaselineNN\datasets\FVP-Baseline.v5-v5.yolov8\valid\labels'
OUTPUT_DIR = r'D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_BaselineNN\runs\detect\FYP_Matrix_Proof'

# Check your training results (F1_curve.png) to find the exact confidence YOLO used.
# If you aren't sure, 0.50 is the standard sweet spot for strict matrices.
CONFIDENCE_THRESHOLD = 0.614  
IOU_THRESHOLD = 0.50

model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. GROUND TRUTH PARSER (Handles Polygons)
# ==========================================
def get_ground_truth(label_path, img_width, img_height):
    gt_boxes, gt_classes = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    
                    # IF STANDARD BBOX (Exactly 4 coordinates: xc, yc, w, h)
                    if len(coords) == 4:
                        x_c, y_c, w, h = coords
                        x1 = (x_c - w / 2) * img_width
                        y1 = (y_c - h / 2) * img_height
                        x2 = (x_c + w / 2) * img_width
                        y2 = (y_c + h / 2) * img_height
                        
                    # IF POLYGON/SEGMENTATION (Many coordinates: x1, y1, x2, y2...)
                    else:
                        x_coords = coords[0::2] # Get all even indexes (X)
                        y_coords = coords[1::2] # Get all odd indexes (Y)
                        
                        # Find the outermost points to create a perfect bounding box
                        x1 = min(x_coords) * img_width
                        y1 = min(y_coords) * img_height
                        x2 = max(x_coords) * img_width
                        y2 = max(y_coords) * img_height
                        
                    gt_boxes.append([x1, y1, x2, y2])
                    gt_classes.append(class_id)
                    
    # Ensure correct tensor shapes even if the image has 0 labels
    if len(gt_boxes) == 0:
        return torch.empty((0, 4)), torch.empty((0,))
    return torch.tensor(gt_boxes), torch.tensor(gt_classes)

print("Starting strict matrix analysis with Polygon Support...")

# ==========================================
# 3. EVALUATE & MATCH
# ==========================================
for img_name in os.listdir(IMAGES_PATH):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    img_path = os.path.join(IMAGES_PATH, img_name)
    label_path = os.path.join(LABELS_PATH, img_name.rsplit('.', 1)[0] + '.txt')
    
    # Run model
    result = model.predict(img_path, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    
    # Use Ultralytics' internal image array to guarantee EXIF rotations match perfectly
    img = result.orig_img.copy()
    h, w = img.shape[:2]
    
    # Get True Labels (using the new polygon function)
    gt_boxes, gt_classes = get_ground_truth(label_path, w, h)
    
    # Get AI Predictions
    pred_boxes = result.boxes.xyxy.cpu()
    pred_classes = result.boxes.cls.cpu()
    pred_confs = result.boxes.conf.cpu()
    
    # Sort predictions by confidence (Highest first) to perfectly mimic YOLO's metric engine
    if len(pred_boxes) > 0:
        sort_idx = torch.argsort(pred_confs, descending=True)
        pred_boxes = pred_boxes[sort_idx]
        pred_classes = pred_classes[sort_idx]
        pred_confs = pred_confs[sort_idx]
        
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
    
    # ==========================================
    # 4. IOU MATH (Mapping to the Matrix)
    # ==========================================
    if len(gt_boxes) > 0 and len(pred_boxes) > 0:
        ious = box_iou(pred_boxes, gt_boxes) # Compare every box
        
        for p_idx in range(len(pred_boxes)):
            best_iou, best_gt_idx = ious[p_idx].max(0)
            
            # If overlap > 50% AND the ground truth hasn't been claimed by a stronger box yet
            if best_iou > IOU_THRESHOLD and not gt_matched[best_gt_idx]:
                gt_matched[best_gt_idx] = True
                
                # Check for Class Confusion (Inner matrix errors)
                if int(gt_classes[best_gt_idx]) != int(pred_classes[p_idx]):
                    true_name = CLASS_NAMES[int(gt_classes[best_gt_idx])]
                    pred_name = CLASS_NAMES[int(pred_classes[p_idx])]
                    folder = f"True_{true_name}_Pred_{pred_name}"
                    
                    error_img = img.copy()
                    cv2.rectangle(error_img, (int(gt_boxes[best_gt_idx][0]), int(gt_boxes[best_gt_idx][1])), 
                                       (int(gt_boxes[best_gt_idx][2]), int(gt_boxes[best_gt_idx][3])), (0, 255, 0), 2)
                    cv2.rectangle(error_img, (int(pred_boxes[p_idx][0]), int(pred_boxes[p_idx][1])), 
                                       (int(pred_boxes[p_idx][2]), int(pred_boxes[p_idx][3])), (0, 0, 255), 2)
                    
                    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)
                    cv2.imwrite(os.path.join(OUTPUT_DIR, folder, f"CONFUSION_{img_name}"), error_img)
            
            else:
                # FALSE POSITIVE: Drew a box over empty space (or a box that was already claimed)
                pred_name = CLASS_NAMES[int(pred_classes[p_idx])]
                folder = f"True_background_Pred_{pred_name}"
                
                error_img = img.copy()
                cv2.rectangle(error_img, (int(pred_boxes[p_idx][0]), int(pred_boxes[p_idx][1])), 
                                   (int(pred_boxes[p_idx][2]), int(pred_boxes[p_idx][3])), (0, 0, 255), 2)
                cv2.putText(error_img, f"Ghost: {pred_name}", (int(pred_boxes[p_idx][0]), int(pred_boxes[p_idx][1])-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)
                cv2.imwrite(os.path.join(OUTPUT_DIR, folder, f"GHOST_{p_idx}_{img_name}"), error_img)

    elif len(pred_boxes) > 0 and len(gt_boxes) == 0:
        # FALSE POSITIVE: Image was completely empty, but AI drew boxes
        for p_idx in range(len(pred_boxes)):
            pred_name = CLASS_NAMES[int(pred_classes[p_idx])]
            folder = f"True_background_Pred_{pred_name}"
            
            error_img = img.copy()
            cv2.rectangle(error_img, (int(pred_boxes[p_idx][0]), int(pred_boxes[p_idx][1])), 
                               (int(pred_boxes[p_idx][2]), int(pred_boxes[p_idx][3])), (0, 0, 255), 2)
            cv2.putText(error_img, f"Ghost: {pred_name}", (int(pred_boxes[p_idx][0]), int(pred_boxes[p_idx][1])-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)
            cv2.imwrite(os.path.join(OUTPUT_DIR, folder, f"GHOST_{p_idx}_{img_name}"), error_img)

    # FALSE NEGATIVES: Any ground truth box that we didn't match = Missed!
    for gt_idx, matched in enumerate(gt_matched):
        if not matched:
            true_name = CLASS_NAMES[int(gt_classes[gt_idx])]
            folder = f"True_{true_name}_Pred_background"
            
            error_img = img.copy()
            cv2.rectangle(error_img, (int(gt_boxes[gt_idx][0]), int(gt_boxes[gt_idx][1])), 
                               (int(gt_boxes[gt_idx][2]), int(gt_boxes[gt_idx][3])), (0, 255, 0), 2)
            cv2.putText(error_img, f"Missed: {true_name}", (int(gt_boxes[gt_idx][0]), int(gt_boxes[gt_idx][1])-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)
            cv2.imwrite(os.path.join(OUTPUT_DIR, folder, f"MISSED_{gt_idx}_{img_name}"), error_img)

print(f"\nDone! Clean results are waiting in: {OUTPUT_DIR}")