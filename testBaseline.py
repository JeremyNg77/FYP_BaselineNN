from ultralytics import YOLO

# You MUST wrap the execution in this block on Windows
if __name__ == '__main__':
    
    # 1. Load your fully trained model
    model = YOLO(r'D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_BaselineNN\runs\detect\train10\weights\best.pt')

    # 2. Run the evaluation using default YOLOv8 hyperparameters
    print("Starting Final Evaluation with default settings...")
    metrics = model.val(
        data=r'D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_BaselineNN\datasets\FVP-Baseline.v17-vfinal.yolov8\data.yaml'
    )

    # 3. Print Overall Metrics
    print("\n" + "="*45)
    print("OVERALL METRICS")
    print("="*45)
    print(f"mAP@50:     {metrics.box.map50:.4f}")
    print(f"mAP@50-95:  {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")

    # 4. Print Class-Specific Metrics
    print("\n" + "="*45)
    print("CLASS-SPECIFIC METRICS")
    print("="*45)
    print(f"{'Class':<12} | {'Metric':<15} | {'Value':<10}")
    print("-" * 45)
    
    # Extract the internal class indices evaluated during validation
    evaluated_classes = metrics.box.ap_class_index
    
    # Loop through each evaluated class to get its specific scores
    for i, class_idx in enumerate(evaluated_classes):
        # Map the index to the actual string name (e.g., "Part 1")
        class_name = model.names[class_idx]
        
        # Extract the specific metrics using the index 'i'
        class_map50 = metrics.box.ap50[i]
        class_map_50_95 = metrics.box.ap[i]
        class_p = metrics.box.p[i]
        class_r = metrics.box.r[i]
        
        print(f"{class_name:<12} | {'mAP@50':<15} | {class_map50:.4f}")
        print(f"{'':<12} | {'mAP@50-95':<15} | {class_map_50_95:.4f}")
        print(f"{'':<12} | {'Precision':<15} | {class_p:.4f}")
        print(f"{'':<12} | {'Recall':<15} | {class_r:.4f}")
        print("-" * 45)